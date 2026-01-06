from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from scipy import stats

import matplotlib.pyplot as plt
try:
    # when running `python -m src.main ...`
    from .data_loading import load_raw_data  # type: ignore
    from .preprocessing import preprocess_data  # type: ignore
except Exception:  # pragma: no cover
    # when running `python src/main.py ...`
    from data_loading import load_raw_data  # type: ignore
    from preprocessing import preprocess_data  # type: ignore


# ============================================================
# RESULTS FOLDERS 
# ============================================================
    
# Results directories 
ADV_ROOT = Path("results") 
ADV_DIR_TRAIN = ADV_ROOT / "training"
ADV_DIR_PRED = ADV_ROOT / "predictions"
ADV_DIR_EV = ADV_ROOT / "ev"
ADV_DIR_TEST = ADV_ROOT / "test_set"
ADV_DIR_SIG = ADV_ROOT / "significance"
ADV_DIR_FIG = ADV_ROOT / "figures"
ADV_DIR_EFF = ADV_ROOT / "efficiency"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def american_to_decimal(odds: float) -> float:
    if odds is None or pd.isna(odds) or odds == 0:
        return float("nan")
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / (-odds))


def moneyline_to_prob(ml: float) -> float:
    # Market-implied probability from American odds.
    if ml > 0:
        return 100 / (ml + 100)
    return -ml / (-ml + 100)


def expected_value(model_prob: float, moneyline: float) -> float:
    """
    EV for a 1-unit stake:
      EV = p * payout_if_win - (1 - p)
    where payout_if_win excludes stake:
      +odds: odds/100
      -odds: 100/abs(odds)
    """
    if moneyline > 0:
        payout = moneyline / 100
    else:
        payout = 100 / abs(moneyline)
    return model_prob * payout - (1 - model_prob)


def _ensure_cleaned_csv() -> Path:
    p = Path("data") / "cleaned_data.csv"
    if p.exists():
        return p

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    # IMPORTANT: create a stable row_id so predictions can be merged back safely
    # This avoids silent misalignment when advanced feature building drops rows
    df_clean = df_clean.reset_index(drop=True)
    df_clean["row_id"] = np.arange(len(df_clean))

    p.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(p, index=False)
    return p


# ============================================================
# MODEL
# ============================================================

def build_advanced_features(path: str = "data/cleaned_data.csv") -> pd.DataFrame:
    """
    The advanced model expands the feature space beyond raw implied probabilities, it must only use information available pre-game to avoid leakage.
    """
    df = pd.read_csv(path)

    # ------------------------------------------------------------
    # remove post-game leakage columns because they directly influence the game outcome and make the AUC = 1 which is impossible
    # ------------------------------------------------------------
    leakage_cols = ["score", "opponentScore"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # Bookmaker implied probabilities (pre-game)
    df["market_prob"] = df["moneyLine"].apply(moneyline_to_prob)
    if "opponentMoneyLine" in df.columns:
        df["opp_market_prob"] = df["opponentMoneyLine"].apply(moneyline_to_prob)

    # Nonlinear transforms (still pre-game)
    df["market_prob_sq"] = df["market_prob"] ** 2
    mp = df["market_prob"].clip(1e-9, 1 - 1e-9)
    df["market_prob_logit"] = np.log(mp / (1 - mp))

    # Home indicator if present (pre-game)
    if "home/visitor" in df.columns:
        df["is_home"] = (df["home/visitor"].astype(str).str.lower() == "home").astype(int)

    # Keep row_id for safe merging later
    if "row_id" not in df.columns:
        # If user already had cleaned_data.csv without row_id, create it defensively
        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df))

    # Select columns: we allow numeric predictors + season + row_id + win
    keep_cols: List[str] = []
    for c in df.columns:
        if c in {"row_id", "season", "win"}:
            keep_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            keep_cols.append(c)

    df = df[keep_cols].copy()
    df["win"] = df["win"].astype(int)

    # Drop rows with missing values in predictors/target
    df = df.dropna().copy()

    return df

def train_test_split_by_season(df: pd.DataFrame, n_test_seasons: int = 2) -> Tuple[pd.Index, pd.Index]:
    """
    As it can be seen here, in this part, the train and test sets are split by season. This avoids leakage across time and mimics a realistic forecasting setup
      (train on earlier seasons, test on later seasons).
    """
    if "season" not in df.columns:
        raise ValueError("Advanced features must contain a 'season' column for time-based split.")

    seasons = sorted(df["season"].unique())
    if len(seasons) <= n_test_seasons:
        raise ValueError(f"Not enough distinct seasons ({len(seasons)}) for a {n_test_seasons}-season test split.")

    test_seasons = set(seasons[-n_test_seasons:])
    is_test = df["season"].isin(test_seasons)

    test_idx = df[is_test].index
    train_idx = df[~is_test].index

    return train_idx, test_idx


def run_advanced_train() -> None:
    """
    This part will train the 3 different models (LR, RF, GB), select the best model based on ROC AUC and save row probabilities for all games 
    It is necessary to have this because we need to know which of the 3 models should be used for the rest of the project 
    """
    ensure_dir(ADV_DIR_TRAIN)
    ensure_dir(ADV_DIR_PRED)

    df = build_advanced_features("data/cleaned_data.csv")

    y = df["win"]
    X = df.drop(columns=["win"])

    train_idx, test_idx = train_test_split_by_season(df, n_test_seasons=2)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    rows: List[Dict[str, float]] = []
    fitted: Dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        row = {
            "model": name,
            "acc_train": float(accuracy_score(y_train, train_proba >= 0.5)),
            "acc_test": float(accuracy_score(y_test, test_proba >= 0.5)),
            "roc_train": float(roc_auc_score(y_train, train_proba)),
            "roc_test": float(roc_auc_score(y_test, test_proba)),
            "logloss_train": float(log_loss(y_train, train_proba)),
            "logloss_test": float(log_loss(y_test, test_proba)),
            "brier_train": float(brier_score_loss(y_train, train_proba)),
            "brier_test": float(brier_score_loss(y_test, test_proba)),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("roc_test", ascending=False)
    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted[str(best_model_name)]

    # Save training summary
    out_txt = ADV_DIR_TRAIN / "model_comparison.txt"
    out_txt.write_text(results_df.to_string(index=False) + "\n")
    print("[ADV TRAIN] Saved:", out_txt)
    print("[ADV TRAIN] Best model by test ROC AUC:", best_model_name)

    # Predict for all rows to support EV scripts 
    best_proba_full = best_model.predict_proba(X)[:, 1]

    set_col = pd.Series("train", index=df.index)
    set_col.loc[test_idx] = "test"

    preds_df = pd.DataFrame({"logit_prob": best_proba_full, "set": set_col, "model": best_model_name})
    out_csv = ADV_DIR_PRED / "advanced_predictions.csv"
    preds_df.to_csv(out_csv, index=False)
    print("[ADV TRAIN] Saved:", out_csv)
    print(f"\nBest model selected: {best_model_name}\n")


def run_advanced_ev_full() -> None:
    """
    - uses cleaned_data + advanced predictions
    - computes EV, edge, profit
    - buckets EV into deciles
    - saves row-level EV results + bucket summary
    """
    ensure_dir(ADV_DIR_EV)

    pred_path = ADV_DIR_PRED / "advanced_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing advanced predictions: {pred_path}. Run `advanced-train` first."
        )

    # --- Load cleaned data (raw)
    cleaned_path = _ensure_cleaned_csv()
    raw = pd.read_csv(cleaned_path)

    # --- Load predictions prevents NameError
    preds = pd.read_csv(pred_path)

    # Backward compatibility to ensure row_id exists on both sides
    if "row_id" not in raw.columns:
        raw = raw.reset_index(drop=True).copy()
        raw["row_id"] = np.arange(len(raw))

    if "row_id" not in preds.columns:
        preds = preds.reset_index(drop=True).copy()
        preds["row_id"] = np.arange(len(preds))

    # If set column is missing, still allow full EV but skip train/test filtering
    if "set" not in preds.columns:
        preds["set"] = "unknown"

    # --- Merge safely (never rely on row order)
    df = raw.merge(
        preds[["row_id", "logit_prob", "set"]],
        on="row_id",
        how="inner",
    ).copy()

    if df.empty:
        raise ValueError("Merge produced 0 rows. Check that raw and preds refer to the same dataset/order.")

    # EV + edge
    df["market_prob"] = df["moneyLine"].apply(moneyline_to_prob)
    df["EV"] = df.apply(lambda r: expected_value(r["logit_prob"], r["moneyLine"]), axis=1)
    df["edge"] = df["logit_prob"] - df["market_prob"]

    # Profit for 1-unit stake
    def profit_row(r: pd.Series) -> float:
        if r["win"] == 1:
            if r["moneyLine"] > 0:
                return r["moneyLine"] / 100
            return 100 / abs(r["moneyLine"])
        return -1.0

    df["profit"] = df.apply(profit_row, axis=1)

    # EV buckets 
    df["ev_bucket"] = pd.qcut(df["EV"].rank(method="first"), 10, labels=False)

    bucket_results = (
        df.groupby("ev_bucket")
        .agg(
            avg_EV=("EV", "mean"),
            ROI=("profit", "mean"),
            avg_edge=("edge", "mean"),
            count=("profit", "count"),
        )
        .reset_index()
    )

    # Save outputs
    out_csv = ADV_DIR_EV / "ev_results_full.csv"
    df.to_csv(out_csv, index=False)

    out_txt = ADV_DIR_EV / "ev_bucket_summary_full.txt"
    out_txt.write_text(bucket_results.to_string(index=False) + "\n")

    print("[ADV EV FULL] Saved:", out_csv)
    print("[ADV EV FULL] Saved:", out_txt)



def run_advanced_ev_testset() -> None:
    """
    - restricted to rows labeled set=='test'
    - computes EV, edge, profit
    - buckets EV into deciles
    - runs Spearman between avg_EV and ROI at bucket level
    """
    ensure_dir(ADV_DIR_TEST)

    pred_path = ADV_DIR_PRED / "advanced_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing advanced predictions: {pred_path}. Run `advanced-train` first."
        )

    # Load cleaned data (raw)
    cleaned_path = _ensure_cleaned_csv()
    raw = pd.read_csv(cleaned_path)

    # Load predictions (preds) to prevent the NameError
    preds = pd.read_csv(pred_path)

    # Backward compatibility to ensure row_id exists on both sides
    if "row_id" not in raw.columns:
        raw = raw.reset_index(drop=True).copy()
        raw["row_id"] = np.arange(len(raw))

    if "row_id" not in preds.columns:
        preds = preds.reset_index(drop=True).copy()
        preds["row_id"] = np.arange(len(preds))

    # We need the train/test split label to filter the test set
    if "set" not in preds.columns:
        raise ValueError("Predictions file missing 'set' column. Re-run advanced-train.")

    # --- Merge safely (never rely on row order)
    df = raw.merge(
        preds[["row_id", "logit_prob", "set"]],
        on="row_id",
        how="inner",
    ).copy()

    # Keep only test-set rows
    df = df[df["set"] == "test"].copy()
    if df.empty:
        raise ValueError("No rows labeled set=='test'. Re-run advanced-train or check split logic.")

    # EV + edge
    df["market_prob"] = df["moneyLine"].apply(moneyline_to_prob)
    df["EV"] = df.apply(lambda r: expected_value(r["logit_prob"], r["moneyLine"]), axis=1)
    df["edge"] = df["logit_prob"] - df["market_prob"]

    # Profit for 1-unit stake
    def profit_row(r: pd.Series) -> float:
        if r["win"] == 1:
            if r["moneyLine"] > 0:
                return r["moneyLine"] / 100
            return 100 / abs(r["moneyLine"])
        return -1.0

    df["profit"] = df.apply(profit_row, axis=1)

    # EV buckets (robust deciles even if EV has ties)
    df["ev_bucket"] = pd.qcut(df["EV"].rank(method="first"), 10, labels=False)

    bucket_results = (
        df.groupby("ev_bucket")
        .agg(
            avg_EV=("EV", "mean"),
            ROI=("profit", "mean"),
            avg_edge=("edge", "mean"),
            count=("profit", "count"),
        )
        .reset_index()
    )

    # Spearman significance on bucket-level relationship
    rho, pval = stats.spearmanr(bucket_results["avg_EV"], bucket_results["ROI"])

    # Save outputs
    out_csv = ADV_DIR_TEST / "test_ev_results.csv"
    df.to_csv(out_csv, index=False)

    out_txt = ADV_DIR_TEST / "test_ev_bucket_summary.txt"
    out_txt.write_text(bucket_results.to_string(index=False) + "\n")

    out_sig = ADV_DIR_TEST / "test_spearman.txt"
    out_sig.write_text(f"Spearman rho: {rho:.6f}\np-value: {pval:.6f}\n")

    print("[ADV TEST EV] Saved:", out_csv)
    print("[ADV TEST EV] Saved:", out_txt)
    print("[ADV TEST EV] Saved:", out_sig)



def run_advanced_significance_from_full() -> None:
    """
    - loads full EV results
    - recomputes buckets
    - Spearman between avg_EV and ROI at bucket level
    """
    ensure_dir(ADV_DIR_SIG)

    ev_path = ADV_DIR_EV / "ev_results_full.csv"
    if not ev_path.exists():
        raise FileNotFoundError(f"Missing EV results: {ev_path}. Run `advanced-ev` first.")

    df = pd.read_csv(ev_path)
    df["ev_bucket"] = pd.qcut(df["EV"], 10, labels=False, duplicates="drop")

    bucket_results = (
        df.groupby("ev_bucket")
        .agg(avg_EV=("EV", "mean"), ROI=("profit", "mean"), count=("profit", "count"))
        .reset_index()
    )

    rho, p_value = stats.spearmanr(bucket_results["avg_EV"], bucket_results["ROI"])

    out_txt = ADV_DIR_SIG / "spearman_full.txt"
    out_txt.write_text(
        "SPEARMAN SIGNIFICANCE (FULL DATA)\n\n"
        + bucket_results.to_string(index=False)
        + "\n\n"
        + f"Spearman rho: {rho:.6f}\n"
        + f"p-value: {p_value:.6f}\n"
    )
    print("[ADV SIG] Saved:", out_txt)


def run_advanced_figures() -> None:
    """
    Generates 6 diagnostic figures for the advanced model:
    1) feature importance (GB)
    2) ROC curve (test)
    3) calibration curve (test)
    4) avg EV by bucket (test)
    5) total profit by bucket (test)
    6) edge vs profit scatter (test)
    """
    ensure_dir(ADV_DIR_FIG)

    pred_path = ADV_DIR_PRED / "advanced_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}. Run `advanced-train` first.")

    # Load cleaned data and predictions 
    cleaned_path = _ensure_cleaned_csv()
    raw = pd.read_csv(cleaned_path)
    preds = pd.read_csv(pred_path)

    # Backward compatibility to ensure row_id exists on both sides
    if "row_id" not in raw.columns:
        raw = raw.reset_index(drop=True).copy()
        raw["row_id"] = np.arange(len(raw))

    if "row_id" not in preds.columns:
        preds = preds.reset_index(drop=True).copy()
        preds["row_id"] = np.arange(len(preds))

    if "set" not in preds.columns:
        raise ValueError("Predictions file missing 'set'. Re-run advanced-train.")

    # Safe merge (never rely on row order)
    df = raw.merge(
        preds[["row_id", "logit_prob", "set"]],
        on="row_id",
        how="inner",
    ).copy()

    test_df = df[df["set"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No rows labeled set=='test'. Re-run advanced-train or check split logic.")

    # 1) Feature importance (Gradient Boosting) on advanced features (no leakage)
    feat = build_advanced_features("data/cleaned_data.csv")
    y = feat["win"].astype(int)
    X = feat.drop(columns=["win"])

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X, y)
    importances = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

    plt.figure()
    importances.sort_values().plot(kind="barh")
    plt.title("Top Feature Importances (Gradient Boosting)")
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "feature_importance_gb.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 2) ROC curve (test set)
    y_true = test_df["win"].astype(int).to_numpy()
    y_score = test_df["logit_prob"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "roc_curve_test.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 3) Calibration curve (test set)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Win Rate")
    plt.title("Calibration Curve (Test Set)")
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "calibration_curve_test.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 4-6) EV/profit/edge plots on test set
    test_df["market_prob"] = test_df["moneyLine"].apply(moneyline_to_prob)
    test_df["EV"] = test_df.apply(lambda r: expected_value(r["logit_prob"], r["moneyLine"]), axis=1)
    test_df["edge"] = test_df["logit_prob"] - test_df["market_prob"]

    def profit_row(r: pd.Series) -> float:
        if r["win"] == 1:
            if r["moneyLine"] > 0:
                return r["moneyLine"] / 100
            return 100 / abs(r["moneyLine"])
        return -1.0

    test_df["profit"] = test_df.apply(profit_row, axis=1)

    # robust deciles
    test_df["ev_bucket"] = pd.qcut(test_df["EV"].rank(method="first"), 10, labels=False)

    bucket = (
        test_df.groupby("ev_bucket")
        .agg(
            avg_EV=("EV", "mean"),
            total_profit=("profit", "sum"),
            count=("profit", "count"),
        )
        .reset_index()
    )

    # EV by bucket
    plt.figure()
    plt.plot(bucket["ev_bucket"], bucket["avg_EV"], marker="o")
    plt.xlabel("EV Bucket")
    plt.ylabel("Average EV")
    plt.title("Average EV by Bucket (Test Set)")
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "ev_by_bucket_test.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Profit by bucket
    plt.figure()
    plt.plot(bucket["ev_bucket"], bucket["total_profit"], marker="o")
    plt.xlabel("EV Bucket")
    plt.ylabel("Total Profit")
    plt.title("Total Profit by Bucket (Test Set)")
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "profit_by_bucket_test.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Edge vs profit scatter
    plt.figure()
    plt.scatter(test_df["edge"], test_df["profit"], s=10)
    plt.xlabel("Edge (model_prob - market_prob)")
    plt.ylabel("Realized Profit")
    plt.title("Edge vs Realized Profit (Test Set)")
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "edge_vs_profit_test.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("[ADV FIGURES] Saved 6 figures to:", ADV_DIR_FIG)

def run_advanced_efficiency() -> None:
    """
    Runs the full advanced-model pipeline and saves outputs to results/advanced/
    """
    ensure_dir(ADV_DIR_EFF)

    run_advanced_train()
    run_advanced_ev_full()
    run_advanced_ev_testset()
    run_advanced_significance_from_full()
    run_advanced_figures()

    (ADV_DIR_EFF / "run_log.txt").write_text("Completed advanced model suite.\n")
    print("[ADV EFF] Saved:", ADV_DIR_EFF / "run_log.txt")

# ============================================================
# CLI: 
# ============================================================

# main() function centralizes all execution logic so the file can act as a single reproducible entrypoint for the project.
# Accepting argv explicitly also allows programmatic execution without relying on sys.argv.
# in the sub = parser.ad... function, I changed the "required" part. Before, when it was "required = True", 
# I wasn't able to run the entire code. Changing this to False allows anybody to run the entire pipeline.
def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Main runner (advanced model).")
    sub = parser.add_subparsers(dest="cmd", required=False)
    # -------- Advanced commands 
    sub.add_parser("advanced-train")
    sub.add_parser("advanced-ev")
    sub.add_parser("advanced-test-ev")
    sub.add_parser("advanced-significance")
    sub.add_parser("advanced-figures")
    sub.add_parser("advanced-efficiency")

    args = parser.parse_args(argv)
    # if no cmd is given, this allows to run the entire model without fail 
    if args.cmd is None:
        run_advanced_efficiency()
        return

    # ---- Advanced dispatch
    elif args.cmd == "advanced-train":
        run_advanced_train()
    elif args.cmd == "advanced-ev":
        run_advanced_ev_full()
    elif args.cmd == "advanced-test-ev":
        run_advanced_ev_testset()
    elif args.cmd == "advanced-significance":
        run_advanced_significance_from_full()
    elif args.cmd == "advanced-figures":
        run_advanced_figures()
    elif args.cmd == "advanced-efficiency":
        run_advanced_efficiency()
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
