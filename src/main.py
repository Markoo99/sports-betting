from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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


# ------------------------------------------------------------
# Imports from your separate modules for SIMPLE MODEL
# (NOT copied into this file)
# ------------------------------------------------------------
try:
    # when running `python -m src.main ...`
    from .data_loading import load_raw_data  # type: ignore
    from .preprocessing import preprocess_data  # type: ignore
    from .modeling import FEATURE_COLUMNS, make_feature_matrix  # type: ignore
except Exception:  # pragma: no cover
    # when running `python src/main.py ...`
    from data_loading import load_raw_data  # type: ignore
    from preprocessing import preprocess_data  # type: ignore
    from modeling import FEATURE_COLUMNS, make_feature_matrix  # type: ignore


# ============================================================
# RESULTS FOLDERS (auto-created)
# ============================================================

# this part of the code is meant to create two separate results directories in order to distinguish simple and advanced model outputs
    
# Simple model results 
SIMPLE_ROOT = Path("results") / "simple"
DIR_BACKTEST = SIMPLE_ROOT / "backtest"
DIR_EV = SIMPLE_ROOT / "ev"
DIR_MULTI_EV = SIMPLE_ROOT / "multi_threshold_ev"
DIR_CAL = SIMPLE_ROOT / "calibration"
DIR_HL = SIMPLE_ROOT / "hl_test"
DIR_EFF = SIMPLE_ROOT / "efficiency"

# Advanced model results 
ADV_ROOT = Path("results") / "advanced"
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
# SHARED HELPERS
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
    p.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(p, index=False)
    return p


# ============================================================
# ===================== SIMPLE MODEL (unchanged) =====================
# ============================================================

def train_model_for_backtest(
    df_clean: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]:
    X, y = make_feature_matrix(df_clean)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df_clean, test_size=test_size, random_state=random_state, stratify=y
    ) # relatively large test set so that calibration and EV estimates are statistically meaningful. 
      # fixed seed makes the results reproducible and stratification is there to preserve the win/loss ratio across splits

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    df_test = df_test.copy()
    df_test["model_prob"] = model.predict_proba(X_test)[:, 1]
    return model, df_train.copy(), df_test


def run_simple_backtest() -> None:
    ensure_dir(DIR_BACKTEST)

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    X, y = make_feature_matrix(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "brier": float(brier_score_loss(y_test, y_proba)),
    }

    out_txt = DIR_BACKTEST / "metrics.txt"
    out_txt.write_text("\n".join([f"{k}: {v}" for k, v in metrics.items()]) + "\n")
    print("[SIMPLE BACKTEST] Saved:", out_txt)


def compute_ev_for_bets(edge_threshold: float = 0.02) -> pd.DataFrame:
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    _, _, df_test = train_model_for_backtest(df_clean)

    df_test = df_test.copy()
    df_test["decimal_odds"] = df_test["moneyLine"].apply(american_to_decimal)
    df_test["edge"] = df_test["model_prob"] - df_test["team_prob"]

    bets = df_test[df_test["edge"] > edge_threshold].copy() # this is the part which will categorize which bets to use 
    if bets.empty:
        return bets

    bets["payoff_if_win"] = bets["decimal_odds"] - 1.0
    bets["ev_market"] = bets["team_prob"] * bets["payoff_if_win"] - (1 - bets["team_prob"]) * 1.0
    bets["ev_model"] = bets["model_prob"] * bets["payoff_if_win"] - (1 - bets["model_prob"]) * 1.0
    bets["realized_return"] = np.where(bets["win"] == 1, bets["payoff_if_win"], -1.0)
    return bets
# edge is defined here as the difference between the model and market probability. A positive edge means that the model gives higher win probability.
# so, in this case, the model will take into consideration all the bets where the model assigns a probability 0.02 higher than what the market gives.
# therefore, this part of the code is defining and extracting such bets, to later use for evaluation

def run_simple_ev(edge_threshold: float = 0.02) -> None:
    ensure_dir(DIR_EV)

    bets = compute_ev_for_bets(edge_threshold=edge_threshold)
    out_csv = DIR_EV / f"ev_bets_edge_{edge_threshold:.3f}.csv"

    if bets.empty:
        (DIR_EV / "note.txt").write_text(f"No bets selected for edge_threshold={edge_threshold}\n")
        print("[SIMPLE EV] No bets selected.")
        return

    bets.to_csv(out_csv, index=False)
    print("[SIMPLE EV] Saved:", out_csv)

# the following function is similar to the one above, but instead of one, it tests for multiple thresholds to see whether the profitability is robust
def run_simple_multi_threshold_ev() -> None:
    ensure_dir(DIR_MULTI_EV)

    cleaned_path = _ensure_cleaned_csv()
    df = pd.read_csv(cleaned_path)

    X = df[["team_prob"]]
    y = df["win"].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    df = df.copy()
    df["model_prob"] = model.predict_proba(X)[:, 1]

    thresholds = [0.005, 0.01, 0.02]
    summary_lines: List[str] = ["EV SUMMARY ACROSS THRESHOLDS\n"]

    for t in thresholds:
        d = df.copy()
        d["edge"] = d["model_prob"] - d["team_prob"]
        bets = d[d["edge"].abs() >= t].copy()

        if bets.empty:
            summary_lines.append(f"\nThreshold {t}: 0 bets\n")
            continue

        bets["market_odds"] = bets["moneyLine"].abs() / 100.0
        bets["market_ev"] = bets["team_prob"] - (1 - bets["team_prob"]) * bets["market_odds"]
        bets["model_ev"] = bets["model_prob"] - (1 - bets["model_prob"]) * bets["market_odds"]
        bets["profit"] = np.where(bets["win"] == 1, 1.0, -bets["market_odds"])

        out_csv = DIR_MULTI_EV / f"ev_bets_edge_{t:.3f}.csv"
        bets.to_csv(out_csv, index=False)

        summary_lines.append(f"\nThreshold: {t}\n")
        summary_lines.append(f"Number of bets: {len(bets)}\n")
        summary_lines.append(f"Hit rate: {bets['win'].mean():.4f}\n")
        summary_lines.append(f"Avg EV (model): {bets['model_ev'].mean():.6f}\n")
        summary_lines.append(f"Avg EV (market): {bets['market_ev'].mean():.6f}\n")
        summary_lines.append(f"Avg realized return: {bets['profit'].mean():.6f}\n")

    out_txt = DIR_MULTI_EV / "ev_threshold_summary.txt"
    out_txt.write_text("".join(summary_lines))
    print("[SIMPLE MULTI-EV] Saved:", out_txt)

#the next function is calibrating the market. Calibration evaluates whether events that are priced at probability p actually occur with that frequenc. 
#in an efficient market, predicted probabilities shoudl be the same as the obsereved ones
def run_simple_market_calibration(n_bins: int = 10) -> None:
    ensure_dir(DIR_CAL)

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    data = df_clean[["team_prob", "win"]].dropna().copy()
    data["team_prob"] = data["team_prob"].clip(0.0, 1.0)

    bins = np.linspace(0, 1, n_bins + 1)
    data["prob_bin"] = pd.cut(data["team_prob"], bins=bins, include_lowest=True)

    table = (
        data.groupby("prob_bin")
        .agg(mean_prob=("team_prob", "mean"), win_rate=("win", "mean"), count=("win", "size"))
        .reset_index()
    )

    out_csv = DIR_CAL / "market_calibration_table.csv"
    table.to_csv(out_csv, index=False)
    print("[SIMPLE CAL] Saved:", out_csv)

#the next part is preparing simple testing. It will evaluate calibration by comparing observed and expected wins withing probability bins. 
# this test is used here because this simple model focuses on probability accuracy rather than ranking ability, since it is too simple and basic
def hosmer_lemeshow(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    df = pd.DataFrame({"p": probs, "y": outcomes})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")

    g = df.groupby("bin")
    n = g.size().values
    p_hat = g["p"].mean().values
    obs = g["y"].sum().values

    exp = n * p_hat
    hl = np.sum((obs - exp) ** 2 / (n * p_hat * (1 - p_hat)))
    dof = len(n) - 2
    p_value = 1 - stats.chi2.cdf(hl, dof)
    return float(hl), float(p_value)

# afterwards, the test is ran in this portion of the code, 
def run_simple_hl() -> None:
    ensure_dir(DIR_HL)

    cleaned_path = _ensure_cleaned_csv()
    df = pd.read_csv(cleaned_path).dropna(subset=["team_prob", "opp_prob", "win"]).copy()

    X = df[["team_prob", "opp_prob"]]
    y = df["win"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    model_prob = model.predict_proba(X_test)[:, 1]
    book_prob = X_test["team_prob"].to_numpy()
    outcomes = y_test.to_numpy()

    hl_book, p_book = hosmer_lemeshow(book_prob, outcomes, n_bins=10)
    hl_model, p_model = hosmer_lemeshow(model_prob, outcomes, n_bins=10)

    out_txt = DIR_HL / "hl_summary.txt"
    out_txt.write_text(
        "Hosmer–Lemeshow (HL) Test\n\n"
        f"Bookmaker: HL={hl_book:.6f}, p={p_book:.6f}\n"
        f"Model:     HL={hl_model:.6f}, p={p_model:.6f}\n"
    )
    print("[SIMPLE HL] Saved:", out_txt)

# the final part of the simple model combines all the different parts of the code and ensures that it can be run as one, instead of doing each part separately. 
# if its more convenient to run only a simple model, in Nuvolos termina, run "python src/main.py run_simple_efficiency" to get all the output for the simple model part. 
# if not, it is possible to run each part separately, using the names of each part defined below. If not, the entire script can be ran simply with "python src/main.py and it will show all the outputs of the entire pipeline 
def run_simple_efficiency(edge_threshold: float = 0.02) -> None:
    ensure_dir(DIR_EFF)

    run_simple_backtest()
    run_simple_market_calibration(n_bins=10)
    run_simple_ev(edge_threshold=edge_threshold)
    run_simple_multi_threshold_ev()
    run_simple_hl()

    (DIR_EFF / "run_log.txt").write_text(
        f"Completed simple efficiency suite with edge_threshold={edge_threshold}\n"
    )
    print("[SIMPLE EFF] Saved:", DIR_EFF / "run_log.txt")


# ============================================================
# ===================== ADVANCED MODEL (NEW) =====================
# ============================================================

def build_advanced_features(path: str = "data/cleaned_data.csv") -> pd.DataFrame:
    """
    WHY this exists:
    - The advanced model expands the feature space beyond raw implied probabilities.
    - This is the "stronger competitor" to the market that we use for Spearman/EV tests.
    """
    df = pd.read_csv(path)

    # Bookmaker implied probabilities
    df["market_prob"] = df["moneyLine"].apply(moneyline_to_prob)
    df["opp_market_prob"] = df["opponentMoneyLine"].apply(moneyline_to_prob) if "opponentMoneyLine" in df.columns else np.nan

    # Nonlinear transforms 
    if "market_prob" in df.columns:
        df["market_prob_sq"] = df["market_prob"] ** 2
        df["market_prob_logit"] = np.log(df["market_prob"].clip(1e-9, 1 - 1e-9) / (1 - df["market_prob"].clip(1e-9, 1 - 1e-9)))

    # Home indicator if present. It is an important feature that can influence win probability 
    if "home/visitor" in df.columns:
        df["is_home"] = (df["home/visitor"].astype(str).str.lower() == "home").astype(int)

    # Keep season for time-based split
    keep_cols = []
    for c in df.columns:
        if c == "season":
            keep_cols.append(c)
        elif c == "win":
            keep_cols.append(c)
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                keep_cols.append(c)

    df = df[keep_cols].copy()

    # Ensure win exists and is integer
    df["win"] = df["win"].astype(int)

    # Drop rows with missing in key columns
    df = df.dropna()

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


def run_advanced_ev_full() -> None:
    """
    - uses cleaned_data + advanced predictions
    - computes EV, edge, buckets, ROI by bucket
    """
    ensure_dir(ADV_DIR_EV)
    pred_path = ADV_DIR_PRED / "advanced_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing advanced predictions: {pred_path}. Run `advanced-train` first.")

    raw = pd.read_csv("data/cleaned_data.csv")
    preds = pd.read_csv(pred_path)

    df = raw.copy()
    df["logit_prob"] = preds["logit_prob"]
    if "set" in preds.columns:
        df["set"] = preds["set"]

    df["market_prob"] = df["moneyLine"].apply(moneyline_to_prob)
    df["EV"] = df.apply(lambda r: expected_value(r["logit_prob"], r["moneyLine"]), axis=1)
    df["edge"] = df["logit_prob"] - df["market_prob"]

    df["ev_bucket"] = pd.qcut(df["EV"], 10, labels=False, duplicates="drop")

    # Profit for 1 unit stake 
    def profit_row(r: pd.Series) -> float:
        if r["win"] == 1:
            if r["moneyLine"] > 0:
                return r["moneyLine"] / 100
            return 100 / abs(r["moneyLine"])
        return -1.0

    df["profit"] = df.apply(profit_row, axis=1)

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

    out_csv = ADV_DIR_EV / "ev_results_full.csv"
    df.to_csv(out_csv, index=False)

    out_txt = ADV_DIR_EV / "ev_bucket_summary_full.txt"
    out_txt.write_text(bucket_results.to_string(index=False) + "\n")

    print("[ADV EV FULL] Saved:", out_csv)
    print("[ADV EV FULL] Saved:", out_txt)


def run_advanced_ev_testset() -> None:
    """
    Merged from test_set.py:
    - same EV computation but restricted to rows labeled set=='test'
    - runs Spearman between avg_EV and ROI at bucket level
    """
    ensure_dir(ADV_DIR_TEST)

    pred_path = ADV_DIR_PRED / "advanced_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing advanced predictions: {pred_path}. Run `advanced-train` first.")

    raw = pd.read_csv("data/cleaned_data.csv")
    preds = pd.read_csv(pred_path)

    if "set" not in preds.columns:
        raise ValueError("Predictions file does not contain 'set' column. Re-run advanced-train.")

    df = raw.copy()
    df["logit_prob"] = preds["logit_prob"]
    df["set"] = preds["set"]

    df = df[df["set"] == "test"].copy()

    df["market_prob"] = df["moneyLine"].apply(moneyline_to_prob)
    df["EV"] = df.apply(lambda r: expected_value(r["logit_prob"], r["moneyLine"]), axis=1)
    df["edge"] = df["logit_prob"] - df["market_prob"]

    def profit_row(r: pd.Series) -> float:
        if r["win"] == 1:
            if r["moneyLine"] > 0:
                return r["moneyLine"] / 100
            return 100 / abs(r["moneyLine"])
        return -1.0

    df["profit"] = df.apply(profit_row, axis=1)
    df["ev_bucket"] = pd.qcut(df["EV"], 10, labels=False, duplicates="drop")

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

    # Spearman significance on test-set buckets (as in your script)
    rho, pval = stats.spearmanr(bucket_results["avg_EV"], bucket_results["ROI"])

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
    ensure_dir(ADV_DIR_FIG)

    pred_path = ADV_DIR_PRED / "advanced_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}. Run `advanced-train` first.")

    raw = pd.read_csv("data/cleaned_data.csv")
    preds = pd.read_csv(pred_path)
    if "set" not in preds.columns:
        raise ValueError("Predictions file does not contain 'set'. Re-run advanced-train.")

    # Build advanced features for feature importance + for test curves
    feat = build_advanced_features("data/cleaned_data.csv")
    y = feat["win"].astype(int)
    X = feat.drop(columns=["win"])

    # 1) Feature importance (Gradient Boosting)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    gb.fit(X, y)
    importances = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

    plt.figure()
    importances.sort_values().plot(kind="barh")
    plt.title("Top Feature Importances (Gradient Boosting)")
    plt.tight_layout()
    out_path = ADV_DIR_FIG / "feature_importance_gb.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Prepare test set rows from preds
    df = raw.copy()
    df["logit_prob"] = preds["logit_prob"]
    df["set"] = preds["set"]
    test_df = df[df["set"] == "test"].copy()

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
    test_df["ev_bucket"] = pd.qcut(test_df["EV"], 10, labels=False, duplicates="drop")

    bucket = (
        test_df.groupby("ev_bucket")
        .agg(avg_EV=("EV", "mean"), avg_profit=("profit", "mean"), total_profit=("profit", "sum"), count=("profit", "count"))
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
    This code allows to only run the advanced model as follows "python src/main.py run_advanced_efficiency"
    If not, possible to run each part of the code using the same logic, but instead of run advanced function, choose the one that is needed from the list below
    Lastly, possible to also run just the entire main.py giving all the outputs from this pipeline 
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

def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Main runner (simple + advanced).")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # -------- Simple commands (unchanged)
    sub.add_parser("simple-backtest")
    p_ev = sub.add_parser("simple-ev")
    p_ev.add_argument("--edge-threshold", type=float, default=0.02)

    sub.add_parser("simple-multi-ev")

    p_cal = sub.add_parser("simple-calibration")
    p_cal.add_argument("--bins", type=int, default=10)

    sub.add_parser("simple-hl")

    p_eff = sub.add_parser("simple-efficiency")
    p_eff.add_argument("--edge-threshold", type=float, default=0.02)

    # -------- Advanced commands (new)
    sub.add_parser("advanced-train")
    sub.add_parser("advanced-ev")
    sub.add_parser("advanced-test-ev")
    sub.add_parser("advanced-significance")
    sub.add_parser("advanced-figures")
    sub.add_parser("advanced-efficiency")

    args = parser.parse_args(argv)
    # if no cmd is given, this allows to run the entire model without fail 
    if args.cmd is None:
        run_simple_efficiency()
        run_advanced_efficiency()
    return


    # ---- Simple dispatch
    if args.cmd == "simple-backtest":
        run_simple_backtest()
    elif args.cmd == "simple-ev":
        run_simple_ev(edge_threshold=float(args.edge_threshold))
    elif args.cmd == "simple-multi-ev":
        run_simple_multi_threshold_ev()
    elif args.cmd == "simple-calibration":
        run_simple_market_calibration(n_bins=int(args.bins))
    elif args.cmd == "simple-hl":
        run_simple_hl()
    elif args.cmd == "simple-efficiency":
        run_simple_efficiency(edge_threshold=float(args.edge_threshold))

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
