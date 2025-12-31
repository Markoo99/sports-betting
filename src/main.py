from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from scipy import stats

#t he try/except lets the same file run both as a module (`python -m src.main...`)
# and as a script (`python src/main.py...`) in nuvolos
try:
    from .data_loading import load_raw_data  # type: ignore
    from .preprocessing import preprocess_data  # type: ignore
    from .modeling import FEATURE_COLUMNS, make_feature_matrix  # type: ignore
except Exception:  # pragma: no cover
    from data_loading import load_raw_data  # type: ignore
    from preprocessing import preprocess_data  # type: ignore
    from modeling import FEATURE_COLUMNS, make_feature_matrix  # type: ignore


# ============================================================
# Results folders 
# ============================================================
RESULTS_ROOT = Path("results") / "simple"
DIR_BACKTEST = RESULTS_ROOT / "backtest"
DIR_EV = RESULTS_ROOT / "ev"
DIR_MULTI_EV = RESULTS_ROOT / "multi_threshold_ev"
DIR_CAL = RESULTS_ROOT / "calibration"
DIR_SIG = RESULTS_ROOT / "significance_hl"
DIR_EFF = RESULTS_ROOT / "efficiency"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================
def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds is None or pd.isna(odds) or odds == 0:
        return float("nan") # this part is added to avoid the ZeroDivisionError
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / (-odds))


# ============================================================
# BACKTESTING 
# ============================================================
def train_model_for_backtest(
    df_clean: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Train logistic regression and return:
      model, df_train, df_test(with model_prob), y_train, y_test
    """
    X, y = make_feature_matrix(df_clean)

    # Split X,y and the df together so df_test aligns with probabilities
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df_clean, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    df_test = df_test.copy()
    df_test["model_prob"] = model.predict_proba(X_test)[:, 1]

    return model, df_train.copy(), df_test, y_train, y_test


def run_simple_backtest() -> None:
    ensure_dir(DIR_BACKTEST)

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    X, y = make_feature_matrix(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # A 30% test split ensures that out-of-sample calibration and EV estimates are based on a sufficiently large sample, while still leaving enough data to estimate model parameters reliably
    # Random state of 42 is a conventional number
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

    # Save metrics. this will allow me to save this output as a txt file that is easily readable
    out_txt = DIR_BACKTEST / "metrics.txt"
    out_txt.write_text("\n".join([f"{k}: {v}" for k, v in metrics.items()]) + "\n")

    
    print(metrics)


# ============================================================
# EV ANALYSIS 
# ============================================================
# In this part of the code, I will be focusing on an edge threshold of 0.02. What this code will do is use all the bets that have a probability differential of at least 0.02
# This is an interesting point to start at as I feel like the edge of 0.02 gives a large enough sample size but still gives "safer" bets

def compute_ev_for_bets(edge_threshold: float = 0.02) -> pd.DataFrame:
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    _, _, df_test, _, _ = train_model_for_backtest(df_clean)

    df_test = df_test.copy()
    df_test["decimal_odds"] = df_test["moneyLine"].apply(american_to_decimal)
    df_test["edge"] = df_test["model_prob"] - df_test["team_prob"]

    bets = df_test[df_test["edge"] > edge_threshold].copy()
    if bets.empty:
        return bets

    bets["payoff_if_win"] = bets["decimal_odds"] - 1.0

    # Market EV (using bookmaker probability)
    bets["ev_market"] = (
        bets["team_prob"] * bets["payoff_if_win"] - (1.0 - bets["team_prob"]) * 1.0
    )

    # Model EV (using model probability)
    bets["ev_model"] = (
        bets["model_prob"] * bets["payoff_if_win"] - (1.0 - bets["model_prob"]) * 1.0
    )

    # Realized return (1 unit stake)
    bets["realized_return"] = np.where(bets["win"] == 1, bets["payoff_if_win"], -1.0)

    return bets


def summarize_ev(bets: pd.DataFrame) -> Dict[str, float]:
    if bets.empty:
        return {
            "num_bets": 0.0,
            "hit_rate": float("nan"),
            "avg_ev_model": float("nan"),
            "avg_ev_market": float("nan"),
            "avg_realized_return": float("nan"),
        }
    return {
        "num_bets": float(len(bets)),
        "hit_rate": float(bets["win"].mean()),
        "avg_ev_model": float(bets["ev_model"].mean()),
        "avg_ev_market": float(bets["ev_market"].mean()),
        "avg_realized_return": float(bets["realized_return"].mean()),
    }


def run_simple_ev(edge_threshold: float = 0.02) -> None:
    ensure_dir(DIR_EV)

    bets = compute_ev_for_bets(edge_threshold=edge_threshold)
    out_csv = DIR_EV / f"ev_bets_edge_{edge_threshold:.3f}.csv"

    if bets.empty:
        (DIR_EV / "note.txt").write_text(
            f"No bets selected for edge_threshold={edge_threshold}\n"
        )
        print("[EV] No bets selected.")
        return

    bets.to_csv(out_csv, index=False)
    summary = summarize_ev(bets)

    out_txt = DIR_EV / f"ev_summary_edge_{edge_threshold:.3f}.txt"
    out_txt.write_text("\n".join([f"{k}: {v}" for k, v in summary.items()]) + "\n")

    print("[EV] Saved:", out_csv)
    print("[EV] Saved:", out_txt)
    print(summary)


# ============================================================
# MULTI-THRESHOLD EV 
# ============================================================


def load_cleaned_for_multi_ev() -> pd.DataFrame:
    p = Path("data") / "cleaned_data.csv"
    if not p.exists():
        # create it if missing
        df_raw = load_raw_data()
        df_clean = preprocess_data(df_raw)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(p, index=False)
    return pd.read_csv(p)


def train_model_multi_ev(df: pd.DataFrame) -> Tuple[pd.DataFrame, LogisticRegression]:
    # your original multi-threshold script trains on team_prob only
    X = df[["team_prob"]]
    y = df["win"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    out = df.copy()
    out["model_prob"] = model.predict_proba(X)[:, 1]
    return out, model


def compute_ev_for_threshold(df: pd.DataFrame, threshold: float) -> Dict[str, object]:
    df = df.copy()
    df["edge"] = df["model_prob"] - df["team_prob"]
    bets = df[df["edge"].abs() >= threshold].copy()

    if bets.empty:
        return {
            "threshold": float(threshold),
            "num_bets": 0,
            "hit_rate": None,
            "avg_ev_model": None,
            "avg_ev_market": None,
            "realized_return": None,
        }

    bets["market_odds"] = bets["moneyLine"].abs() / 100.0
    bets["market_ev"] = bets["team_prob"] - (1 - bets["team_prob"]) * bets["market_odds"]
    bets["model_ev"] = bets["model_prob"] - (1 - bets["model_prob"]) * bets["market_odds"]

    # Realized profit (1 unit per bet)
    bets["profit"] = np.where(bets["win"] == 1, 1.0, -bets["market_odds"])

    return {
        "threshold": float(threshold),
        "num_bets": int(len(bets)),
        "hit_rate": float(bets["win"].mean()),
        "avg_ev_model": float(bets["model_ev"].mean()),
        "avg_ev_market": float(bets["market_ev"].mean()),
        "realized_return": float(bets["profit"].mean()),
        "bets_df": bets,  # internal: used for saving
    }


def run_simple_multi_threshold_ev() -> None:
    ensure_dir(DIR_MULTI_EV)

    df = load_cleaned_for_multi_ev()
    df, _ = train_model_multi_ev(df)

    thresholds = [0.005, 0.01, 0.02]
    results: List[Dict[str, object]] = []

    for t in thresholds:
        res = compute_ev_for_threshold(df, t)
        results.append(res)

        # Save bet-level file per threshold
        bets_df = res.get("bets_df")
        if isinstance(bets_df, pd.DataFrame) and not bets_df.empty:
            out_csv = DIR_MULTI_EV / f"ev_bets_edge_{t:.3f}.csv"
            bets_df.to_csv(out_csv, index=False)

    # Save summary text
    out_txt = DIR_MULTI_EV / "ev_threshold_summary.txt"
    with open(out_txt, "w") as f:
        f.write("EV SUMMARY ACROSS THRESHOLDS\n\n")
        for r in results:
            f.write(f"Threshold: {r['threshold']}\n")
            f.write(f"Number of bets: {r['num_bets']}\n")
            f.write(f"Hit rate: {r['hit_rate']}\n")
            f.write(f"Avg EV (model): {r['avg_ev_model']}\n")
            f.write(f"Avg EV (market): {r['avg_ev_market']}\n")
            f.write(f"Avg realized return: {r['realized_return']}\n")
            f.write("\n")

    print("[MULTI-EV] Saved:", out_txt)
    print("[MULTI-EV] Bet-level CSVs saved in:", DIR_MULTI_EV)


# ============================================================
# MARKET CALIBRATION (from market_calibration.py)
# ============================================================
def compute_calibration_table(
    df: pd.DataFrame,
    prob_col: str = "team_prob",
    outcome_col: str = "win",
    n_bins: int = 10,
) -> pd.DataFrame:
    data = df[[prob_col, outcome_col]].dropna().copy()
    data[prob_col] = data[prob_col].clip(0.0, 1.0)

    bins = np.linspace(0, 1, n_bins + 1)
    data["prob_bin"] = pd.cut(data[prob_col], bins=bins, include_lowest=True)
    grouped = (
        data.groupby("prob_bin")
        .agg(
            mean_prob=(prob_col, "mean"),
            win_rate=(outcome_col, "mean"),
            count=(outcome_col, "size"),
        )
        .reset_index()
    )
    return grouped


def run_simple_market_calibration(n_bins: int = 10) -> None:
    ensure_dir(DIR_CAL)

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    table = compute_calibration_table(df_clean, prob_col="team_prob", outcome_col="win", n_bins=n_bins)
    out_csv = DIR_CAL / "market_calibration_table.csv"
    table.to_csv(out_csv, index=False)

    print("[CALIBRATION] Saved:", out_csv)


# ============================================================
# SIGNIFICANCE / HL TESTS 
# ============================================================
def hosmer_lemeshow(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """
    Hosmer–Lemeshow goodness-of-fit test.
    Lower statistic + high p-value = better calibration.
    """
    df = pd.DataFrame({"p": probs, "y": outcomes})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")

    grouped = df.groupby("bin")
    n = grouped.size().values
    p_hat = grouped["p"].mean().values
    obs = grouped["y"].sum().values

    exp = n * p_hat
    hl_stat = np.sum((obs - exp) ** 2 / (n * p_hat * (1 - p_hat)))
    dof = len(n) - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, dof)

    return float(hl_stat), float(p_value)


def paired_logloss_test(
    book_prob: np.ndarray,
    model_prob: np.ndarray,
    outcomes: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Paired t-test on per-bet log-loss.
    Returns: bookmaker LL, model LL, mean diff, t-stat, p-value
    """
    eps = 1e-15
    b = np.clip(book_prob, eps, 1 - eps)
    m = np.clip(model_prob, eps, 1 - eps)
    y = outcomes.astype(int)

    loss_book = -(y * np.log(b) + (1 - y) * np.log(1 - b))
    loss_model = -(y * np.log(m) + (1 - y) * np.log(1 - m))

    ll_book = float(loss_book.mean())
    ll_model = float(loss_model.mean())

    diff = loss_book - loss_model
    mean_diff = float(diff.mean())
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_stat = float(mean_diff / se)
    df = len(diff) - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    return ll_book, ll_model, mean_diff, t_stat, p_value


def paired_brier_test(
    book_prob: np.ndarray,
    model_prob: np.ndarray,
    outcomes: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Paired t-test on Brier score."""
    y = outcomes.astype(int)

    se_book = (book_prob - y) ** 2
    se_model = (model_prob - y) ** 2

    brier_book = float(se_book.mean())
    brier_model = float(se_model.mean())

    diff = se_book - se_model
    mean_diff = float(diff.mean())
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_stat = float(mean_diff / se)
    df = len(diff) - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    return brier_book, brier_model, mean_diff, t_stat, p_value


def roi_ttest(csv_path: Path) -> Optional[Tuple[str, int, float, float, float]]:
    """
    One-sample t-test:
        H0: mean profit per bet = 0
        H1: mean profit per bet != 0
    """
    if not csv_path.exists():
        return None

    bets = pd.read_csv(csv_path)
    if "profit" not in bets.columns:
        raise ValueError(f"{csv_path} does not contain a 'profit' column.")

    profits = bets["profit"].to_numpy()
    n = len(profits)
    mean_profit = float(profits.mean())
    se = profits.std(ddof=1) / np.sqrt(n)
    t_stat = float(mean_profit / se)
    df = n - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    return csv_path.name, n, mean_profit, t_stat, p_value


def run_simple_significance_hl() -> None:
    ensure_dir(DIR_SIG)

    # Train model on cleaned data using team_prob/opp_prob (as in your HL script)
    cleaned_path = Path("data") / "cleaned_data.csv"
    if not cleaned_path.exists():
        df_raw = load_raw_data()
        df_clean = preprocess_data(df_raw)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(cleaned_path, index=False)

    df = pd.read_csv(cleaned_path)
    required_cols = {"team_prob", "opp_prob", "win"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"cleaned_data.csv is missing columns: {missing}")

    X = df[["team_prob", "opp_prob"]]
    y = df["win"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    model_prob_test = model.predict_proba(X_test)[:, 1]
    book_prob_test = X_test["team_prob"].values
    outcomes = y_test.to_numpy()

    # HL
    hl_book, p_book = hosmer_lemeshow(book_prob_test, outcomes, n_bins=10)
    hl_model, p_model = hosmer_lemeshow(model_prob_test, outcomes, n_bins=10)

    # Paired logloss + brier
    ll_book, ll_model, ll_diff, ll_t, ll_p = paired_logloss_test(book_prob_test, model_prob_test, outcomes)
    b_book, b_model, b_diff, b_t, b_p = paired_brier_test(book_prob_test, model_prob_test, outcomes)

    # Save summary
    out_txt = DIR_SIG / "hl_significance_summary.txt"
    with open(out_txt, "w") as f:
        f.write("CALIBRATION TESTS (Hosmer–Lemeshow)\n\n")
        f.write("Bookmaker probabilities:\n")
        f.write(f"  HL statistic : {hl_book:.6f}\n")
        f.write(f"  p-value      : {p_book:.6f}\n\n")
        f.write("Model probabilities:\n")
        f.write(f"  HL statistic : {hl_model:.6f}\n")
        f.write(f"  p-value      : {p_model:.6f}\n\n")

        f.write("PAIRED TESTS (Bookmaker vs Model)\n\n")
        f.write("Log-loss:\n")
        f.write(f"  LL(book)     : {ll_book:.6f}\n")
        f.write(f"  LL(model)    : {ll_model:.6f}\n")
        f.write(f"  mean diff    : {ll_diff:.6f}  (positive means model better)\n")
        f.write(f"  t-stat       : {ll_t:.6f}\n")
        f.write(f"  p-value      : {ll_p:.6f}\n\n")

        f.write("Brier:\n")
        f.write(f"  Brier(book)  : {b_book:.6f}\n")
        f.write(f"  Brier(model) : {b_model:.6f}\n")
        f.write(f"  mean diff    : {b_diff:.6f}  (positive means model better)\n")
        f.write(f"  t-stat       : {b_t:.6f}\n")
        f.write(f"  p-value      : {b_p:.6f}\n\n")

        f.write("ROI tests (from any bet CSVs that contain a 'profit' column)\n")
        f.write("Note: multi-threshold EV saves profit; single EV saves realized_return.\n\n")

    # ROI tests: scan only the folders where we save profit-based CSVs
    roi_lines: List[str] = []
    for folder in [DIR_MULTI_EV]:
        for csv_path in sorted(folder.glob("*.csv")):
            res = roi_ttest(csv_path)
            if res is None:
                continue
            name, n_bets, mean_profit, t_stat, p_value = res
            roi_lines.append(
                f"{name}: n={n_bets}, mean_profit={mean_profit:.6f}, t={t_stat:.3f}, p={p_value:.6f}"
            )

    if roi_lines:
        (DIR_SIG / "roi_tests.txt").write_text("\n".join(roi_lines) + "\n")

    print("[SIGNIFICANCE/HL] Saved:", out_txt)
    if roi_lines:
        print("[SIGNIFICANCE/HL] Saved:", DIR_SIG / "roi_tests.txt")


# ============================================================
# EFFICIENCY 
# ============================================================
def run_simple_efficiency(edge_threshold: float = 0.02) -> None:
    ensure_dir(DIR_EFF)

    log = []
    log.append("=== SIMPLE MODEL EFFICIENCY RUN ===\n")
    log.append(f"edge_threshold={edge_threshold}\n\n")

    run_simple_market_calibration(n_bins=10)
    run_simple_ev(edge_threshold=edge_threshold)
    run_simple_multi_threshold_ev()
    run_simple_significance_hl()

    out_txt = DIR_EFF / "efficiency_run_log.txt"
    out_txt.write_text("".join(log))
    print("[EFFICIENCY] Saved:", out_txt)


# ============================================================
# CLI ENTRYPOINT
# ============================================================
def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Main runner (simple model for now).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("simple-backtest", help="Run simple model backtest.")
    p_ev = sub.add_parser("simple-ev", help="Run simple EV analysis (single threshold).")
    p_ev.add_argument("--edge-threshold", type=float, default=0.02)

    sub.add_parser("simple-multi-ev", help="Run multi-threshold EV analysis.")
    p_cal = sub.add_parser("simple-calibration", help="Run market calibration.")
    p_cal.add_argument("--bins", type=int, default=10)

    sub.add_parser("simple-hl", help="Run HL + paired tests (book vs model).")
    p_eff = sub.add_parser("simple-efficiency", help="Run full simple-model efficiency suite.")
    p_eff.add_argument("--edge-threshold", type=float, default=0.02)

    args = parser.parse_args(argv)

    if args.cmd == "simple-backtest":
        run_simple_backtest()
    elif args.cmd == "simple-ev":
        run_simple_ev(edge_threshold=float(args.edge_threshold))
    elif args.cmd == "simple-multi-ev":
        run_simple_multi_threshold_ev()
    elif args.cmd == "simple-calibration":
        run_simple_market_calibration(n_bins=int(args.bins))
    elif args.cmd == "simple-hl":
        run_simple_significance_hl()
    elif args.cmd == "simple-efficiency":
        run_simple_efficiency(edge_threshold=float(args.edge_threshold))
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
