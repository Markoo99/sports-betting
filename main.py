from __future__ import annotations

from pathlib import Path


def main() -> None:
    # -----------------------------
    # 1) Load + 2) Preprocess
    # -----------------------------
    from src.data_loading import load_raw_data
    from src.preprocessing import preprocess_data

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    # Always write cleaned dataset for reproducibility
    out_clean = Path("data") / "cleaned_data.csv"
    out_clean.parent.mkdir(parents=True, exist_ok=True)

    df_clean = df_clean.reset_index(drop=True).copy()
    if "row_id" not in df_clean.columns:
        df_clean["row_id"] = range(len(df_clean))

    df_clean.to_csv(out_clean, index=False)
    print(f"[MAIN] Saved cleaned data to: {out_clean}")

    # -----------------------------
    # 3) Run baseline modeling.py
    # -----------------------------
    from src.modeling import make_feature_matrix, train_logistic_regression

    X, y = make_feature_matrix(df_clean)
    model, metrics = train_logistic_regression(X, y)

    # Save baseline metrics 
    results_dir = Path("results") / "simple"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "baseline_metrics.txt").write_text(
        "Baseline Logistic Regression (random split)\n"
        + "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        + "\n"
    )
    print("Baseline model metrics:", metrics)

    # -----------------------------
    # 4) Run evaluation.py
    # -----------------------------
    from src.evaluation import (
        run_advanced_train,
        run_advanced_ev_full,
        run_advanced_ev_testset,
        run_advanced_significance_from_full,
        run_advanced_figures,
        run_advanced_efficiency,
    )

    run_advanced_train()
    run_advanced_ev_full()
    run_advanced_ev_testset()
    run_advanced_significance_from_full()
    run_advanced_figures()

if __name__ == "__main__":
    main()
