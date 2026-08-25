"""Run the Chapter 09 NUTS-3 AutoML pipeline (mirrors notebooks/03_automl_nuts3.ipynb).

Temporal splits use the year of the impact target month t+h, not the predictor
month t, so validation/test periods do not leak into training labels.
"""
from __future__ import annotations

import json
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_SRC_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from automl_search import (
    DEV_TARGET_MAX_YEAR,
    FORECAST_HORIZONS,
    TOP_10_CLASSES,
    assert_target_year_splits,
    best_config_from_study,
    cv_score_config,
    evaluate_config_all_horizons,
    export_temporal_spatial_diagnostics,
    load_feature_groups,
    macro_from_class_metrics,
    make_optuna_objective,
    shap_mean_abs_importance,
    split_by_target_year,
    target_years,
    trials_dataframe,
    year_expanding_folds,
)

PACKAGE_ROOT = _PACKAGE_ROOT
OUT_DIR = PACKAGE_ROOT / "data" / "processed"
RESULT_DIR = PACKAGE_ROOT / "results" / "automl_results"
FIG_DIR = RESULT_DIR / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 100
RANDOM_SEED = 42
PRIMARY_HORIZON = 1
SHAP_N_MODELS_THRESHOLD = 0.05
PANEL_PATH = OUT_DIR / "supervised_nuts3_forecast_2018_2025_complete.parquet"

_ENV_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "joblib",
    "scikit-learn",
    "optuna",
    "shap",
    "xgboost",
    "catboost",
)


def _environment_manifest() -> dict:
    packages: dict[str, str] = {}
    for name in _ENV_PACKAGES:
        try:
            packages[name] = pkg_version(name)
        except PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "random_seed": RANDOM_SEED,
        "n_trials": N_TRIALS,
        "shap_n_models_threshold": SHAP_N_MODELS_THRESHOLD,
        "split_rule": "target_year_t_plus_h",
        "primary_horizon": PRIMARY_HORIZON,
    }


def main() -> None:
    print("Loading complete panel...")
    if not PANEL_PATH.exists():
        raise FileNotFoundError(PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    panel["year_month"] = pd.to_datetime(panel["year_month"])
    feature_groups = load_feature_groups(OUT_DIR / "feature_manifest.csv")

    print(f"Asserting target-year splits (h={PRIMARY_HORIZON})...")
    assert_target_year_splits(panel, horizon=PRIMARY_HORIZON)
    for h in FORECAST_HORIZONS:
        assert_target_year_splits(panel, horizon=h)

    # CV uses development target years only (impact years <= 2023).
    cv_mask = target_years(panel, PRIMARY_HORIZON) <= DEV_TARGET_MAX_YEAR
    cv_panel = panel.loc[cv_mask].reset_index(drop=True)
    folds = year_expanding_folds(cv_panel, horizon=PRIMARY_HORIZON)
    print(
        f"Folds: {len(folds)} | CV panel={cv_panel.shape} | "
        f"Full panel={panel.shape} | split=target_year(t+h)"
    )
    for f in folds:
        print(
            f"  fold {f.fold}: train targets {f.train_years} -> "
            f"val {f.valid_year} | n_train={len(f.train_idx)} n_val={len(f.val_idx)}"
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        study_name="ch9_drought_impact_automl",
    )
    objective = make_optuna_objective(
        cv_panel, feature_groups, folds, horizon=PRIMARY_HORIZON, classes=TOP_10_CLASSES
    )
    print(f"Starting Optuna search: {N_TRIALS} trials...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"Best trial #{study.best_trial.number}: {study.best_value:.4f}")
    print(f"Best model: {study.best_trial.params['model']}")

    trials_df = trials_dataframe(study)
    trials_df.to_csv(RESULT_DIR / "cv_trials.csv", index=False)

    best = best_config_from_study(study)
    best.save(RESULT_DIR / "best_config.json")
    pd.DataFrame({"feature": best.features}).to_csv(
        RESULT_DIR / "selected_features.csv", index=False
    )

    cv_best = cv_score_config(cv_panel, best, folds, horizon=PRIMARY_HORIZON)
    cv_best.to_csv(RESULT_DIR / "best_config_cv_folds.csv", index=False)
    print(cv_best.to_string(index=False))

    print("Retraining + test evaluation (per-horizon target-year splits)...")
    test_metrics, models_by_h, proba_by_h = evaluate_config_all_horizons(
        panel, best, horizons=FORECAST_HORIZONS, classes=TOP_10_CLASSES
    )
    test_metrics.to_csv(RESULT_DIR / "test_metrics_by_class_horizon.csv", index=False)
    macro_test = macro_from_class_metrics(test_metrics)
    macro_test.to_csv(RESULT_DIR / "test_macro_by_horizon.csv", index=False)
    print(macro_test.round(4).to_string(index=False))

    skill = test_metrics.copy()
    skill["pr_auc_skill"] = skill["pr_auc"] - skill["prevalence"]
    skill.to_csv(RESULT_DIR / "test_metrics_with_skill.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(macro_test["horizon"], macro_test["macro_pr_auc"], marker="o", label="macro AP")
    ax.plot(
        macro_test["horizon"],
        macro_test["macro_prevalence"],
        marker="s",
        linestyle="--",
        label="macro prevalence",
    )
    ax.set_xlabel("Forecast horizon h (months)")
    ax.set_ylabel("Score")
    ax.set_xticks(FORECAST_HORIZONS)
    ax.set_title(f"Test performance by horizon — {best.model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_test_macro_by_horizon.png", dpi=150)
    plt.close(fig)

    _, test_h1 = split_by_target_year(panel, horizon=PRIMARY_HORIZON)

    print("Computing SHAP importance...")
    shap_imp = shap_mean_abs_importance(
        models_by_h[PRIMARY_HORIZON],
        test_h1,
        best.features,
        model_name=best.model_name,
        max_samples=400,
        max_bg=200,
        shap_threshold=SHAP_N_MODELS_THRESHOLD,
    )
    shap_imp.to_csv(RESULT_DIR / "shap_importance_h1.csv", index=False)

    top_n = min(20, len(shap_imp))
    plot_df = shap_imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP| (averaged over classes)")
    ax.set_title(f"Global predictor importance — {best.model_name}, h={PRIMARY_HORIZON}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_shap_importance_h1.png", dpi=150)
    plt.close(fig)

    print("Temporal / spatial diagnostics (h=1)...")
    diag_paths = export_temporal_spatial_diagnostics(
        test_h1,
        proba_by_h[PRIMARY_HORIZON],
        TOP_10_CLASSES,
        RESULT_DIR,
        horizon=PRIMARY_HORIZON,
        notebook_dir=PACKAGE_ROOT,
    )
    for k, p in diag_paths.items():
        print(f"  {k}: {p.name}")

    n_complete = sum(1 for t in study.trials if str(t.state).endswith("COMPLETE"))
    n_pruned = sum(1 for t in study.trials if str(t.state).endswith("PRUNED"))
    n_fail = sum(1 for t in study.trials if str(t.state).endswith("FAIL"))

    summary = {
        "best_model": best.model_name,
        "cv_macro_pr_auc_h1": best.cv_macro_pr_auc,
        "n_features": len(best.features),
        "feature_groups": best.feature_groups_used,
        "group_subsets": best.group_subsets,
        "n_trials": N_TRIALS,
        "n_trials_complete": n_complete,
        "n_trials_pruned": n_pruned,
        "n_trials_fail": n_fail,
        "year_excluded": True,
        "split_rule": "target_year_t_plus_h",
        "test_macro_pr_auc_by_horizon": {
            int(r.horizon): float(r.macro_pr_auc) for r in macro_test.itertuples()
        },
        "diagnostics_exported": sorted(diag_paths.keys()),
        "environment": _environment_manifest(),
    }
    (RESULT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Done. Results in", RESULT_DIR)


if __name__ == "__main__":
    main()
