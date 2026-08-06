"""Run the Chapter 09 NUTS-3 AutoML pipeline (mirrors notebooks/03_automl_nuts3.ipynb)."""
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
    FORECAST_HORIZONS,
    TOP_10_CLASSES,
    best_config_from_study,
    cv_score_config,
    evaluate_config_all_horizons,
    export_temporal_spatial_diagnostics,
    load_feature_groups,
    macro_from_class_metrics,
    make_optuna_objective,
    shap_mean_abs_importance,
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
    }


def main() -> None:
    print("Loading data...")
    dev = pd.read_parquet(OUT_DIR / "dev_nuts3_forecast_2018_2023.parquet")
    test = pd.read_parquet(OUT_DIR / "test_nuts3_forecast_2024_2025.parquet")
    feature_groups = load_feature_groups(OUT_DIR / "feature_manifest.csv")
    dev["year_month"] = pd.to_datetime(dev["year_month"])
    test["year_month"] = pd.to_datetime(test["year_month"])

    folds = year_expanding_folds(dev)
    print(f"Folds: {len(folds)} | Dev={dev.shape} Test={test.shape}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        study_name="ch9_drought_impact_automl",
    )
    objective = make_optuna_objective(
        dev, feature_groups, folds, horizon=PRIMARY_HORIZON, classes=TOP_10_CLASSES
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

    cv_best = cv_score_config(dev, best, folds, horizon=PRIMARY_HORIZON)
    cv_best.to_csv(RESULT_DIR / "best_config_cv_folds.csv", index=False)
    print(cv_best.to_string(index=False))

    print("Retraining + test evaluation...")
    test_metrics, models_by_h, proba_by_h = evaluate_config_all_horizons(
        dev, test, best, horizons=FORECAST_HORIZONS, classes=TOP_10_CLASSES
    )
    test_metrics.to_csv(RESULT_DIR / "test_metrics_by_class_horizon.csv", index=False)
    macro_test = macro_from_class_metrics(test_metrics)
    macro_test.to_csv(RESULT_DIR / "test_macro_by_horizon.csv", index=False)
    print(macro_test.round(4).to_string(index=False))

    skill = test_metrics.copy()
    skill["pr_auc_skill"] = skill["pr_auc"] - skill["prevalence"]
    skill.to_csv(RESULT_DIR / "test_metrics_with_skill.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(macro_test["horizon"], macro_test["macro_pr_auc"], marker="o", label="macro PR-AUC")
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

    print("Computing SHAP importance...")
    shap_imp = shap_mean_abs_importance(
        models_by_h[PRIMARY_HORIZON],
        test,
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
        test,
        proba_by_h[PRIMARY_HORIZON],
        TOP_10_CLASSES,
        RESULT_DIR,
        horizon=PRIMARY_HORIZON,
        notebook_dir=PACKAGE_ROOT,
    )
    for k, p in diag_paths.items():
        print(f"  {k}: {p.name}")

    summary = {
        "best_model": best.model_name,
        "cv_macro_pr_auc_h1": best.cv_macro_pr_auc,
        "n_features": len(best.features),
        "feature_groups": best.feature_groups_used,
        "group_subsets": best.group_subsets,
        "n_trials": N_TRIALS,
        "year_excluded": True,
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
