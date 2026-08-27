"""AutoML search helpers for Chapters 08–09 drought-impact prediction.

Target-month year-based expanding-window CV, Binary Relevance multi-label
modelling, and Optuna joint search over models / predictor groups /
hyperparameters.

Temporal splits use the year of the *target* month t+h (impact occurrence),
not the predictor month t, so held-out periods do not leak into training labels.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

TOP_10_CLASSES = [
    "Crop Failure & Yield Reduction",
    "Reservoir & Surface Water Shortage",
    "Freshwater Ecosystem Degradation",
    "Groundwater Depletion",
    "Water Use Restrictions",
    "Forest Dieback & Vegetation Stress",
    "Agricultural Economic Loss",
    "Irrigation Shortage",
    "Water Supply & Sanitation Issues",
    "Wildfire Occurrence",
]

TOP_15_CLASSES = TOP_10_CLASSES + [
    "Wildfire Risk Increase",
    "Broader Economic Disruption",
    "Social Impacts",
    "Inland Waterway Disruption",
    "Wetland Loss",
]

# Predictability rule (development CV at h=1) — moderate bar
PREDICTABLE_MIN_SKILL = 0.15
PREDICTABLE_MIN_PR_AUC = 0.25

FORECAST_HORIZONS = [0, 1, 2, 3]
ID_COLS = ["year_month", "nuts3_id", "nuts3_name", "nuts2_id", "nuts2_name", "period"]

# Target-year expanding window matching thesis Figure (temporal CV).
# Years refer to the year of the impact target month t+h, not predictor month t.
YEAR_EXPANDING_FOLDS: list[tuple[list[int], int]] = [
    ([2018, 2019], 2020),
    ([2018, 2019, 2020], 2021),
    ([2018, 2019, 2020, 2021], 2022),
    ([2018, 2019, 2020, 2021, 2022], 2023),
]

DEV_TARGET_MAX_YEAR = 2023
TEST_TARGET_YEARS = (2024, 2025)

MODEL_NAMES = [
    "logistic_regression",
    "elastic_net",
    "random_forest",
    "xgboost",
    "catboost",
]

# Never used as a model feature (calendar year is for CV splits only).
EXCLUDED_FEATURES = {"year"}

ALL_GROUPS = [
    "meteo",
    "temporal",
    "static_land",
    "static_pop",
    "static_elev",
    "static_soil",
    "class_lag",
]

METEO_SUBSETS = ("all", "spi_only", "spei_only", "short", "long", "indices_only", "aux_only")
TEMPORAL_SUBSETS = ("seasonality", "persistence", "both")
SOIL_SUBSETS = ("all", "dominant_only", "organic", "sand_podzol", "clay_loam")
LAND_SUBSETS = ("all", "crop_only", "crop_tree", "water_only")
ELEV_SUBSETS = ("both", "mean_only", "std_only")
CLASS_LAG_SUBSETS = ("all", "agriculture", "water", "fire")

SOIL_ORGANIC = {"soil_veengronden", "soil_moerige_gronden"}
SOIL_SAND_PODZOL = {
    "soil_kalkhoudende_zandgronden",
    "soil_kalkloze_zandgronden",
    "soil_podzolgronden",
}
SOIL_CLAY_LOAM = {
    "soil_brikgronden",
    "soil_keileemgronden",
    "soil_leemgronden",
    "soil_rivierkleigronden",
    "soil_oude_rivierkleigronden",
    "soil_zeekleigronden",
}

CLASS_LAG_AGRICULTURE = {
    "Crop Failure & Yield Reduction_lag1",
    "Agricultural Economic Loss_lag1",
    "Irrigation Shortage_lag1",
    "Forest Dieback & Vegetation Stress_lag1",
}
CLASS_LAG_WATER = {
    "Reservoir & Surface Water Shortage_lag1",
    "Groundwater Depletion_lag1",
    "Water Use Restrictions_lag1",
    "Water Supply & Sanitation Issues_lag1",
    "Freshwater Ecosystem Degradation_lag1",
    "Inland Waterway Disruption_lag1",
    "Wetland Loss_lag1",
}
CLASS_LAG_FIRE = {
    "Wildfire Occurrence_lag1",
    "Wildfire Risk Increase_lag1",
}


@dataclass
class YearFold:
    fold: int
    train_years: list[int]
    valid_year: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class BestConfig:
    model_name: str
    features: list[str]
    params: dict[str, Any]
    cv_macro_pr_auc: float
    trial_number: int
    feature_groups_used: list[str] = field(default_factory=list)
    group_subsets: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "features": self.features,
            "params": self.params,
            "cv_macro_pr_auc": self.cv_macro_pr_auc,
            "trial_number": self.trial_number,
            "feature_groups_used": self.feature_groups_used,
            "group_subsets": self.group_subsets,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BestConfig:
        d = json.loads(path.read_text(encoding="utf-8"))
        # Backward compat with older best_config.json that stored meteo_subset only
        if "group_subsets" not in d and "meteo_subset" in d:
            d["group_subsets"] = {"meteo": d.pop("meteo_subset")}
        d.pop("meteo_subset", None)
        return cls(**d)


def load_feature_groups(manifest_path: Path) -> dict[str, list[str]]:
    """Load feature groups; permanently drop excluded columns such as `year`."""
    mf = pd.read_csv(manifest_path)
    groups: dict[str, list[str]] = {}
    for g, sub in mf.groupby("group"):
        cols = [c for c in sub["column"].tolist() if c not in EXCLUDED_FEATURES]
        if cols:
            groups[str(g)] = cols
    return groups


def _filter_meteo(cols: list[str], subset: str) -> list[str]:
    short_sfx = ("01", "03", "06")
    long_sfx = ("09", "12", "24")
    aux = {"soil_moisture_anom", "cdd_days"}

    def is_short(c: str) -> bool:
        return any(c.startswith(p) and c.endswith(s) for p in ("spei_", "spi_") for s in short_sfx)

    def is_long(c: str) -> bool:
        return any(c.startswith(p) and c.endswith(s) for p in ("spei_", "spi_") for s in long_sfx)

    if subset == "all":
        return list(cols)
    if subset == "spi_only":
        return [c for c in cols if c.startswith("spi_") or c in aux]
    if subset == "spei_only":
        return [c for c in cols if c.startswith("spei_") or c in aux]
    if subset == "short":
        return [c for c in cols if is_short(c) or c in aux]
    if subset == "long":
        return [c for c in cols if is_long(c) or c in aux]
    if subset == "indices_only":
        return [c for c in cols if c.startswith("spi_") or c.startswith("spei_")]
    if subset == "aux_only":
        return [c for c in cols if c in aux]
    raise ValueError(f"Unknown meteo subset: {subset}")


def _filter_temporal(cols: list[str], subset: str) -> list[str]:
    season = {"month_sin", "month_cos"}
    persist = {"has_impact_lag1"}
    # year already stripped in load_feature_groups
    if subset == "seasonality":
        return [c for c in cols if c in season]
    if subset == "persistence":
        return [c for c in cols if c in persist]
    if subset == "both":
        return [c for c in cols if c in season | persist]
    raise ValueError(f"Unknown temporal subset: {subset}")


def _filter_soil(cols: list[str], subset: str) -> list[str]:
    if subset == "all":
        return list(cols)
    if subset == "dominant_only":
        return [c for c in cols if c == "soil_dominant_fraction"]
    if subset == "organic":
        return [c for c in cols if c in SOIL_ORGANIC]
    if subset == "sand_podzol":
        return [c for c in cols if c in SOIL_SAND_PODZOL]
    if subset == "clay_loam":
        return [c for c in cols if c in SOIL_CLAY_LOAM]
    raise ValueError(f"Unknown soil subset: {subset}")


def _filter_land(cols: list[str], subset: str) -> list[str]:
    if subset == "all":
        return list(cols)
    if subset == "crop_only":
        return [c for c in cols if c == "crop_cover_fraction"]
    if subset == "crop_tree":
        return [c for c in cols if c in {"crop_cover_fraction", "tree_cover_fraction"}]
    if subset == "water_only":
        return [c for c in cols if c == "water_cover_fraction"]
    raise ValueError(f"Unknown land subset: {subset}")


def _filter_elev(cols: list[str], subset: str) -> list[str]:
    if subset == "both":
        return list(cols)
    if subset == "mean_only":
        return [c for c in cols if c == "elevation_mean"]
    if subset == "std_only":
        return [c for c in cols if c == "elevation_std"]
    raise ValueError(f"Unknown elev subset: {subset}")


def _filter_class_lag(cols: list[str], subset: str) -> list[str]:
    if subset == "all":
        return list(cols)
    if subset == "agriculture":
        return [c for c in cols if c in CLASS_LAG_AGRICULTURE]
    if subset == "water":
        return [c for c in cols if c in CLASS_LAG_WATER]
    if subset == "fire":
        return [c for c in cols if c in CLASS_LAG_FIRE]
    raise ValueError(f"Unknown class_lag subset: {subset}")


def class_target(cls: str, horizon: int) -> str:
    return f"{cls}_h{horizon}"


def target_year_month(year_month: pd.Series | pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Calendar month of the impact target: t_target = t + h."""
    ym = pd.to_datetime(year_month)
    if horizon == 0:
        return ym.dt.to_period("M").dt.to_timestamp()
    return (ym.dt.to_period("M") + horizon).dt.to_timestamp()


def target_years(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Year of t+h for each row (used for temporal train/val/test masks)."""
    return target_year_month(df["year_month"], horizon).dt.year.astype(int)


def year_expanding_folds(df: pd.DataFrame, *, horizon: int = 1) -> list[YearFold]:
    """Build expanding-window folds using the year of the target month t+h.

    Training rows have year(t+h) in ``train_years``; validation rows have
    year(t+h) == ``valid_year``. This automatically excludes predictors whose
    labels fall inside the validation year (e.g. Dec 2019 at h=1 → Jan 2020).
    """
    years = target_years(df, horizon)

    folds: list[YearFold] = []
    for i, (train_years, valid_year) in enumerate(YEAR_EXPANDING_FOLDS, start=1):
        tr_mask = years.isin(train_years)
        va_mask = years.eq(valid_year)
        tr_idx = np.where(tr_mask.to_numpy())[0]
        va_idx = np.where(va_mask.to_numpy())[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append(
            YearFold(
                fold=i,
                train_years=list(train_years),
                valid_year=valid_year,
                train_idx=tr_idx,
                val_idx=va_idx,
            )
        )
    return folds


def split_by_target_year(
    df: pd.DataFrame,
    *,
    horizon: int,
    max_train_year: int = DEV_TARGET_MAX_YEAR,
    test_years: tuple[int, ...] = TEST_TARGET_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel by year of t+h into development train and independent test."""
    years = target_years(df, horizon)
    train_df = df.loc[years <= max_train_year].copy()
    test_df = df.loc[years.isin(test_years)].copy()
    return train_df, test_df


def assert_target_year_splits(df: pd.DataFrame, *, horizon: int = 1) -> None:
    """Fail fast if target-year CV/test masks still leak labels across boundaries."""
    folds = year_expanding_folds(df, horizon=horizon)
    years = target_years(df, horizon)
    ym = pd.to_datetime(df["year_month"])

    for fold in folds:
        train_tgt = years.iloc[fold.train_idx]
        if (train_tgt == fold.valid_year).any():
            raise AssertionError(
                f"h={horizon} fold {fold.fold}: train rows have target year "
                f"{fold.valid_year} (label leakage)"
            )
        # At h=1, Dec of the last train calendar year must be in validation
        # (target = next January in valid_year).
        if horizon >= 1 and fold.valid_year - 1 in fold.train_years:
            boundary = ym.eq(pd.Timestamp(fold.valid_year - 1, 12, 1))
            boundary_idx = set(np.where(boundary.to_numpy())[0])
            val_set = set(fold.val_idx.tolist())
            train_set = set(fold.train_idx.tolist())
            leaked = boundary_idx & train_set
            if leaked:
                raise AssertionError(
                    f"h={horizon} fold {fold.fold}: Dec {fold.valid_year - 1} "
                    f"predictors still in train ({len(leaked)} rows)"
                )
            if boundary_idx and not (boundary_idx <= val_set):
                missing = boundary_idx - val_set
                raise AssertionError(
                    f"h={horizon} fold {fold.fold}: Dec {fold.valid_year - 1} "
                    f"predictors not fully in validation ({len(missing)} missing)"
                )

    train_df, test_df = split_by_target_year(df, horizon=horizon)
    train_years = target_years(train_df, horizon)
    test_years = target_years(test_df, horizon)
    if (train_years >= min(TEST_TARGET_YEARS)).any():
        raise AssertionError(
            f"h={horizon}: final train contains target years >= {min(TEST_TARGET_YEARS)}"
        )
    if horizon >= 1:
        dec_train = pd.to_datetime(train_df["year_month"]).eq(pd.Timestamp(DEV_TARGET_MAX_YEAR, 12, 1))
        if dec_train.any():
            raise AssertionError(
                f"h={horizon}: Dec {DEV_TARGET_MAX_YEAR} predictors still in final train"
            )
        dec_test = pd.to_datetime(test_df["year_month"]).eq(pd.Timestamp(DEV_TARGET_MAX_YEAR, 12, 1))
        if not dec_test.any():
            raise AssertionError(
                f"h={horizon}: Dec {DEV_TARGET_MAX_YEAR} predictors missing from test "
                f"(should predict Jan {min(TEST_TARGET_YEARS)})"
            )
    if len(test_df) == 0 or len(train_df) == 0:
        raise AssertionError(f"h={horizon}: empty train ({len(train_df)}) or test ({len(test_df)})")


def resolve_features(
    groups: dict[str, list[str]],
    *,
    use_groups: list[str],
    group_subsets: dict[str, str] | None = None,
) -> list[str]:
    """Assemble feature list from enabled groups and within-group subsets."""
    group_subsets = group_subsets or {}
    feats: list[str] = []

    for g in use_groups:
        cols = list(groups.get(g, []))
        if not cols:
            continue
        subset = group_subsets.get(g, "all")
        if g == "meteo":
            cols = _filter_meteo(cols, subset)
        elif g == "temporal":
            cols = _filter_temporal(cols, subset)
        elif g == "static_soil":
            cols = _filter_soil(cols, subset)
        elif g == "static_land":
            cols = _filter_land(cols, subset)
        elif g == "static_elev":
            cols = _filter_elev(cols, subset)
        elif g == "class_lag":
            cols = _filter_class_lag(cols, subset)
        elif g == "static_pop":
            cols = list(cols)  # single feature when enabled
        feats.extend(cols)

    # Drop excluded columns defensively; stable unique order
    return list(dict.fromkeys(c for c in feats if c not in EXCLUDED_FEATURES))


def suggest_feature_config(
    trial, groups: dict[str, list[str]]
) -> tuple[list[str], list[str], dict[str, str]]:
    """Sample optional groups + within-group subsets.

    Returns (features, groups_used, group_subsets).
    """
    used: list[str] = []
    group_subsets: dict[str, str] = {}

    for g in ALL_GROUPS:
        if g not in groups:
            continue
        if not trial.suggest_categorical(f"use_{g}", [True, False]):
            continue
        used.append(g)

        if g == "meteo":
            group_subsets[g] = trial.suggest_categorical("meteo_subset", list(METEO_SUBSETS))
        elif g == "temporal":
            group_subsets[g] = trial.suggest_categorical("temporal_subset", list(TEMPORAL_SUBSETS))
        elif g == "static_soil":
            group_subsets[g] = trial.suggest_categorical("soil_subset", list(SOIL_SUBSETS))
        elif g == "static_land":
            group_subsets[g] = trial.suggest_categorical("land_subset", list(LAND_SUBSETS))
        elif g == "static_elev":
            group_subsets[g] = trial.suggest_categorical("elev_subset", list(ELEV_SUBSETS))
        elif g == "class_lag":
            group_subsets[g] = trial.suggest_categorical("class_lag_subset", list(CLASS_LAG_SUBSETS))
        elif g == "static_pop":
            group_subsets[g] = "all"

    features = resolve_features(groups, use_groups=used, group_subsets=group_subsets)
    return features, used, group_subsets


def pos_weight(y: pd.Series) -> float:
    p = float(np.asarray(y).mean())
    return (1.0 - p) / max(p, 1e-6)


def build_estimator(model_name: str, params: dict[str, Any], y_tr: pd.Series):
    """Build a sklearn-compatible estimator (possibly a Pipeline)."""
    if model_name == "logistic_regression":
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "elastic_net":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params.get("C", 0.1)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            scale_pos_weight=pos_weight(y_tr),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(params.get("iterations", 200)),
            depth=int(params.get("depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            auto_class_weights="Balanced",
            random_seed=42,
            # Fixed thread count for deterministic multi-threaded training.
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
        )

    raise ValueError(f"Unknown model: {model_name}")


def suggest_model_params(trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-4, 20.0, log=True)}

    if model_name == "elastic_net":
        return {
            "C": trial.suggest_float("C", 1e-4, 20.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    if model_name == "random_forest":
        depth = trial.suggest_int("max_depth", 3, 16)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": depth,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
        }

    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        }

    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 600, step=50),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 15.0),
        }

    raise ValueError(model_name)


def evaluate_binary(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    out: dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["brier"] = float(brier_score_loss(y_true, y_score))
    out["prevalence"] = float(y_true.mean())
    return out


def fit_predict_proba(
    model_name: str,
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
) -> tuple[np.ndarray, Any]:
    est = build_estimator(model_name, params, y_tr)
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_va)[:, 1]
    return np.asarray(proba, dtype=float), est


def macro_pr_auc_fold(
    df: pd.DataFrame,
    fold: YearFold,
    features: list[str],
    model_name: str,
    params: dict[str, Any],
    *,
    horizon: int = 1,
    classes: list[str] | None = None,
) -> float:
    """Mean PR-AUC over classes with both labels present in the validation fold."""
    classes = classes or TOP_10_CLASSES
    tr = df.iloc[fold.train_idx]
    va = df.iloc[fold.val_idx]
    feats = [c for c in features if c in df.columns]
    if not feats:
        return float("nan")

    scores: list[float] = []
    for cls in classes:
        target = class_target(cls, horizon)
        if target not in df.columns:
            continue
        y_tr = tr[target].astype(int)
        y_va = va[target].astype(int)
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue
        try:
            proba, _ = fit_predict_proba(model_name, params, tr[feats], y_tr, va[feats])
            scores.append(float(average_precision_score(y_va, proba)))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else float("nan")


def make_optuna_objective(
    dev: pd.DataFrame,
    groups: dict[str, list[str]],
    folds: list[YearFold],
    *,
    horizon: int = 1,
    classes: list[str] | None = None,
):
    """Return Optuna objective maximizing mean fold macro PR-AUC at the given horizon."""
    import optuna

    classes = classes or TOP_10_CLASSES

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", MODEL_NAMES)
        features, used_groups, group_subsets = suggest_feature_config(trial, groups)
        if not features:
            return 0.0

        params = suggest_model_params(trial, model_name)

        # Record config before the fold loop so pruned trials keep metadata.
        trial.set_user_attr("features", features)
        trial.set_user_attr("feature_groups_used", used_groups)
        trial.set_user_attr("group_subsets", group_subsets)
        trial.set_user_attr("model_params", params)
        trial.set_user_attr("n_features", len(features))

        fold_scores: list[float] = []
        for fold in folds:
            score = macro_pr_auc_fold(
                dev,
                fold,
                features,
                model_name,
                params,
                horizon=horizon,
                classes=classes,
            )
            if np.isnan(score):
                continue
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), fold.fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_scores:
            return 0.0

        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        return mean_score

    return objective


def best_config_from_study(study) -> BestConfig:
    t = study.best_trial
    return BestConfig(
        model_name=t.params["model"],
        features=list(t.user_attrs["features"]),
        params=dict(t.user_attrs["model_params"]),
        cv_macro_pr_auc=float(t.value),
        trial_number=int(t.number),
        feature_groups_used=list(t.user_attrs["feature_groups_used"]),
        group_subsets=dict(t.user_attrs.get("group_subsets", {})),
    )


def trials_dataframe(study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        # COMPLETE trials have t.value; PRUNED trials often have value=None but
        # retain intermediate reports — use the last report so pruning is visible.
        value = t.value
        if value is None and t.intermediate_values:
            value = float(t.intermediate_values[max(t.intermediate_values)])
        if value is None:
            continue
        subsets = t.user_attrs.get("group_subsets", {}) or {}
        row = {
            "trial": t.number,
            "state": str(t.state),
            "macro_pr_auc_h1": value,
            "model": t.params.get("model"),
            "n_features": t.user_attrs.get("n_features"),
            "feature_groups": ",".join(t.user_attrs.get("feature_groups_used", [])),
            "group_subsets": json.dumps(subsets, sort_keys=True),
        }
        for k, v in subsets.items():
            row[f"subset_{k}"] = v
        for k, v in t.params.items():
            if k != "model":
                row[f"param_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_pr_auc_h1", ascending=False)


def train_binary_relevance(
    train_df: pd.DataFrame,
    features: list[str],
    model_name: str,
    params: dict[str, Any],
    *,
    horizon: int,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """Fit one classifier per class for a fixed horizon. Returns {class: estimator}."""
    classes = classes or TOP_10_CLASSES
    feats = [c for c in features if c in train_df.columns]
    models: dict[str, Any] = {}
    for cls in classes:
        target = class_target(cls, horizon)
        y = train_df[target].astype(int)
        if y.nunique() < 2:
            continue
        est = build_estimator(model_name, params, y)
        est.fit(train_df[feats], y)
        models[cls] = est
    return models


def predict_binary_relevance(
    models: dict[str, Any],
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    feats = [c for c in features if c in df.columns]
    out = {}
    for cls, est in models.items():
        out[cls] = est.predict_proba(df[feats])[:, 1]
    return pd.DataFrame(out, index=df.index)


def evaluate_multilabel(
    df: pd.DataFrame,
    proba_df: pd.DataFrame,
    *,
    horizon: int,
    classes: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    classes = classes or [c for c in TOP_10_CLASSES if c in proba_df.columns]
    rows = []
    for cls in classes:
        target = class_target(cls, horizon)
        y_true = df[target].astype(int)
        y_score = proba_df[cls].to_numpy()
        y_pred = (y_score >= threshold).astype(int)
        metrics = evaluate_binary(y_true, y_pred, y_score)
        rows.append({"class": cls, "horizon": horizon, **metrics})
    return pd.DataFrame(rows)


def macro_from_class_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-class metrics to macro averages by horizon."""
    num_cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "prevalence"]
    return (
        metrics_df.groupby("horizon", as_index=False)[num_cols]
        .mean()
        .rename(columns={c: f"macro_{c}" for c in num_cols})
    )


def evaluate_config_all_horizons(
    panel: pd.DataFrame,
    config: BestConfig,
    *,
    horizons: list[int] | None = None,
    classes: list[str] | None = None,
    max_train_year: int = DEV_TARGET_MAX_YEAR,
    test_years: tuple[int, ...] = TEST_TARGET_YEARS,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], dict[int, pd.DataFrame]]:
    """Retrain per horizon with target-year train/test masks on the full panel.

    For each horizon h, train on rows with year(t+h) <= max_train_year and
    score rows with year(t+h) in test_years.
    """
    horizons = horizons or FORECAST_HORIZONS
    classes = classes or TOP_10_CLASSES
    all_metrics = []
    models_by_h: dict[int, dict[str, Any]] = {}
    proba_by_h: dict[int, pd.DataFrame] = {}

    for h in horizons:
        train_df, test_df = split_by_target_year(
            panel,
            horizon=h,
            max_train_year=max_train_year,
            test_years=test_years,
        )
        models = train_binary_relevance(
            train_df,
            config.features,
            config.model_name,
            config.params,
            horizon=h,
            classes=classes,
        )
        proba = predict_binary_relevance(models, test_df, config.features)
        metrics = evaluate_multilabel(test_df, proba, horizon=h, classes=list(models.keys()))
        all_metrics.append(metrics)
        models_by_h[h] = models
        proba_by_h[h] = proba

    return pd.concat(all_metrics, ignore_index=True), models_by_h, proba_by_h


def cv_score_config(
    dev: pd.DataFrame,
    config: BestConfig,
    folds: list[YearFold],
    *,
    horizon: int = 1,
    classes: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        score = macro_pr_auc_fold(
            dev,
            fold,
            config.features,
            config.model_name,
            config.params,
            horizon=horizon,
            classes=classes,
        )
        rows.append(
            {
                "fold": fold.fold,
                "train_years": "-".join(map(str, fold.train_years)),
                "valid_year": fold.valid_year,
                "macro_pr_auc": score,
                "horizon": horizon,
            }
        )
    return pd.DataFrame(rows)


def per_class_cv_under_config(
    dev: pd.DataFrame,
    config: BestConfig,
    folds: list[YearFold],
    classes: list[str],
    *,
    horizon: int = 1,
) -> pd.DataFrame:
    """Per-class mean CV PR-AUC / prevalence / skill under a fixed shared BestConfig."""
    rows = []
    for cls in classes:
        fold_scores: list[float] = []
        for fold in folds:
            score = _class_fold_pr_auc(
                dev,
                fold,
                config.features,
                config.model_name,
                config.params,
                class_name=cls,
                horizon=horizon,
            )
            if score is not None:
                fold_scores.append(score)
        cv_pr = float(np.mean(fold_scores)) if fold_scores else float("nan")
        cv_prev = _cv_prevalence(dev, folds, cls, horizon)
        cv_skill = cv_pr - cv_prev if not (np.isnan(cv_pr) or np.isnan(cv_prev)) else float("nan")
        rows.append(
            {
                "class": cls,
                "cv_pr_auc": cv_pr,
                "cv_prevalence": cv_prev,
                "cv_skill": cv_skill,
                "n_folds_scored": len(fold_scores),
                "model": config.model_name,
                "n_features": len(config.features),
                "horizon": horizon,
            }
        )
    return pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Temporal / spatial diagnostics (post-hoc XAI on held-out predictions)
# ---------------------------------------------------------------------------

DIAG_MIN_POS = 5
DROUGHT_BIN_ORDER = ["dry", "near_normal", "wet"]


def drought_bin_from_spei(spei: pd.Series | np.ndarray) -> pd.Series:
    """Bin SPEI-12 into dry / near_normal / wet."""
    s = pd.Series(spei, dtype=float)
    out = pd.Series(np.where(s <= -1.0, "dry", np.where(s >= 1.0, "wet", "near_normal")), index=s.index)
    return out


def binary_slice_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    min_pos: int = DIAG_MIN_POS,
) -> dict[str, float]:
    """Slice metrics: always Brier/prevalence; PR-AUC/skill only if both classes and enough positives."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = int(len(y_true))
    n_pos = int(y_true.sum())
    prevalence = float(y_true.mean()) if n else float("nan")
    mean_pred = float(np.mean(y_score)) if n else float("nan")
    brier = float(brier_score_loss(y_true, y_score)) if n else float("nan")

    pr_auc = float("nan")
    skill = float("nan")
    if n > 0 and n_pos >= min_pos and len(np.unique(y_true)) > 1:
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
            skill = pr_auc - prevalence
        except Exception:
            pr_auc = float("nan")
            skill = float("nan")

    return {
        "n": float(n),
        "n_pos": float(n_pos),
        "prevalence": prevalence,
        "mean_pred": mean_pred,
        "brier": brier,
        "pr_auc": pr_auc,
        "skill": skill,
    }


def build_prediction_long(
    test_df: pd.DataFrame,
    proba_df: pd.DataFrame,
    classes: list[str],
    *,
    horizon: int = 1,
    spei_col: str = "spei_12",
) -> pd.DataFrame:
    """Long frame of y_true / y_score per class with time, region, and drought bin."""
    base = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(base["year_month"]):
        base["year_month"] = pd.to_datetime(base["year_month"])
    base["month"] = base["year_month"].dt.month.astype(int)
    if spei_col in base.columns:
        base["spei_12"] = base[spei_col].astype(float)
        base["drought_bin"] = drought_bin_from_spei(base["spei_12"])
    else:
        base["spei_12"] = np.nan
        base["drought_bin"] = "near_normal"

    frames: list[pd.DataFrame] = []
    for cls in classes:
        if cls not in proba_df.columns:
            continue
        target = class_target(cls, horizon)
        if target not in base.columns:
            continue
        part = pd.DataFrame(
            {
                "year_month": base["year_month"].values,
                "month": base["month"].values,
                "nuts3_id": base["nuts3_id"].values,
                "nuts3_name": base["nuts3_name"].values if "nuts3_name" in base.columns else base["nuts3_id"].values,
                "class": cls,
                "y_true": base[target].astype(int).values,
                "y_score": proba_df[cls].reindex(base.index).to_numpy(dtype=float),
                "spei_12": base["spei_12"].values,
                "drought_bin": base["drought_bin"].values,
                "horizon": horizon,
            }
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame(
            columns=[
                "year_month",
                "month",
                "nuts3_id",
                "nuts3_name",
                "class",
                "y_true",
                "y_score",
                "spei_12",
                "drought_bin",
                "horizon",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def metrics_by_group(
    long_df: pd.DataFrame,
    group_cols: list[str],
    *,
    min_pos: int = DIAG_MIN_POS,
) -> pd.DataFrame:
    """Compute binary_slice_metrics for each group (includes class if present in group_cols)."""
    metric_cols = ["n", "n_pos", "prevalence", "mean_pred", "brier", "pr_auc", "skill"]
    if long_df.empty:
        return pd.DataFrame(columns=list(group_cols) + metric_cols)
    rows = []
    for keys, sub in long_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = binary_slice_metrics(sub["y_true"], sub["y_score"], min_pos=min_pos)
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_metrics_over_classes(
    class_metrics: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Macro-average PR-AUC/skill over classes with valid PR-AUC; mean Brier/prevalence over all classes."""
    if class_metrics.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in class_metrics.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = g["pr_auc"].notna()
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update(
            {
                "n": float(g["n"].sum()),
                "n_pos": float(g["n_pos"].sum()),
                "n_classes": float(len(g)),
                "n_classes_pr": float(valid.sum()),
                "macro_prevalence": float(g["prevalence"].mean()),
                "macro_mean_pred": float(g["mean_pred"].mean()),
                "macro_brier": float(g["brier"].mean()),
                "macro_pr_auc": float(g.loc[valid, "pr_auc"].mean()) if valid.any() else float("nan"),
                "macro_skill": float(g.loc[valid, "skill"].mean()) if valid.any() else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_nuts_geojson(notebook_dir: Path | None = None) -> Path:
    """Locate NL NUTS geojson relative to chapter_08_09 package / notebook dir."""
    candidates: list[Path] = []
    here = Path(__file__).resolve().parent  # chapter_08_09/src
    package_root = here.parent  # chapter_08_09
    candidates.append(package_root / "data" / "input" / "nuts_nl_simplified.geojson")
    if notebook_dir is not None:
        nd = Path(notebook_dir)
        candidates.append(nd / "data" / "input" / "nuts_nl_simplified.geojson")
        candidates.append(nd.parent / "data" / "input" / "nuts_nl_simplified.geojson")
        candidates.append(nd / "nuts_nl_simplified.geojson")
        # Legacy monorepo locations (fallback)
        candidates.append(
            nd.parent
            / "chapter 6 geocoding"
            / "NUTS-3-Coder"
            / "data"
            / "geo"
            / "nuts_nl_simplified.geojson"
        )
    repo_root = package_root.parent
    candidates.append(repo_root / "chapter_06" / "data" / "geo" / "nuts_nl_simplified.geojson")
    candidates.append(
        repo_root
        / "Thesis specific chapters"
        / "chapter 6 geocoding"
        / "NUTS-3-Coder"
        / "data"
        / "geo"
        / "nuts_nl_simplified.geojson"
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"NUTS geojson not found. Tried: {candidates}")


def plot_nuts3_choropleth(
    regional_df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    *,
    title: str,
    notebook_dir: Path | None = None,
    cmap: str = "RdYlGn",
) -> Path:
    """Save a NUTS-3 choropleth for a regional metrics table."""
    import geopandas as gpd
    import matplotlib.pyplot as plt

    geo_path = resolve_nuts_geojson(notebook_dir)
    gdf = gpd.read_file(geo_path)
    if "LEVL_CODE" in gdf.columns:
        gdf = gdf[gdf["LEVL_CODE"] == 3].copy()
    id_col = "NUTS_ID" if "NUTS_ID" in gdf.columns else "nuts3_id"
    gdf = gdf.rename(columns={id_col: "nuts3_id"})
    merged = gdf.merge(regional_df, on="nuts3_id", how="left")

    fig, ax = plt.subplots(figsize=(8, 9))
    merged.plot(
        column=value_col,
        ax=ax,
        legend=True,
        cmap=cmap,
        missing_kwds={"color": "lightgrey", "label": "NA"},
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_temporal_spatial_diagnostics(
    test_df: pd.DataFrame,
    proba_df: pd.DataFrame,
    classes: list[str],
    result_dir: Path,
    *,
    horizon: int = 1,
    notebook_dir: Path | None = None,
    min_pos: int = DIAG_MIN_POS,
) -> dict[str, Path]:
    """Compute temporal/spatial/class diagnostics and write CSVs + figures. Returns output paths."""
    import matplotlib.pyplot as plt

    result_dir = Path(result_dir)
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    long_df = build_prediction_long(test_df, proba_df, classes, horizon=horizon)
    long_path = result_dir / "diag_prediction_long_h1.csv"
    long_df.to_csv(long_path, index=False)
    paths["prediction_long"] = long_path

    # --- Temporal: year_month ---
    by_ym_cls = metrics_by_group(long_df, ["year_month", "class"], min_pos=min_pos)
    by_ym = macro_metrics_over_classes(by_ym_cls, ["year_month"]).sort_values("year_month")
    p = result_dir / "diag_temporal_by_month.csv"
    by_ym.to_csv(p, index=False)
    paths["temporal_by_month"] = p

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_ym["year_month"], by_ym["macro_skill"], marker="o", label="macro skill")
    ax.plot(by_ym["year_month"], by_ym["macro_brier"], marker="s", linestyle="--", label="macro Brier")
    ax.set_xlabel("Year-month")
    ax.set_ylabel("Score")
    ax.set_title(f"Test temporal performance (h={horizon})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fp = fig_dir / "fig_diag_temporal_by_month.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths["fig_temporal_by_month"] = fp

    # --- Temporal: calendar month ---
    by_cal_cls = metrics_by_group(long_df, ["month", "class"], min_pos=min_pos)
    by_cal = macro_metrics_over_classes(by_cal_cls, ["month"]).sort_values("month")
    p = result_dir / "diag_temporal_by_calendar_month.csv"
    by_cal.to_csv(p, index=False)
    paths["temporal_by_calendar_month"] = p

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_cal["month"] - 0.15, by_cal["macro_skill"], width=0.3, label="macro skill", color="#4C72B0")
    ax.bar(by_cal["month"] + 0.15, by_cal["macro_brier"], width=0.3, label="macro Brier", color="#DD8452")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Score")
    ax.set_title(f"Test performance by calendar month (h={horizon})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fp = fig_dir / "fig_diag_temporal_by_calendar_month.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths["fig_temporal_by_calendar_month"] = fp

    # --- Temporal: drought bin ---
    by_dry_cls = metrics_by_group(long_df, ["drought_bin", "class"], min_pos=min_pos)
    by_dry = macro_metrics_over_classes(by_dry_cls, ["drought_bin"])
    by_dry["drought_bin"] = pd.Categorical(by_dry["drought_bin"], categories=DROUGHT_BIN_ORDER, ordered=True)
    by_dry = by_dry.sort_values("drought_bin")
    p = result_dir / "diag_temporal_by_drought_bin.csv"
    by_dry.to_csv(p, index=False)
    paths["temporal_by_drought_bin"] = p

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(by_dry))
    ax.bar(x - 0.15, by_dry["macro_skill"], width=0.3, label="macro skill", color="#4C72B0")
    ax.bar(x + 0.15, by_dry["macro_brier"], width=0.3, label="macro Brier", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in by_dry["drought_bin"]])
    ax.set_xlabel("SPEI-12 drought bin")
    ax.set_ylabel("Score")
    ax.set_title(f"Test performance by drought regime (h={horizon})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fp = fig_dir / "fig_diag_temporal_by_drought_bin.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths["fig_temporal_by_drought_bin"] = fp

    # --- Spatial: NUTS-3 ---
    by_nuts_cls = metrics_by_group(long_df, ["nuts3_id", "nuts3_name", "class"], min_pos=min_pos)
    by_nuts = macro_metrics_over_classes(by_nuts_cls, ["nuts3_id", "nuts3_name"])
    # Rank: prefer skill; fallback lower Brier (negate for sort)
    by_nuts["rank_score"] = by_nuts["macro_skill"].fillna(-by_nuts["macro_brier"])
    by_nuts = by_nuts.sort_values("rank_score", ascending=False).reset_index(drop=True)
    p = result_dir / "diag_spatial_by_nuts3.csv"
    by_nuts.to_csv(p, index=False)
    paths["spatial_by_nuts3"] = p

    try:
        fp = plot_nuts3_choropleth(
            by_nuts,
            "macro_skill",
            fig_dir / "fig_spatial_skill_choropleth.png",
            title=f"Regional macro skill (test, h={horizon})",
            notebook_dir=notebook_dir,
        )
        paths["fig_spatial_choropleth"] = fp
    except Exception as exc:
        print(f"Choropleth skipped: {exc}")

    # --- Class interactions ---
    class_cal = metrics_by_group(long_df, ["class", "month"], min_pos=min_pos)
    p = result_dir / "diag_class_by_calendar_month.csv"
    class_cal.to_csv(p, index=False)
    paths["class_by_calendar_month"] = p

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, ttl in zip(axes, ["skill", "brier"], ["Skill", "Brier"]):
        piv = class_cal.pivot(index="class", columns="month", values=col)
        piv = piv.reindex(columns=range(1, 13))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn_r" if col == "brier" else "RdYlGn")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(list(piv.index), fontsize=8)
        ax.set_xticks(range(12))
        ax.set_xticklabels(list(range(1, 13)))
        ax.set_xlabel("Calendar month")
        ax.set_title(f"Class × month — {ttl} (h={horizon})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fp = fig_dir / "fig_diag_class_by_calendar_month.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths["fig_class_by_calendar_month"] = fp

    class_dry = metrics_by_group(long_df, ["class", "drought_bin"], min_pos=min_pos)
    p = result_dir / "diag_class_by_drought_bin.csv"
    class_dry.to_csv(p, index=False)
    paths["class_by_drought_bin"] = p

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, col, ttl in zip(axes, ["skill", "brier"], ["Skill", "Brier"]):
        piv = class_dry.pivot(index="class", columns="drought_bin", values=col)
        piv = piv.reindex(columns=[c for c in DROUGHT_BIN_ORDER if c in piv.columns])
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn_r" if col == "brier" else "RdYlGn")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(list(piv.index), fontsize=8)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(list(piv.columns), rotation=30, ha="right")
        ax.set_title(f"Class × drought bin — {ttl}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fp = fig_dir / "fig_diag_class_by_drought_bin.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths["fig_class_by_drought_bin"] = fp

    class_nuts = metrics_by_group(long_df, ["class", "nuts3_id", "nuts3_name"], min_pos=min_pos)
    p = result_dir / "diag_class_by_nuts3.csv"
    class_nuts.to_csv(p, index=False)
    paths["class_by_nuts3"] = p

    # Top/bottom regions per class
    top_bottom_rows = []
    for cls, g in class_nuts.groupby("class"):
        g2 = g.copy()
        g2["rank_score"] = g2["skill"].fillna(-g2["brier"])
        g2 = g2.sort_values("rank_score", ascending=False)
        for kind, part in [("best", g2.head(3)), ("worst", g2.tail(3).iloc[::-1])]:
            for r in part.itertuples(index=False):
                top_bottom_rows.append(
                    {
                        "class": cls,
                        "kind": kind,
                        "nuts3_id": r.nuts3_id,
                        "nuts3_name": r.nuts3_name,
                        "skill": r.skill,
                        "brier": r.brier,
                        "pr_auc": r.pr_auc,
                        "n_pos": r.n_pos,
                    }
                )
    top_bottom = pd.DataFrame(top_bottom_rows)
    p = result_dir / "diag_class_nuts3_top_bottom.csv"
    top_bottom.to_csv(p, index=False)
    paths["class_nuts3_top_bottom"] = p

    # --- JJA regional ---
    jja = long_df.loc[long_df["month"].isin([6, 7, 8])].copy()
    jja_cls = metrics_by_group(jja, ["nuts3_id", "nuts3_name", "class"], min_pos=min_pos)
    jja_nuts = macro_metrics_over_classes(jja_cls, ["nuts3_id", "nuts3_name"])
    jja_nuts["rank_score"] = jja_nuts["macro_skill"].fillna(-jja_nuts["macro_brier"])
    jja_nuts = jja_nuts.sort_values("rank_score", ascending=False).reset_index(drop=True)
    p = result_dir / "diag_spatial_jja_by_nuts3.csv"
    jja_nuts.to_csv(p, index=False)
    paths["spatial_jja_by_nuts3"] = p

    try:
        fp = plot_nuts3_choropleth(
            jja_nuts,
            "macro_skill",
            fig_dir / "fig_spatial_skill_choropleth_jja.png",
            title=f"Regional macro skill — JJA only (h={horizon})",
            notebook_dir=notebook_dir,
        )
        paths["fig_spatial_choropleth_jja"] = fp
    except Exception as exc:
        print(f"JJA choropleth skipped: {exc}")

    # Compare all-months vs JJA ranks
    cmp = by_nuts[["nuts3_id", "nuts3_name", "macro_skill", "macro_brier", "rank_score"]].rename(
        columns={
            "macro_skill": "skill_all",
            "macro_brier": "brier_all",
            "rank_score": "rank_score_all",
        }
    )
    cmp = cmp.merge(
        jja_nuts[["nuts3_id", "macro_skill", "macro_brier", "rank_score"]].rename(
            columns={
                "macro_skill": "skill_jja",
                "macro_brier": "brier_jja",
                "rank_score": "rank_score_jja",
            }
        ),
        on="nuts3_id",
        how="outer",
    )
    cmp["delta_skill_jja_minus_all"] = cmp["skill_jja"] - cmp["skill_all"]
    p = result_dir / "diag_spatial_all_vs_jja.csv"
    cmp.to_csv(p, index=False)
    paths["spatial_all_vs_jja"] = p

    return paths


def shap_mean_abs_importance(
    models: dict[str, Any],
    background_df: pd.DataFrame,
    features: list[str],
    *,
    model_name: str,
    max_samples: int = 400,
    max_bg: int = 200,
    shap_threshold: float = 0.05,
) -> pd.DataFrame:
    """Aggregate mean |SHAP| across Binary Relevance class models at one horizon.

    ``n_models`` counts class-models where a feature's mean |SHAP| exceeds
    ``shap_threshold``. ``n_models_evaluated`` is the number of class-models
    that produced SHAP values successfully.
    """
    import shap

    feats = [c for c in features if c in background_df.columns]
    if len(background_df) > max_samples:
        sample = background_df.sample(max_samples, random_state=42)
    else:
        sample = background_df
    X = sample[feats]

    if len(background_df) > max_bg:
        bg = background_df.sample(max_bg, random_state=0)[feats]
    else:
        bg = background_df[feats]

    per_model_means: list[np.ndarray] = []

    for cls, est in models.items():
        try:
            if model_name in {"xgboost", "catboost", "random_forest"}:
                # Unwrap pipeline if present (trees are not piped)
                model = est
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X)
            elif model_name in {"logistic_regression", "elastic_net"}:
                # Pipeline: scaler + clf
                scaler = est.named_steps["scaler"]
                clf = est.named_steps["clf"]
                X_scaled = scaler.transform(X)
                bg_scaled = scaler.transform(bg)
                explainer = shap.LinearExplainer(clf, bg_scaled)
                sv = explainer.shap_values(X_scaled)
            else:
                continue

            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            per_model_means.append(np.abs(sv).mean(axis=0))
        except Exception as exc:
            print(f"SHAP skipped for {cls}: {exc}")
            continue

    if not per_model_means:
        return pd.DataFrame(
            columns=["feature", "mean_abs_shap", "n_models", "n_models_evaluated"]
        )

    mat = np.vstack(per_model_means)
    mean_abs_shap = mat.mean(axis=0)
    n_models = (mat > shap_threshold).sum(axis=0).astype(int)
    out = pd.DataFrame(
        {
            "feature": feats,
            "mean_abs_shap": mean_abs_shap,
            "n_models": n_models,
            "n_models_evaluated": mat.shape[0],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-class AutoML (class-aware feature / model search)
# ---------------------------------------------------------------------------


@dataclass
class ClassBestConfig:
    class_name: str
    model_name: str
    features: list[str]
    params: dict[str, Any]
    cv_pr_auc: float
    cv_prevalence: float
    cv_skill: float
    trial_number: int
    feature_groups_used: list[str] = field(default_factory=list)
    group_subsets: dict[str, str] = field(default_factory=dict)
    predictable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "model_name": self.model_name,
            "features": self.features,
            "params": self.params,
            "cv_pr_auc": self.cv_pr_auc,
            "cv_prevalence": self.cv_prevalence,
            "cv_skill": self.cv_skill,
            "trial_number": self.trial_number,
            "feature_groups_used": self.feature_groups_used,
            "group_subsets": self.group_subsets,
            "predictable": self.predictable,
        }

    def to_best_config(self) -> BestConfig:
        return BestConfig(
            model_name=self.model_name,
            features=self.features,
            params=self.params,
            cv_macro_pr_auc=self.cv_pr_auc,
            trial_number=self.trial_number,
            feature_groups_used=self.feature_groups_used,
            group_subsets=self.group_subsets,
        )


def _class_fold_pr_auc(
    df: pd.DataFrame,
    fold: YearFold,
    features: list[str],
    model_name: str,
    params: dict[str, Any],
    *,
    class_name: str,
    horizon: int,
) -> float | None:
    target = class_target(class_name, horizon)
    if target not in df.columns:
        return None
    feats = [c for c in features if c in df.columns]
    if not feats:
        return None
    tr = df.iloc[fold.train_idx]
    va = df.iloc[fold.val_idx]
    y_tr = tr[target].astype(int)
    y_va = va[target].astype(int)
    if y_tr.nunique() < 2 or y_va.nunique() < 2:
        return None
    try:
        proba, _ = fit_predict_proba(model_name, params, tr[feats], y_tr, va[feats])
        return float(average_precision_score(y_va, proba))
    except Exception:
        return None


def make_per_class_objective(
    dev: pd.DataFrame,
    groups: dict[str, list[str]],
    folds: list[YearFold],
    *,
    class_name: str,
    horizon: int = 1,
):
    import optuna

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", MODEL_NAMES)
        features, used_groups, group_subsets = suggest_feature_config(trial, groups)
        if not features:
            return 0.0
        params = suggest_model_params(trial, model_name)

        # Record config before the fold loop so pruned trials keep metadata.
        trial.set_user_attr("features", features)
        trial.set_user_attr("feature_groups_used", used_groups)
        trial.set_user_attr("group_subsets", group_subsets)
        trial.set_user_attr("model_params", params)
        trial.set_user_attr("n_features", len(features))

        fold_scores: list[float] = []
        for fold in folds:
            score = _class_fold_pr_auc(
                dev,
                fold,
                features,
                model_name,
                params,
                class_name=class_name,
                horizon=horizon,
            )
            if score is None:
                continue
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), fold.fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_scores:
            return 0.0

        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        return mean_score

    return objective


def _cv_prevalence(dev: pd.DataFrame, folds: list[YearFold], class_name: str, horizon: int) -> float:
    target = class_target(class_name, horizon)
    rates = []
    for fold in folds:
        y = dev.iloc[fold.val_idx][target].astype(int)
        rates.append(float(y.mean()))
    return float(np.mean(rates)) if rates else float("nan")


def run_per_class_automl(
    dev: pd.DataFrame,
    groups: dict[str, list[str]],
    folds: list[YearFold],
    classes: list[str],
    *,
    n_trials: int = 100,
    horizon: int = 1,
    seed: int = 42,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, ClassBestConfig]]:
    """Independent Optuna search per class (feature groups + model + hyperparams)."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []
    configs: dict[str, ClassBestConfig] = {}

    for i, cls in enumerate(classes, start=1):
        if show_progress:
            print(f"[{i}/{len(classes)}] Per-class AutoML: {cls}  ({n_trials} trials)")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed + i),
            pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=1),
            study_name=f"per_class_{i}",
        )
        objective = make_per_class_objective(
            dev, groups, folds, class_name=cls, horizon=horizon
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

        if study.best_trial is None or study.best_value is None:
            print(f"  WARNING: no successful trials for {cls}")
            continue

        t = study.best_trial
        prev = _cv_prevalence(dev, folds, cls, horizon)
        pr = float(t.value)
        skill = pr - prev
        cfg = ClassBestConfig(
            class_name=cls,
            model_name=t.params["model"],
            features=list(t.user_attrs["features"]),
            params=dict(t.user_attrs["model_params"]),
            cv_pr_auc=pr,
            cv_prevalence=prev,
            cv_skill=skill,
            trial_number=int(t.number),
            feature_groups_used=list(t.user_attrs["feature_groups_used"]),
            group_subsets=dict(t.user_attrs.get("group_subsets", {})),
        )
        configs[cls] = cfg
        rows.append(
            {
                "class": cls,
                "model": cfg.model_name,
                "cv_pr_auc": cfg.cv_pr_auc,
                "cv_prevalence": cfg.cv_prevalence,
                "cv_skill": cfg.cv_skill,
                "n_features": len(cfg.features),
                "feature_groups": ",".join(cfg.feature_groups_used),
                "group_subsets": json.dumps(cfg.group_subsets, sort_keys=True),
                "trial": cfg.trial_number,
            }
        )
        if show_progress:
            print(
                f"  best={cfg.model_name}  PR-AUC={cfg.cv_pr_auc:.4f}  "
                f"skill={cfg.cv_skill:.4f}  n_feat={len(cfg.features)}"
            )

    results = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    return results, configs


def classify_predictability(
    results_df: pd.DataFrame,
    *,
    min_skill: float = PREDICTABLE_MIN_SKILL,
    min_pr_auc: float = PREDICTABLE_MIN_PR_AUC,
) -> pd.DataFrame:
    """Mark classes predictable vs non-predictable using CV skill rule."""
    out = results_df.copy()
    out["predictable"] = (out["cv_skill"] >= min_skill) & (out["cv_pr_auc"] >= min_pr_auc)
    out["predictability"] = np.where(out["predictable"], "predictable", "non_predictable")
    return out.sort_values(["predictable", "cv_skill"], ascending=[False, False]).reset_index(
        drop=True
    )


def apply_predictability_to_configs(
    configs: dict[str, ClassBestConfig],
    classified: pd.DataFrame,
) -> dict[str, ClassBestConfig]:
    pred_map = dict(zip(classified["class"], classified["predictable"]))
    for cls, cfg in configs.items():
        cfg.predictable = bool(pred_map.get(cls, False))
    return configs


def train_eval_per_class_configs(
    panel: pd.DataFrame,
    configs: dict[str, ClassBestConfig],
    *,
    horizons: list[int] | None = None,
    max_train_year: int = DEV_TARGET_MAX_YEAR,
    test_years: tuple[int, ...] = TEST_TARGET_YEARS,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], dict[int, pd.DataFrame]]:
    """Retrain each class with its own config; evaluate all horizons on target-year test."""
    horizons = horizons or FORECAST_HORIZONS
    all_metrics = []
    models_by_h: dict[int, dict[str, Any]] = {h: {} for h in horizons}
    proba_by_h: dict[int, pd.DataFrame] = {}

    for h in horizons:
        train_df, test_df = split_by_target_year(
            panel,
            horizon=h,
            max_train_year=max_train_year,
            test_years=test_years,
        )
        proba_cols = {}
        for cls, cfg in configs.items():
            target = class_target(cls, h)
            feats = [c for c in cfg.features if c in train_df.columns]
            y_tr = train_df[target].astype(int)
            if y_tr.nunique() < 2 or not feats:
                continue
            est = build_estimator(cfg.model_name, cfg.params, y_tr)
            est.fit(train_df[feats], y_tr)
            models_by_h[h][cls] = est
            y_score = est.predict_proba(test_df[feats])[:, 1]
            proba_cols[cls] = y_score
            y_true = test_df[target].astype(int)
            y_pred = (y_score >= 0.5).astype(int)
            metrics = evaluate_binary(y_true, y_pred, y_score)
            all_metrics.append(
                {
                    "class": cls,
                    "horizon": h,
                    "model": cfg.model_name,
                    "n_features": len(feats),
                    **metrics,
                }
            )
        proba_by_h[h] = pd.DataFrame(proba_cols, index=test_df.index)

    if not all_metrics:
        metrics_df = pd.DataFrame(
            columns=["class", "horizon", "model", "n_features", "pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "prevalence"]
        )
    else:
        metrics_df = pd.DataFrame(all_metrics)
    return metrics_df, models_by_h, proba_by_h


def save_class_configs(configs: dict[str, ClassBestConfig], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {cls: cfg.to_dict() for cls, cfg in configs.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_class_configs(path: Path) -> dict[str, ClassBestConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ClassBestConfig] = {}
    for cls, d in raw.items():
        out[cls] = ClassBestConfig(**d)
    return out
