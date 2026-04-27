from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureImportanceResult:
    features: List[str]
    scores: List[float]
    ranked_importances: pd.Series
    per_model_rankings: Dict[str, pd.Series]


def _validate_inputs(
    data: pd.DataFrame,
    label_column: str,
    numerical_columns: Sequence[str],
    categorical_columns: Optional[Sequence[str]],
) -> tuple[list[str], list[str]]:
    if label_column not in data.columns:
        raise ValueError(f"label_column '{label_column}' not found in data.")

    num_cols = list(numerical_columns)
    cat_cols = list(categorical_columns or [])

    missing_num = [col for col in num_cols if col not in data.columns]
    missing_cat = [col for col in cat_cols if col not in data.columns]
    if missing_num or missing_cat:
        missing = missing_num + missing_cat
        raise ValueError(f"Columns not found in data: {missing}")

    return num_cols, cat_cols


def _build_feature_frame(
    data: pd.DataFrame,
    label_column: str,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = list(numerical_columns) + list(categorical_columns)
    x = data[feature_columns].copy()
    y = data[label_column].copy()
    return x, y


def _standardize_numeric(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    numerical_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not numerical_columns:
        return x_train, x_valid, x_test

    scaler = StandardScaler()
    scaler.fit(x_train[list(numerical_columns)])

    x_train = x_train.copy()
    x_valid = x_valid.copy()
    x_test = x_test.copy()

    x_train.loc[:, list(numerical_columns)] = scaler.transform(x_train[list(numerical_columns)])
    x_valid.loc[:, list(numerical_columns)] = scaler.transform(x_valid[list(numerical_columns)])
    x_test.loc[:, list(numerical_columns)] = scaler.transform(x_test[list(numerical_columns)])
    return x_train, x_valid, x_test


def _minmax_normalize_importance(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value - min_value < 1e-12:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - min_value) / (max_value - min_value)


def _mean_series(series_list: Sequence[pd.Series]) -> pd.Series:
    if not series_list:
        raise ValueError("series_list must not be empty.")
    return pd.concat(series_list, axis=1).mean(axis=1)


def _fit_tabnet_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    device_name: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    virtual_batch_size: int,
    cat_idxs: Optional[Sequence[int]],
    cat_dims: Optional[Sequence[int]],
    cat_emb_dim: int,
) -> pd.Series:
    from pytorch_tabnet.tab_model import TabNetClassifier

    model = TabNetClassifier(
        device_name=device_name,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        momentum=0.3,
        mask_type="entmax",
        cat_idxs=list(cat_idxs or []),
        cat_dims=list(cat_dims or []),
        cat_emb_dim=int(cat_emb_dim),
        seed=int(random_state),
    )
    model.fit(
        X_train=x_train.values,
        y_train=y_train.values,
        eval_set=[(x_train.values, y_train.values), (x_valid.values, y_valid.values)],
        eval_name=["train", "valid"],
        max_epochs=int(max_epochs),
        patience=int(patience),
        batch_size=int(batch_size),
        virtual_batch_size=int(virtual_batch_size),
    )
    return pd.Series(model.feature_importances_, index=x_train.columns, dtype=float)


def _fit_xgboost_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    n_classes: int,
) -> pd.Series:
    from xgboost import XGBClassifier

    objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes <= 2 else "mlogloss"
    model = XGBClassifier(
        n_estimators=100,
        random_state=int(random_state),
        objective=objective,
        eval_metric=eval_metric,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return pd.Series(model.feature_importances_, index=x_train.columns, dtype=float)


def _fit_lightgbm_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    n_classes: int,
) -> pd.Series:
    from lightgbm import LGBMClassifier

    objective = "binary" if n_classes <= 2 else "multiclass"
    model = LGBMClassifier(
        n_estimators=100,
        random_state=int(random_state),
        verbose=-1,
        objective=objective,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
    return pd.Series(model.feature_importances_, index=x_train.columns, dtype=float)


def _fit_catboost_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    n_classes: int,
) -> pd.Series:
    from catboost import CatBoostClassifier

    loss_function = "Logloss" if n_classes <= 2 else "MultiClass"
    model = CatBoostClassifier(
        verbose=0,
        n_estimators=100,
        random_state=int(random_state),
        loss_function=loss_function,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
    return pd.Series(model.feature_importances_, index=x_train.columns, dtype=float)


def _fit_random_forest_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
) -> pd.Series:
    model = RandomForestClassifier(
        n_estimators=100,
        verbose=0,
        n_jobs=-1,
        random_state=int(random_state),
    )
    model.fit(x_train, y_train)
    return pd.Series(model.feature_importances_, index=x_train.columns, dtype=float)


def compute_ensemble_feature_importance(
    data: pd.DataFrame,
    *,
    label_column: str,
    numerical_columns: Sequence[str],
    categorical_columns: Optional[Sequence[str]] = None,
    reruns: int = 5,
    test_size: float = 0.2,
    valid_size_within_train: float = 0.2,
    device_name: str = "cpu",
    tabnet_max_epochs: int = 65,
    tabnet_patience: int = 65,
    tabnet_batch_size: int = 1024,
    tabnet_virtual_batch_size: int = 128,
    cat_idxs: Optional[Sequence[int]] = None,
    cat_dims: Optional[Sequence[int]] = None,
    cat_emb_dim: int = 1,
) -> FeatureImportanceResult:
    """
    Reproduces the TabDoor feature-importance pipeline:
    - 5 surrogate models: TabNet, XGBoost, LightGBM, CatBoost, Random Forest
    - repeat across multiple random splits
    - standardize numeric columns using train split statistics
    - min-max normalize importances per run, average per model, then average across models
    - final ranking keeps only numerical features, as done in the author notebooks
    """

    if reruns <= 0:
        raise ValueError("reruns must be > 0")

    num_cols, cat_cols = _validate_inputs(data, label_column, numerical_columns, categorical_columns)
    x, y = _build_feature_frame(data, label_column, num_cols, cat_cols)
    n_classes = int(pd.Series(y).nunique())

    importances_by_model: Dict[str, list[pd.Series]] = {
        "tabnet": [],
        "xgboost": [],
        "lightgbm": [],
        "catboost": [],
        "random_forest": [],
    }

    for random_state in range(int(reruns)):
        x_train_valid, x_test, y_train_valid, y_test = train_test_split(
            x,
            y,
            stratify=y,
            test_size=float(test_size),
            random_state=random_state,
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_valid,
            y_train_valid,
            stratify=y_train_valid,
            test_size=float(valid_size_within_train),
            random_state=random_state,
        )

        x_train, x_valid, x_test = _standardize_numeric(x_train, x_valid, x_test, num_cols)

        importances_by_model["tabnet"].append(
            _fit_tabnet_importance(
                x_train,
                y_train,
                x_valid,
                y_valid,
                random_state=random_state,
                device_name=device_name,
                max_epochs=tabnet_max_epochs,
                patience=tabnet_patience,
                batch_size=tabnet_batch_size,
                virtual_batch_size=tabnet_virtual_batch_size,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=cat_emb_dim,
            )
        )
        importances_by_model["xgboost"].append(
            _fit_xgboost_importance(
                x_train,
                y_train,
                x_valid,
                y_valid,
                random_state=random_state,
                n_classes=n_classes,
            )
        )
        importances_by_model["lightgbm"].append(
            _fit_lightgbm_importance(
                x_train,
                y_train,
                x_valid,
                y_valid,
                random_state=random_state,
                n_classes=n_classes,
            )
        )
        importances_by_model["catboost"].append(
            _fit_catboost_importance(
                x_train,
                y_train,
                x_valid,
                y_valid,
                random_state=random_state,
                n_classes=n_classes,
            )
        )
        importances_by_model["random_forest"].append(
            _fit_random_forest_importance(
                x_train,
                y_train,
                random_state=random_state,
            )
        )

    averaged_by_model: Dict[str, pd.Series] = {}
    normalized_model_averages: list[pd.Series] = []
    for model_name, importance_runs in importances_by_model.items():
        normalized_runs = [_minmax_normalize_importance(series) for series in importance_runs]
        averaged = _mean_series(normalized_runs)
        averaged_by_model[model_name] = averaged[averaged.index.isin(num_cols)].sort_values(ascending=False)
        normalized_model_averages.append(averaged)

    average_importances = _mean_series(normalized_model_averages)
    ranked_importances = average_importances[average_importances.index.isin(num_cols)].sort_values(ascending=False)

    return FeatureImportanceResult(
        features=ranked_importances.index.tolist(),
        scores=[float(v) for v in ranked_importances.values.tolist()],
        ranked_importances=ranked_importances,
        per_model_rankings=averaged_by_model,
    )


def format_feature_importance_result(result: FeatureImportanceResult) -> Dict[str, list[Any]]:
    return {
        "features": list(result.features),
        "scores": [float(score) for score in result.scores],
    }
