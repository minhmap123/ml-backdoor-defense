from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd


def resolve_feature_groups(
    features: pd.DataFrame,
    categorical_columns: Optional[Sequence[str]],
    numerical_columns: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    if categorical_columns is None:
        inferred_categorical = [
            col
            for col in features.columns
            if pd.api.types.is_object_dtype(features[col]) or pd.api.types.is_categorical_dtype(features[col])
        ]
    else:
        inferred_categorical = [str(col) for col in categorical_columns]

    if numerical_columns is None:
        inferred_numerical = [col for col in features.columns if col not in set(inferred_categorical)]
    else:
        inferred_numerical = [str(col) for col in numerical_columns]

    missing = [col for col in inferred_categorical + inferred_numerical if col not in features.columns]
    if missing:
        raise ValueError(f"Columns not found in features: {missing}")

    return inferred_categorical, inferred_numerical
