"""IsolationForest training and tender risk-score conversion."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import prepare_model_input


def _drop_source_constant_missingness(matrix: pd.DataFrame) -> pd.DataFrame:
    """Remove missingness columns that cannot distinguish rows in this source."""
    constant = [
        column
        for column in matrix.columns
        if column.endswith("_missing") and matrix[column].nunique(dropna=False) <= 1
    ]
    return matrix.drop(columns=constant)


def train_and_score(frame: pd.DataFrame) -> tuple[pd.DataFrame, IsolationForest | None, list[str]]:
    """Fit independent per-source models and rank risk within each source."""
    result = frame.copy()

    if result.empty:
        result["raw_score"] = pd.Series(dtype=float)
        result["risk_score"] = pd.Series(dtype="int64")
        return result, None, []

    result["raw_score"] = pd.Series(index=result.index, dtype=float)
    result["risk_score"] = pd.Series(index=result.index, dtype="int64")
    group_keys = (
        result.groupby("source", dropna=False, sort=False).groups.values()
        if "source" in result.columns
        else [result.index]
    )
    models: list[IsolationForest] = []
    feature_names: list[str] = []

    for indices in group_keys:
        group = result.loc[indices]
        matrix = _drop_source_constant_missingness(prepare_model_input(group))
        feature_names.extend(column for column in matrix.columns if column not in feature_names)
        if len(group) == 1:
            result.loc[indices, "raw_score"] = 0.0
            result.loc[indices, "risk_score"] = 50
            continue

        model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        model.fit(matrix)
        models.append(model)
        raw_score = pd.Series(model.decision_function(matrix), index=indices)
        percentile = raw_score.rank(method="average", pct=True)
        result.loc[indices, "raw_score"] = raw_score
        result.loc[indices, "risk_score"] = (100 * (1 - percentile)).round().clip(0, 100).astype(int)

    result["risk_score"] = result["risk_score"].astype(int)
    return result, models[0] if len(models) == 1 else None, feature_names


def score_tenders(frame: pd.DataFrame) -> pd.DataFrame:
    """Convenience public API returning only the scored table."""
    scored, _, _ = train_and_score(frame)
    return scored
