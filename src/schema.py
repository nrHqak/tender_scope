"""Canonical tender schema shared by every data source."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "tender_id",
    "customer",
    "trade_method",
    "amount_tg",
    "status",
    "source",
)

OPTIONAL_COLUMNS = (
    "customer_bin",
    "category",
    "publish_date",
    "end_date",
    "window_days",
    "n_bidders",
    "winner_supplier",
    "winner_bin",
)

PROVENANCE_COLUMNS = ("is_captured_ground_truth",)
MISSING_FLAG_COLUMNS = tuple(f"{column}_missing" for column in OPTIONAL_COLUMNS)
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + MISSING_FLAG_COLUMNS + PROVENANCE_COLUMNS

DATE_COLUMNS = ("publish_date", "end_date")
NUMERIC_COLUMNS = ("amount_tg", "window_days", "n_bidders")


def empty_frame() -> pd.DataFrame:
    """Return an empty table with every canonical column present."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in CANONICAL_COLUMNS})


def _coerce_amount(values: pd.Series) -> pd.Series:
    """Turn values such as ``100 000,50`` into numeric tenge amounts."""
    text = (
        values.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def ensure_schema(frame: pd.DataFrame | None, source: str | None = None) -> pd.DataFrame:
    """Normalize a source frame without dropping incomplete records.

    Optional values are deliberately retained as missing values.  Their paired
    provenance flag makes the limitation visible to the downstream model and UI.
    """
    result = empty_frame() if frame is None else frame.copy()

    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS + PROVENANCE_COLUMNS:
        if column not in result:
            result[column] = np.nan

    if source is not None:
        result["source"] = source

    result["tender_id"] = result["tender_id"].astype("string")
    result.loc[result["tender_id"].str.strip().eq(""), "tender_id"] = pd.NA
    result["amount_tg"] = _coerce_amount(result["amount_tg"])
    result["window_days"] = pd.to_numeric(result["window_days"], errors="coerce")
    result["n_bidders"] = pd.to_numeric(result["n_bidders"], errors="coerce")
    for column in DATE_COLUMNS:
        result[column] = pd.to_datetime(result[column], errors="coerce")

    # Dates are the source of truth for a duration when they are both available.
    dated = result["publish_date"].notna() & result["end_date"].notna()
    result.loc[dated, "window_days"] = (
        result.loc[dated, "end_date"] - result.loc[dated, "publish_date"]
    ).dt.total_seconds() / 86_400

    for column in OPTIONAL_COLUMNS:
        missing = result[column].isna()
        flag = f"{column}_missing"
        if flag in result:
            existing = result[flag].astype("boolean").fillna(False)
            result[flag] = (existing | missing).astype(bool)
        else:
            result[flag] = missing.astype(bool)

    # Keep canonical fields first but do not discard useful raw/source metadata.
    extra_columns = [column for column in result.columns if column not in CANONICAL_COLUMNS]
    return result.loc[:, list(CANONICAL_COLUMNS) + extra_columns]


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a concise error if an internal pipeline contract is violated."""
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"Missing expected columns: {', '.join(absent)}")
