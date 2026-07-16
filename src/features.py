"""Feature engineering for transparent tender-risk scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import require_columns


MODEL_FEATURES = (
    "single_bidder_flag",
    "price_zscore",
    "window_ratio",
    "customer_category_hhi",
    "competitive_method_flag",
)
MAX_ABS_PRICE_ZSCORE = 8.0
NULLABLE_MODEL_FEATURES = (
    "single_bidder_flag",
    "price_zscore",
    "window_ratio",
    "customer_category_hhi",
    "competitive_method_flag",
)


def _group_median_and_mad(
    frame: pd.DataFrame, value: str, group_keys: list[str], minimum_count: int
) -> tuple[pd.Series, pd.Series]:
    grouped = frame.groupby(group_keys, dropna=False)[value]
    count = grouped.transform("count")
    median = grouped.transform("median")
    mad = grouped.transform(lambda values: (values - values.median()).abs().median())
    valid = count >= minimum_count
    return median.where(valid), mad.where(valid)


def _robust_zscore(values: pd.Series, medians: pd.Series, mads: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna() & medians.notna() & mads.notna()
    denominator = 1.4826 * mads
    regular = valid & denominator.gt(0)
    result.loc[regular] = ((values - medians) / denominator).loc[regular]

    # A constant-price group has MAD=0; preserve a useful, finite anomaly signal.
    zero_mad = valid & denominator.eq(0)
    difference = values - medians
    result.loc[zero_mad & difference.eq(0)] = 0.0
    result.loc[zero_mad & difference.ne(0)] = np.sign(difference.loc[zero_mad & difference.ne(0)]) * 3.0
    return result


def _group_percentile(
    frame: pd.DataFrame, value: str, group_keys: list[str], minimum_count: int
) -> pd.Series:
    """Empirical percentile within a sufficiently populated comparison group."""
    grouped = frame.groupby(group_keys, dropna=False)[value]
    count = grouped.transform("count")
    percentile = grouped.rank(method="average", pct=True)
    return percentile.where(count >= minimum_count)


def _compute_hhi(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = (
        frame["customer"].notna()
        & frame["category"].notna()
        & frame["winner_supplier"].notna()
        & frame["amount_tg"].notna()
        & frame["amount_tg"].ge(0)
    )
    if not valid.any():
        return result

    history = frame.loc[valid, ["customer", "category", "winner_supplier", "amount_tg"]].copy()
    supplier_amounts = (
        history.groupby(["customer", "category", "winner_supplier"], dropna=False)["amount_tg"]
        .sum()
        .rename("supplier_amount")
        .reset_index()
    )
    supplier_amounts["total_amount"] = supplier_amounts.groupby(["customer", "category"])["supplier_amount"].transform("sum")
    supplier_amounts["share"] = np.where(
        supplier_amounts["total_amount"].gt(0),
        supplier_amounts["supplier_amount"] / supplier_amounts["total_amount"],
        np.nan,
    )
    hhi = (supplier_amounts.assign(component=supplier_amounts["share"] ** 2)
           .groupby(["customer", "category"], dropna=False)["component"].sum() * 10_000)
    lookup = hhi.to_dict()
    result.loc[valid] = [lookup.get((customer, category), np.nan) for customer, category in zip(
        frame.loc[valid, "customer"], frame.loc[valid, "category"]
    )]
    return result


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add model features and reference values while retaining every tender row."""
    require_columns(frame, ("customer", "category", "amount_tg", "window_days", "n_bidders", "trade_method"))
    result = frame.copy()
    result["amount_tg"] = pd.to_numeric(result["amount_tg"], errors="coerce")
    result["window_days"] = pd.to_numeric(result["window_days"], errors="coerce")
    result["n_bidders"] = pd.to_numeric(result["n_bidders"], errors="coerce")

    # Keep an amount-scale median for display, but estimate anomalous prices on
    # log1p(amount). Procurement values span several orders of magnitude, and a
    # raw-tenge z-score otherwise lets one capital project dominate small goods.
    pair_median, _ = _group_median_and_mad(result, "amount_tg", ["customer", "category"], 5)
    category_median, _ = _group_median_and_mad(result, "amount_tg", ["category"], 5)
    known_amounts = result["amount_tg"].dropna()
    global_median = known_amounts.median() if not known_amounts.empty else np.nan
    price_median = pair_median.fillna(category_median).fillna(global_median)
    result["price_reference_median"] = price_median
    result["price_reference_level"] = np.select(
        [pair_median.notna(), category_median.notna()],
        ["customer_category", "category"],
        default="global",
    )

    log_amount = np.log1p(result["amount_tg"].clip(lower=0)).where(result["amount_tg"].notna())
    result["_log_amount_tg"] = log_amount
    pair_log_median, pair_log_mad = _group_median_and_mad(result, "_log_amount_tg", ["customer", "category"], 5)
    category_log_median, category_log_mad = _group_median_and_mad(result, "_log_amount_tg", ["category"], 5)
    known_log_amounts = log_amount.dropna()
    global_log_median = known_log_amounts.median() if not known_log_amounts.empty else np.nan
    global_log_mad = (
        (known_log_amounts - global_log_median).abs().median()
        if not known_log_amounts.empty
        else np.nan
    )
    log_median = pair_log_median.fillna(category_log_median).fillna(global_log_median)
    log_mad = pair_log_mad.fillna(category_log_mad).fillna(global_log_mad)
    result["price_zscore_raw"] = _robust_zscore(log_amount, log_median, log_mad)
    # The product risk is price inflation, not a low winning price.  Keep the
    # raw symmetric diagnostic for audit, but feed only bounded upward outliers
    # into the risk model and explanations.
    result["price_zscore"] = result["price_zscore_raw"].clip(lower=0, upper=MAX_ABS_PRICE_ZSCORE)

    pair_percentile = _group_percentile(result, "amount_tg", ["customer", "category"], 5)
    category_percentile = _group_percentile(result, "amount_tg", ["category"], 5)
    global_percentile = pd.Series(np.nan, index=result.index, dtype=float)
    global_percentile.loc[known_amounts.index] = known_amounts.rank(method="average", pct=True)
    result["price_percentile"] = pair_percentile.fillna(category_percentile).fillna(global_percentile)
    # A tiny MAD can make a mundane difference look huge.  Price inflation is
    # a risk signal only in the upper tail of its actual peer distribution.
    result.loc[result["price_percentile"].lt(0.90), "price_zscore"] = 0.0
    result.drop(columns="_log_amount_tg", inplace=True)

    category_window_median, _ = _group_median_and_mad(result, "window_days", ["category"], 5)
    known_windows = result["window_days"].dropna()
    global_window_median = known_windows.median() if not known_windows.empty else np.nan
    window_reference = category_window_median.fillna(global_window_median)
    result["window_reference_median"] = window_reference
    result["window_ratio"] = result["window_days"] / window_reference

    result["single_bidder_flag"] = np.where(result["n_bidders"].notna(), (result["n_bidders"] <= 1).astype(float), np.nan)
    methods = result["trade_method"].astype("string").str.casefold().str.strip()
    competitive = (
        methods.str.contains("открытый конкурс", na=False)
        | methods.str.contains("электронный аукцион", na=False)
        | methods.str.contains("запрос ценовых предложений", na=False)
        | methods.str.contains("тендер", na=False)
        | methods.str.contains("конкурс", na=False)
    )
    noncompetitive = methods.str.contains("из одного источника", na=False)
    result["competitive_method_flag"] = pd.Series(
        np.select([competitive, noncompetitive], [1.0, 0.0], default=np.nan),
        index=result.index,
        dtype=float,
    )
    result["customer_category_hhi"] = _compute_hhi(result)

    for feature in NULLABLE_MODEL_FEATURES:
        result[f"{feature}_missing"] = result[feature].isna().astype(bool)
    return result


def prepare_model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-impute model features and append explicit missingness signals."""
    require_columns(frame, MODEL_FEATURES)
    matrix = pd.DataFrame(index=frame.index)
    for feature in MODEL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        matrix[feature] = values.fillna(values.median() if values.notna().any() else 0.0)
        if feature in NULLABLE_MODEL_FEATURES:
            matrix[f"{feature}_missing"] = values.isna().astype(int)
    return matrix.astype(float)

