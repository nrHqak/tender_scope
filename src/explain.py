"""Human-readable, deterministic peer-group feature-deviation descriptions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import MAX_ABS_PRICE_ZSCORE


def _factor(
    feature: str,
    value: Any,
    deviation: float,
    text: str,
    raw_deviation: float = 0.0,
) -> dict[str, Any]:
    return {
        "feature": feature,
        # A list-of-struct Parquet column requires one stable Arrow type.  The
        # precise numeric signal remains in raw_deviation; value is display data.
        "value": str(value),
        "deviation": round(float(deviation), 3),
        "raw_deviation": round(float(raw_deviation), 3),
        "text": text,
    }


def factors_for_record(row: pd.Series) -> list[dict[str, Any]]:
    """Return a descriptive feature-deviation breakdown, not model attribution."""
    factors: list[dict[str, Any]] = []
    n_bidders = row.get("n_bidders")
    if pd.notna(n_bidders) and float(n_bidders) <= 1:
        factors.append(_factor("single_bidder_flag", float(n_bidders), 1.0, "Единственный участник (1 заявка).", 1.0))

    zscore = row.get("price_zscore")
    amount = row.get("amount_tg")
    price_percentile = row.get("price_percentile")
    if pd.notna(zscore) and float(zscore) > 0 and pd.notna(amount) and pd.notna(price_percentile):
        percentile = float(price_percentile) * 100
        reference_level = {
            "customer_category": "заказчика и категории",
            "category": "категории",
            "global": "всех доступных лотов",
        }.get(str(row.get("price_reference_level", "global")), "сопоставимой группы")
        factors.append(_factor(
            "price_zscore",
            float(amount),
            0.9 * min(abs(float(zscore)) / MAX_ABS_PRICE_ZSCORE, 1.0),
            f"Цена находится в {percentile:.1f}-м перцентиле группы {reference_level}.",
            abs(float(zscore)),
        ))

    ratio = row.get("window_ratio")
    window = row.get("window_days")
    window_reference = row.get("window_reference_median")
    if pd.notna(ratio) and pd.notna(window) and pd.notna(window_reference) and 0 < float(ratio) < 1:
        factors.append(_factor(
            "window_ratio",
            float(window),
            0.8 * min(abs(float(np.log(float(ratio)))) / np.log(12), 1.0),
            f"Окно подачи {float(window):.0f} дн., короче медианы категории ({float(window_reference):.0f} дн.).",
            abs(float(np.log(float(ratio)))),
        ))

    hhi = row.get("customer_category_hhi")
    if pd.notna(hhi):
        hhi_value = float(hhi)
        factors.append(_factor(
            "customer_category_hhi",
            hhi_value,
            0.9 * min(hhi_value / 10_000, 1.0),
            f"Концентрация побед у заказчика в категории: HHI {hhi_value:.0f} из 10000.",
            hhi_value,
        ))

    method_flag = row.get("competitive_method_flag")
    method_missing = row.get("competitive_method_flag_missing", pd.isna(method_flag))
    if not bool(method_missing) and pd.notna(method_flag) and float(method_flag) == 0:
        factors.append(_factor("competitive_method_flag", 0, 0.95, "Неконкурентный способ закупки.", 1.0))

    if not factors:
        # Guarantees a useful drill-down even for sparse public-list rows.
        factors.append(_factor(
            "data_availability",
            "limited",
            0.0,
            "Недостаточно детальных полей; риск рассчитан по доступным данным.",
            0.0,
        ))
    factors.sort(key=lambda item: item["deviation"], reverse=True)
    return factors[:3]


def add_explanations(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach nonempty descriptive peer-group feature deviations to every row."""
    result = frame.copy()
    result["top_factors"] = [factors_for_record(row) for _, row in result.iterrows()]
    return result


def top_factor_text(factors: Any) -> str:
    """Render stored factor dictionaries for a Streamlit drill-down."""
    if isinstance(factors, np.ndarray):
        factors = factors.tolist()
    if not isinstance(factors, list):
        return "Недостаточно данных для разбора отклонений."
    return "; ".join(str(factor.get("text", "")) for factor in factors if isinstance(factor, dict))
