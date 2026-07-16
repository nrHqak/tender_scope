from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import MAX_ABS_PRICE_ZSCORE, engineer_features, prepare_model_input
from src.schema import ensure_schema


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "tender_id": [str(index) for index in range(len(rows))],
        "customer": "Заказчик",
        "category": "Категория",
        "trade_method": "Открытый конкурс",
        "amount_tg": 100.0,
        "status": "Завершен",
        "window_days": 10,
        "n_bidders": 3,
        "winner_supplier": "Поставщик_A",
    }
    frame = pd.DataFrame(base)
    for index, row in enumerate(rows):
        for column, value in row.items():
            frame.loc[index, column] = value
    return ensure_schema(frame, source="synthetic")


def test_hhi_known_case() -> None:
    frame = _frame([
        {"amount_tg": 700.0, "winner_supplier": "A"},
        {"amount_tg": 300.0, "winner_supplier": "B"},
    ])
    featured = engineer_features(frame)

    assert np.allclose(featured["customer_category_hhi"], 5800, atol=50)


def test_price_zscore_flags_outlier() -> None:
    frame = _frame([{"amount_tg": 100.0} for _ in range(10)] + [{"amount_tg": 300.0}])
    featured = engineer_features(frame)

    assert abs(featured.iloc[-1]["price_zscore"]) > 2
    assert (featured.iloc[:-1]["price_zscore"].abs() < 1).all()
    assert featured["price_zscore"].abs().le(MAX_ABS_PRICE_ZSCORE).all()


def test_missing_features_produce_missing_flags_not_nans_in_model_input() -> None:
    frame = _frame([
        {"amount_tg": 100.0, "n_bidders": np.nan, "window_days": np.nan, "winner_supplier": np.nan},
        {"amount_tg": 120.0, "n_bidders": 2, "window_days": 10, "winner_supplier": "A"},
    ])
    featured = engineer_features(frame)
    matrix = prepare_model_input(featured)

    assert not matrix.isna().any().any()
    assert matrix.loc[0, "single_bidder_flag_missing"] == 1
    assert matrix.loc[0, "window_ratio_missing"] == 1


def test_unknown_trade_method_is_missing_not_noncompetitive() -> None:
    featured = engineer_features(_frame([
        {"trade_method": "Неизвестная процедура"},
        {"trade_method": None},
        {"trade_method": "Из одного источника"},
        {"trade_method": "Запрос ценовых предложений"},
    ]))

    assert featured.loc[0:1, "competitive_method_flag"].isna().all()
    assert featured.loc[0:1, "competitive_method_flag_missing"].all()
    assert featured.loc[2, "competitive_method_flag"] == 0
    assert not featured.loc[2, "competitive_method_flag_missing"]
    assert featured.loc[3, "competitive_method_flag"] == 1
