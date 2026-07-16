from __future__ import annotations

import math

import pandas as pd

from generate_synthetic_data import generate_dataset
from src.explain import add_explanations, factors_for_record
from src.features import engineer_features
from src.model import train_and_score
from src.schema import ensure_schema


def _synthetic_frame():
    return ensure_schema(pd.DataFrame(generate_dataset(n_tenders=1500, seed=42)), source="synthetic")


def test_model_runs_end_to_end_on_synthetic() -> None:
    scored, _, _ = train_and_score(engineer_features(_synthetic_frame()))

    assert scored["risk_score"].notna().all()
    assert scored["risk_score"].between(0, 100).all()


def test_model_recall_against_ground_truth() -> None:
    scored, _, _ = train_and_score(engineer_features(_synthetic_frame()))
    top_count = math.ceil(len(scored) * 0.10)
    top = scored.nlargest(top_count, "risk_score")
    captured_total = scored["is_captured_ground_truth"].astype(bool).sum()
    recall = top["is_captured_ground_truth"].astype(bool).sum() / captured_total

    assert recall >= 0.60


def test_explain_output_never_empty() -> None:
    scored, _, _ = train_and_score(engineer_features(_synthetic_frame()))
    explained = add_explanations(scored)
    top_count = math.ceil(len(explained) * 0.20)

    assert explained.nlargest(top_count, "risk_score")["top_factors"].map(lambda value: 1 <= len(value) <= 3).all()


def test_explain_keeps_direct_competition_signals_with_extreme_price() -> None:
    factors = factors_for_record(pd.Series({
        "n_bidders": 1,
        "price_zscore": 8.0,
        "price_percentile": 0.999,
        "price_reference_level": "category",
        "amount_tg": 100_000_000,
        "window_ratio": 0.2,
        "window_days": 2,
        "window_reference_median": 10,
        "customer_category_hhi": 10_000,
        "competitive_method_flag": 0,
    }))
    feature_names = {factor["feature"] for factor in factors}

    assert {"single_bidder_flag", "competitive_method_flag"}.issubset(feature_names)
    assert any("перцентиле" in factor["text"] for factor in factors if factor["feature"] == "price_zscore")


def test_explain_does_not_call_unknown_method_noncompetitive() -> None:
    factors = factors_for_record(pd.Series({
        "competitive_method_flag": float("nan"),
        "competitive_method_flag_missing": True,
        "window_ratio": 2.0,
        "window_days": 20,
        "window_reference_median": 10,
    }))

    assert all(factor["feature"] != "competitive_method_flag" for factor in factors)
    assert all(factor["feature"] != "window_ratio" for factor in factors)


def test_constant_missingness_is_removed_within_source() -> None:
    frame = _synthetic_frame()
    frame["n_bidders"] = float("nan")
    scored, _, feature_names = train_and_score(engineer_features(frame))

    assert scored["risk_score"].notna().all()
    assert "single_bidder_flag_missing" not in feature_names


def test_identical_sources_are_scored_independently() -> None:
    first = engineer_features(_synthetic_frame())
    second = first.copy()
    first["source"] = "first"
    second["source"] = "second"
    combined = pd.concat([first, second], ignore_index=True)

    scored, _, _ = train_and_score(combined)
    first_scores = scored.loc[scored["source"].eq("first"), "risk_score"].reset_index(drop=True)
    second_scores = scored.loc[scored["source"].eq("second"), "risk_score"].reset_index(drop=True)

    assert first_scores.equals(second_scores)
