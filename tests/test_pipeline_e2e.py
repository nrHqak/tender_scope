from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run_pipeline(source: str, output: Path) -> pd.DataFrame:
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--source", source, "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    return pd.read_parquet(output)


def test_full_pipeline_synthetic_source(tmp_path: Path) -> None:
    scored = _run_pipeline("synthetic", tmp_path / "synthetic.parquet")

    assert not scored.empty
    assert {"risk_score", "top_factors", "tender_id", "source"}.issubset(scored.columns)
    assert "is_captured_ground_truth" not in scored.columns


def test_full_pipeline_all_sources_if_available(tmp_path: Path) -> None:
    scored = _run_pipeline("all", tmp_path / "all.parquet")

    assert not scored.empty
    assert scored["source"].eq("synthetic").any()
    assert "is_captured_ground_truth" not in scored.columns


def test_missing_single_source_has_concise_cli_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--source", "scraped", "--scraped-path", str(missing)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert f"Ошибка: файл {missing}" in result.stderr
    assert "источник scraped пропущен" in result.stderr
    assert "Traceback" not in result.stderr


def test_all_skips_empty_optional_sources(tmp_path: Path) -> None:
    empty_scraped = tmp_path / "empty.csv"
    empty_api = tmp_path / "empty.json"
    empty_scraped.write_text("", encoding="utf-8")
    empty_api.write_text("", encoding="utf-8")
    output = tmp_path / "all.parquet"
    result = subprocess.run(
        [
            sys.executable, "-m", "src.pipeline", "--source", "all",
            "--scraped-path", str(empty_scraped), "--api-path", str(empty_api),
            "--output", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "Предупреждение:" in result.stderr
    assert "Traceback" not in result.stderr
