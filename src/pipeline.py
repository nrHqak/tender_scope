"""CLI entry point for the end-to-end Tender Scope pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .explain import add_explanations
from .features import engineer_features
from .loaders import load_api, load_scraped, load_synthetic
from .model import train_and_score


DEFAULT_SYNTHETIC_PATH = Path("data/raw/synthetic_goszakup.csv")
DEFAULT_SCRAPED_PATH = Path("data/raw/goszakup_lots.csv")
DEFAULT_API_PATH = Path("data/raw/api_export.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/scored_tenders.parquet")


def ensure_synthetic_data(path: Path) -> None:
    """Create the demo data only when the user has not supplied it already."""
    if path.exists():
        return
    from generate_synthetic_data import generate_dataset, save_csv

    path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(generate_dataset(), str(path))


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    def optional_source(
        name: str, path: str, loader: Callable[[str], pd.DataFrame]
    ) -> pd.DataFrame:
        message = f"файл {path} пустой, повреждён или не найден, источник {name} пропущен"
        try:
            loaded = loader(path)
        except (ValueError, OSError, pd.errors.ParserError) as error:
            if args.source == name:
                raise ValueError(message) from error
            print(f"Предупреждение: {message}", file=sys.stderr)
            return pd.DataFrame()
        if loaded.empty:
            if args.source == name:
                raise ValueError(message)
            print(f"Предупреждение: {message}", file=sys.stderr)
        return loaded

    sources: list[pd.DataFrame] = []
    if args.source in {"synthetic", "all"}:
        synthetic_path = Path(args.synthetic_path)
        ensure_synthetic_data(synthetic_path)
        sources.append(load_synthetic(synthetic_path))
    if args.source in {"scraped", "all"}:
        sources.append(optional_source("scraped", args.scraped_path, load_scraped))
    if args.source in {"api", "all"}:
        sources.append(optional_source("api", args.api_path, load_api))

    available = [source for source in sources if not source.empty]
    if not available:
        raise ValueError("Нет доступных входных записей для выбранного источника.")
    return pd.concat(available, ignore_index=True, sort=False)


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    dataset = build_dataset(args)
    if "source" in dataset.columns:
        featured = pd.concat(
            [engineer_features(group) for _, group in dataset.groupby("source", dropna=False, sort=False)]
        ).sort_index()
    else:
        featured = engineer_features(dataset)
    scored, _, _ = train_and_score(featured)
    explained = add_explanations(scored)
    explained = explained.drop(columns="is_captured_ground_truth", errors="ignore")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explained.to_parquet(output_path, index=False)
    return explained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Рассчитать риск-скор конкурентности тендеров")
    parser.add_argument("--source", choices=("synthetic", "scraped", "api", "all"), default="synthetic")
    parser.add_argument("--synthetic-path", default=str(DEFAULT_SYNTHETIC_PATH))
    parser.add_argument("--scraped-path", default=str(DEFAULT_SCRAPED_PATH))
    parser.add_argument("--api-path", default=str(DEFAULT_API_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scored = run_pipeline(args)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(f"Готово: {len(scored)} тендеров сохранено в {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
