from __future__ import annotations

from pathlib import Path

import pandas as pd

from generate_synthetic_data import generate_dataset, save_csv
from scrape_goszakup_lots import parse_page
from src.loaders import classify_category, load_scraped, load_synthetic
from src.schema import CANONICAL_COLUMNS, REQUIRED_COLUMNS


def test_synthetic_loader_schema(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    save_csv(generate_dataset(n_tenders=30), str(path))
    loaded = load_synthetic(path)

    assert set(CANONICAL_COLUMNS).issubset(loaded.columns)
    for column in REQUIRED_COLUMNS:
        assert loaded[column].notna().any(), column
    assert loaded["source"].eq("synthetic").all()


def test_scraped_loader_handles_missing_without_crashing(tmp_path: Path) -> None:
    path = tmp_path / "lots.csv"
    pd.DataFrame([
        {
            "lot_number": "1", "announce_id": "10", "announce_name": "Поставка бумаги",
            "customer": "Заказчик", "lot_id": "100", "lot_name": "Бумага", "quantity": "1",
            "amount_tg": "100 000.00", "trade_method": "Открытый конкурс", "status": "Опубликован",
        },
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    loaded = load_scraped(path)

    assert {"n_bidders", "category", "publish_date"}.issubset(loaded.columns)
    assert pd.isna(loaded.loc[0, "n_bidders"])
    assert loaded.loc[0, "n_bidders_missing"]
    assert loaded.loc[0, "publish_date_missing"]
    assert loaded.loc[0, "category"] == "Канцелярские товары"


def test_scraped_loader_single_source_bidder_inference(tmp_path: Path) -> None:
    path = tmp_path / "lots.csv"
    pd.DataFrame([
        {
            "lot_number": "1", "announce_id": "10", "customer": "Заказчик", "lot_id": "100",
            "lot_name": "Капуста", "amount_tg": "100000", "trade_method": "Из одного источника",
            "status": "Опубликован",
        },
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    loaded = load_scraped(path)

    assert loaded.loc[0, "n_bidders"] == 1
    assert not loaded.loc[0, "n_bidders_missing"]


def test_scraped_loader_deduplicates_known_lot_ids(tmp_path: Path) -> None:
    path = tmp_path / "lots.csv"
    row = {
        "lot_number": "1", "announce_id": "10", "customer": "Заказчик", "lot_id": "100",
        "lot_name": "Капуста", "amount_tg": "100000", "trade_method": "Открытый конкурс",
        "status": "Опубликован",
    }
    pd.DataFrame([row, row]).to_csv(path, index=False, encoding="utf-8-sig")

    loaded = load_scraped(path)

    assert len(loaded) == 1
    assert loaded.loc[0, "tender_id"] == "100"


def test_scraped_category_dictionary_covers_new_real_world_groups() -> None:
    assert classify_category("Работы по среднему ремонту улиц", "") == "Дорожные работы"
    assert classify_category("Услуги по поверке средств измерений", "") == "Метрологические услуги"
    assert classify_category("Услуги по обучению персонала", "") == "Обучение и мероприятия"
    assert classify_category("Кабель специализированный", "") == "Электротехническое оборудование"
    assert classify_category("Работы по текущему ремонту канализационных систем", "") == "Сантехника и отопление"


def test_parse_page_against_real_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_lots.html"
    rows = parse_page(fixture.read_text(encoding="utf-8"))

    assert len(rows) == 3
    assert rows[0]["lot_number"] == "87321454-ОИ2"
    assert float(rows[0]["amount_tg"]) == 100_000.00
    assert rows[0]["status"] == "Опубликован"
