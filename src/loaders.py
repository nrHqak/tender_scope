"""Data-source adapters which all return the canonical tender schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, empty_frame, ensure_schema


CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Specific multi-word work types come first so they are not swallowed by
    # generic construction or services categories later in this list.
    ("Дорожные работы", ("ремонт дорог", "ремонту дорог", "ремонту улиц", "дорожн", "асфальт", "проезж")),
    ("Проектные работы", ("проектир", "проектн", "проектно-смет", "сметн", "техническ документац")),
    ("Аварийно-восстановительные работы", ("ликвидации последств", "аварийн")),
    ("Сельхозтехника и запчасти", ("плуг", "сельхоз", "трактор", "посевн")),
    ("Метрологические услуги", ("поверк", "калибровк", "средств измерен")),
    ("Медицинское оборудование", ("медицин", "лекарств", "медоборуд", "фарма", "диагност", "кресло-коляск")),
    ("Лабораторные и исследовательские услуги", ("лаборатор", "инструментальн исследован", "анализов", "анализы")),
    ("Обучение и мероприятия", ("обучени", "семинар", "конференц", "форум", "корпоративн", "культурн", "праздничн")),
    ("Полиграфия и печатная продукция", ("книг", "полиграф", "печат", "издание", "журнал", "бланк", "стенд", "табличк", "трафарет", "флаг")),
    ("Техническое обслуживание", ("техническ обслужив", "промывк", "опрессовк", "климатическ", "кондиционер", "вентиляц", "котельн", "газов", "противопожар")),
    ("Электротехническое оборудование", ("кабель", "провод", "выключател", "светильник", "ламп", "розетк", "прожектор", "электрооборуд")),
    ("Сантехника и отопление", ("смесител", "кран", "сифон", "резьб", "труб", "отвод", "шланг", "водонагрев", "насос", "котел отоп", "радиатор", "вентил", "канализац")),
    ("Автозапчасти и ГСМ", ("шин", "аккумулятор", "масло мотор", "масло трансмис", "автозапчаст", "фильтр", "смазоч", "гсм", "бензин", "дизел", "топлив")),
    ("Топливо и энергетика", ("уголь", "электроэнерг", "теплоснаб", "водоснаб")),
    ("Средства защиты и спецодежда", ("перчатк", "костюм", "огнетушител", "средств защиты")),
    ("Бытовая химия и хозяйственные товары", ("мыло", "чистящ", "дезинфиц", "ветош", "тряпк", "салфет", "щетк", "порошок", "моющ")),
    ("Спорт и досуг", ("спортивн", "хокке", "футбол", "баскетбол", "игрушк", "игр")),
    ("Связь и электроника", ("ноутбук", "телевизор", "флеш", "телефон", "наушник", "компьют", "сервер", "программ", "информац", "интернет")),
    ("Почтовые и страховые услуги", ("почтов", "пересылк", "страхован")),
    ("Вывоз отходов", ("вывоз", "неопасных отход", "отход")),
    ("Металлоконструкции и фурнитура", ("муфт", "замок", "блок двер", "металлическ конструкц", "лист металлическ", "фурнитур", "зажим")),
    ("Текстиль и мягкий инвентарь", ("полотенц", "жалюзи", "занавес", "нить", "одежд")),
    ("Тара и упаковка", ("мешок", "пакет", "ведро", "контейнер")),
    ("Продукты питания", ("продукт", "питани", "молок", "мяс", "хлеб", "овощ", "капуст")),
    ("Стройматериалы", ("ремонт", "строитель", "стройматериал", "цемент", "кирпич", "краск", "валик", "кисть", "эмаль", "колер", "саморез", "растворител", "шпатлевк", "лак", "кле")),
    ("Канцелярские товары", ("канцел", "бумаг", "картридж", "хоз товар", "хозтовар")),
    ("Транспортные услуги", ("транспорт", "перевоз", "автомоб", "горюч")),
    ("Охранные услуги", ("охран", "сигнализац", "видеонаблюден")),
    ("Клининг", ("клининг", "уборк", "чистке одежды", "ковров", "прачеч")),
    ("Мебель", ("мебел", "кресл", "шкаф", "стол")),
    ("Коммунальные услуги", ("коммунал",)),
)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def _existing_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def classify_category(lot_name: Any, announce_name: Any) -> str:
    """Return a deliberately coarse, deterministic category for a scraped lot."""
    # The lot is the granular procurement object.  An announcement may include
    # many heterogeneous lots, so use it only when the lot name has no signal.
    for text in (str(lot_name or "").casefold(), str(announce_name or "").casefold()):
        for category, keywords in CATEGORY_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return category
    return "Прочее"


def load_synthetic(path: str | Path | None) -> pd.DataFrame:
    """Load generator output and preserve its hidden validation label."""
    source_path = _existing_path(path)
    if source_path is None:
        return empty_frame()

    raw = _read_csv(source_path)
    mapped = pd.DataFrame()
    for column in CANONICAL_COLUMNS:
        if column in raw:
            mapped[column] = raw[column]
    return ensure_schema(mapped, source="synthetic")


def _first_nonempty(raw: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    result = pd.Series(pd.NA, index=raw.index, dtype="string")
    for candidate in candidates:
        if candidate in raw:
            values = raw[candidate].astype("string").replace("", pd.NA)
            result = result.fillna(values)
    return result


def load_scraped(path: str | Path | None) -> pd.DataFrame:
    """Load public lot-list CSV without inventing unavailable bidder/date facts."""
    source_path = _existing_path(path)
    if source_path is None:
        return empty_frame()

    raw = _read_csv(source_path)
    if "lot_id" in raw:
        lot_ids = raw["lot_id"].astype("string").str.strip()
        known_lot_id = lot_ids.notna() & lot_ids.ne("")
        # Keep rows without a lot ID: collapsing them would merge unrelated lots.
        raw = pd.concat(
            [raw.loc[known_lot_id].drop_duplicates(subset="lot_id", keep="first"), raw.loc[~known_lot_id]],
        ).sort_index().reset_index(drop=True)
    mapped = pd.DataFrame(index=raw.index)
    mapped["tender_id"] = _first_nonempty(raw, ("lot_id", "announce_id", "lot_number"))
    mapped["customer"] = raw.get("customer", pd.Series(pd.NA, index=raw.index))
    mapped["trade_method"] = raw.get("trade_method", pd.Series(pd.NA, index=raw.index))
    mapped["amount_tg"] = raw.get("amount_tg", pd.Series(np.nan, index=raw.index))
    mapped["status"] = raw.get("status", pd.Series(pd.NA, index=raw.index))

    lot_names = raw.get("lot_name", pd.Series("", index=raw.index))
    announce_names = raw.get("announce_name", pd.Series("", index=raw.index))
    mapped["category"] = [classify_category(lot, announce) for lot, announce in zip(lot_names, announce_names)]

    method = mapped["trade_method"].astype("string").str.casefold()
    direct_source = method.str.contains("из одного источника", na=False)
    mapped["n_bidders"] = np.where(direct_source, 1.0, np.nan)
    # These columns are intentionally absent from the public list page.
    mapped["publish_date"] = pd.NaT
    mapped["end_date"] = pd.NaT
    mapped["window_days"] = np.nan
    mapped["customer_bin"] = pd.NA
    mapped["winner_supplier"] = pd.NA
    mapped["winner_bin"] = pd.NA
    return ensure_schema(mapped, source="scraped")


def _read_api_export(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "results", "items", "rows"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        return pd.DataFrame(payload)
    return _read_csv(path)


def _pick(raw: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for alias in aliases:
        if alias in raw:
            return raw[alias]
    return pd.Series(np.nan, index=raw.index)


def load_api(path_or_none: str | Path | None) -> pd.DataFrame:
    """Map a future CSV/JSON API export into the same contract as other sources."""
    source_path = _existing_path(path_or_none)
    if source_path is None:
        return empty_frame()

    raw = _read_api_export(source_path)
    aliases = {
        "tender_id": ("tender_id", "lot_id", "announce_id", "id"),
        "customer": ("customer", "customer_name", "customerName"),
        "customer_bin": ("customer_bin", "customer_bin_iin", "customerBin"),
        "category": ("category", "category_name", "okpd_name", "okpdName"),
        "trade_method": ("trade_method", "trade_method_name", "method", "tradeMethod"),
        "publish_date": ("publish_date", "publishDate", "date_publish"),
        "end_date": ("end_date", "endDate", "date_end"),
        "window_days": ("window_days",),
        "amount_tg": ("amount_tg", "amount", "lot_amount", "amountTg"),
        "n_bidders": ("n_bidders", "bidders_count", "participant_count", "biddersCount"),
        "winner_supplier": ("winner_supplier", "winner_name", "supplier", "winnerSupplier"),
        "winner_bin": ("winner_bin", "winner_bin_iin", "supplier_bin", "winnerBin"),
        "status": ("status", "status_name", "statusName"),
    }
    mapped = pd.DataFrame({column: _pick(raw, names) for column, names in aliases.items()})
    return ensure_schema(mapped, source="api")
