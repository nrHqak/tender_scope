"""
Скрапер публичной страницы "Реестр лотов" на goszakup.gov.kz
Не требует токена — берёт те же данные, что видит любой посетитель сайта.

Использование:
    python scrape_goszakup_lots.py --pages 100 --out lots.csv

По умолчанию идёт по count_record=50&page=N начиная с 1.
Будьте вежливы к серверу: задержка между запросами и повторные попытки при сбоях уже встроены.
"""

import argparse
import csv
import random
import re
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://goszakup.gov.kz/ru/search/lots"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
FIELDNAMES = [
    "lot_number", "announce_id", "announce_name", "customer",
    "lot_id", "lot_name", "quantity", "amount_tg", "trade_method", "status",
]


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="search-result")
    if not table:
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []

    rows = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 7:
            continue

        lot_number = cells[0].get_text(strip=True)

        announce_cell = cells[1]
        announce_link = announce_cell.find("a")
        announce_name = announce_link.get_text(strip=True) if announce_link else ""
        announce_href = announce_link["href"] if announce_link else ""
        m = re.search(r"/index/(\d+)", announce_href)
        announce_id = m.group(1) if m else ""
        customer_tag = announce_cell.find("small")
        customer = ""
        if customer_tag:
            customer = customer_tag.get_text(" ", strip=True).replace("Заказчик:", "").strip()

        lot_cell = cells[2]
        lot_link = lot_cell.find("a")
        lot_name = lot_link.get_text(strip=True) if lot_link else ""
        history_btn = lot_cell.find("a", class_="btn-select-history")
        lot_id = history_btn.get("data-lot-id", "") if history_btn else ""

        quantity = cells[3].get_text(strip=True)
        amount = cells[4].get_text(strip=True).replace("\xa0", "").replace(" ", "")
        method = cells[5].get_text(strip=True)
        status = cells[6].get_text(strip=True)

        rows.append({
            "lot_number": lot_number,
            "announce_id": announce_id,
            "announce_name": announce_name,
            "customer": customer,
            "lot_id": lot_id,
            "lot_name": lot_name,
            "quantity": quantity,
            "amount_tg": amount,
            "trade_method": method,
            "status": status,
        })
    return rows


def scrape(pages: int, count_record: int, out_path: str, delay=(1.5, 3.0)):
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    seen_lot_ids = set()
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for page in range(1, pages + 1):
            url = f"{BASE_URL}?count_record={count_record}&page={page}"
            rows = None
            for attempt in range(3):
                try:
                    resp = session.get(url, timeout=20)
                    resp.raise_for_status()
                    rows = parse_page(resp.text)
                    break
                except requests.RequestException as e:
                    print(f"[page {page}] попытка {attempt + 1} не удалась: {e}")
                    time.sleep(5)

            if not rows:
                print(f"[page {page}] строк не найдено — похоже, страницы закончились. Останавливаюсь.")
                break

            # The public registry can shift while pages are being collected.
            # Keep each real lot once even if it appears again on a later page.
            unique_rows = []
            for row in rows:
                lot_id = str(row.get("lot_id", "")).strip()
                if lot_id and lot_id in seen_lot_ids:
                    continue
                if lot_id:
                    seen_lot_ids.add(lot_id)
                unique_rows.append(row)

            writer.writerows(unique_rows)
            all_rows.extend(unique_rows)
            print(f"[page {page}] +{len(unique_rows)} уникальных строк (всего {len(all_rows)})")
            time.sleep(random.uniform(*delay))

    print(f"\nГотово. Сохранено {len(all_rows)} строк в {out_path}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=100, help="сколько страниц забрать (по 50 записей)")
    parser.add_argument("--count-record", type=int, default=50)
    parser.add_argument("--out", type=str, default="goszakup_lots.csv")
    args = parser.parse_args()

    scrape(pages=args.pages, count_record=args.count_record, out_path=args.out)
