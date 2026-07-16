"""
Генератор синтетических данных госзакупок — по схеме, близкой к реальным полям
goszakup.gov.kz (объявления/лоты/участники).

Зачем: пока скрапер/токен ещё не дали полный объём реальных данных, можно
уже сегодня разрабатывать признаки и модель на этом датасете, а потом
просто подменить источник на реальный (структура полей совместима).

В данные специально "зашит" реалистичный паттерн риска — часть заказчиков
систематически отдают закупки одному и тому же поставщику (единственная
заявка, короткое окно подачи, цена выше рынка). Это даёт вам ground truth
для проверки модели аномалий: после обучения можно посчитать, какую долю
"captured_pairs" модель реально нашла — готовая метрика для презентации.

Использование:
    python generate_synthetic_data.py --tenders 6000 --out synthetic_goszakup.csv
"""

import argparse
import csv
import datetime
import random

import numpy as np

CATEGORIES_BASE_PRICE = {
    "Продукты питания": 500_000,
    "Стройматериалы": 3_000_000,
    "Медицинское оборудование": 8_000_000,
    "Канцелярские товары": 300_000,
    "IT-услуги": 5_000_000,
    "Транспортные услуги": 1_200_000,
    "Ремонт зданий": 6_000_000,
    "Охранные услуги": 2_000_000,
    "Клининг": 700_000,
    "Мебель": 1_000_000,
}

TRADE_METHODS = [
    "Открытый конкурс", "Электронный аукцион", "Запрос ценовых предложений",
    "Из одного источника по несостоявшимся закупкам", "Из одного источника",
]
TRADE_METHOD_WEIGHTS = [30, 25, 20, 15, 10]

STATUSES = ["Опубликован", "Завершен", "Итоги подведены", "Отменен"]
STATUS_WEIGHTS = [10, 60, 25, 5]


def _random_bin(rng: random.Random) -> str:
    return "".join(rng.choices("0123456789", k=12))


def generate_dataset(n_customers=200, n_suppliers=800, n_tenders=6000, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    customers = [f"Заказчик_{i:04d}" for i in range(n_customers)]
    customer_bins = {c: _random_bin(rng) for c in customers}
    suppliers = [f"Поставщик_{i:04d}" for i in range(n_suppliers)]
    supplier_bins = {s: _random_bin(rng) for s in suppliers}

    # ~8% заказчиков имеют скрытую "захваченную" связку с одним поставщиком —
    # это и есть паттерн, который должна находить модель аномалий
    captured_customers = rng.sample(customers, k=max(1, int(n_customers * 0.08)))
    captured_pairs = {c: rng.choice(suppliers) for c in captured_customers}

    rows = []
    start = datetime.date(2023, 1, 1)

    for i in range(n_tenders):
        category = rng.choice(list(CATEGORIES_BASE_PRICE.keys()))
        customer = rng.choice(customers)
        method = rng.choices(TRADE_METHODS, weights=TRADE_METHOD_WEIGHTS)[0]
        publish_date = start + datetime.timedelta(days=int(np_rng.integers(0, 900)))
        base_price = CATEGORIES_BASE_PRICE[category]

        is_captured = customer in captured_pairs and rng.random() < 0.7

        if is_captured:
            n_bidders = 1
            winner = captured_pairs[customer]
            price = base_price * np_rng.uniform(1.15, 1.45)
            window_days = max(1, int(np_rng.normal(4, 1.5)))
        else:
            n_bidders = max(1, int(np_rng.poisson(3.2)))
            winner = rng.choice(suppliers)
            price = base_price * np_rng.uniform(0.85, 1.1)
            window_days = max(1, int(np_rng.normal(12, 5)))

        end_date = publish_date + datetime.timedelta(days=window_days)

        rows.append({
            "tender_id": 400000 + i,
            "category": category,
            "customer": customer,
            "customer_bin": customer_bins[customer],
            "trade_method": method,
            "publish_date": publish_date.isoformat(),
            "end_date": end_date.isoformat(),
            "window_days": window_days,
            "n_bidders": n_bidders,
            "winner_supplier": winner,
            "winner_bin": supplier_bins[winner],
            "amount_tg": round(float(price), 2),
            "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0],
            "is_captured_ground_truth": is_captured,  # уберите перед демо жюри — это для вашей внутренней валидации
        })

    return rows


def save_csv(rows: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранено {len(rows)} синтетических тендеров в {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--suppliers", type=int, default=800)
    parser.add_argument("--tenders", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="synthetic_goszakup.csv")
    args = parser.parse_args()

    data = generate_dataset(args.customers, args.suppliers, args.tenders, args.seed)
    save_csv(data, args.out)
