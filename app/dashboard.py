"""Streamlit interface for the precomputed Tender Scope dataset."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.network import build_supplier_graph, supplier_graph_figure


DEFAULT_SCORED_PATH = ROOT / "data" / "processed" / "scored_tenders.parquet"
RISK_RED = "#C2413B"
RISK_AMBER = "#D69E2E"
RISK_GREEN = "#2F855A"


@st.cache_data(show_spinner=False)
def load_scored_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def normalize_factors(value: Any) -> list[dict[str, Any]]:
    """Support nested Parquet values as well as JSON exported by other engines."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def risk_row_style(row: pd.Series) -> list[str]:
    """Use restrained traffic-light backgrounds without overpowering the table."""
    risk = float(row.get("risk_score", 0))
    if risk >= 80:
        color = "rgba(194, 65, 59, 0.13)"
    elif risk >= 50:
        color = "rgba(214, 158, 46, 0.12)"
    else:
        color = "transparent"
    return [f"background-color: {color}" for _ in row]


def factor_icon(feature: str) -> str:
    return {
        "single_bidder_flag": "👤",
        "price_zscore": "💰",
        "competitive_method_flag": "📋",
        "window_ratio": "⏱",
        "customer_category_hhi": "🔗",
        "data_availability": "◌",
    }.get(feature, "◆")


def factor_title(feature: str) -> str:
    return {
        "single_bidder_flag": "Один участник",
        "price_zscore": "Цена выше группы",
        "competitive_method_flag": "Способ закупки",
        "window_ratio": "Короткое окно",
        "customer_category_hhi": "Концентрация побед",
        "data_availability": "Доступность данных",
    }.get(feature, "Наблюдаемый фактор")


def render_factor_cards(factors: list[dict[str, Any]]) -> None:
    if not factors:
        st.info("Для этой записи недостаточно данных для разбора отклонений.")
        return
    columns = st.columns(len(factors))
    for column, factor in zip(columns, factors):
        feature = str(factor.get("feature", ""))
        text = html.escape(str(factor.get("text", "Нет описания")))
        title = html.escape(factor_title(feature))
        icon = factor_icon(feature)
        with column:
            st.markdown(
                f"""
                <div class="factor-card">
                    <div class="factor-icon">{icon}</div>
                    <div class="factor-title">{title}</div>
                    <div class="factor-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metrics(filtered: pd.DataFrame) -> None:
    total = len(filtered)
    high_risk = int(filtered["risk_score"].ge(80).sum()) if total else 0
    average_risk = float(filtered["risk_score"].mean()) if total else 0.0
    total_amount = float(filtered["amount_tg"].fillna(0).sum()) if total else 0.0
    columns = st.columns(4)
    columns[0].metric("📋 Тендеров", f"{total:,}".replace(",", " "))
    columns[1].metric("🔴 Высокий риск", f"{high_risk:,}".replace(",", " "))
    columns[2].metric("📊 Средний риск", f"{average_risk:.1f} / 100")
    columns[3].metric("₸ Объём выборки", f"{total_amount / 1_000_000_000:.1f} млрд")


def render_tender_table(filtered: pd.DataFrame, top_n: int) -> None:
    table_columns = [
        "tender_id", "customer", "category", "amount_tg",
        "trade_method", "risk_score", "source",
    ]
    table = filtered.loc[:, table_columns].head(top_n).copy()
    styled = table.style.apply(risk_row_style, axis=1)
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=min(720, 40 + 35 * max(1, len(table))),
        column_config={
            "tender_id": st.column_config.TextColumn("ID тендера", width="small"),
            "customer": st.column_config.TextColumn(
                "Заказчик",
                width="medium",
                help="Длинные названия сокращаются визуально; наведите на ячейку, чтобы увидеть полное.",
            ),
            "category": st.column_config.TextColumn("Категория", width="medium"),
            "amount_tg": st.column_config.NumberColumn("Сумма", format="₸ %,.0f"),
            "trade_method": st.column_config.TextColumn("Способ закупки", width="medium"),
            "risk_score": st.column_config.ProgressColumn(
                "Риск",
                help="Перцентиль статистической нетипичности внутри источника",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "source": st.column_config.TextColumn("Источник", width="small"),
        },
    )


def themed_chart(figure: Any) -> Any:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#243244"},
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="Tender Scope", page_icon="◈", layout="wide")
    st.markdown(
        """
        <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1480px;}
        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #DDE3EA; border-radius: 14px;
            padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(21, 38, 59, .045);
        }
        [data-testid="stMetricLabel"] {color: #526173;}
        .factor-card {
            min-height: 190px; padding: 1.15rem; border: 1px solid #DDE3EA;
            border-radius: 14px; background: #FFFFFF;
            box-shadow: 0 6px 20px rgba(21, 38, 59, .055);
        }
        .factor-icon {font-size: 1.65rem; margin-bottom: .65rem;}
        .factor-title {font-size: .95rem; font-weight: 700; color: #243B5A; margin-bottom: .45rem;}
        .factor-text {font-size: .9rem; line-height: 1.45; color: #526173;}
        .mission {color: #526173; font-size: 1.02rem; margin-top: -.55rem; margin-bottom: 1.2rem;}
        .risk-legend {color: #687587; font-size: .84rem; margin: .15rem 0 .8rem;}
        .stTabs [data-baseweb="tab-list"] {gap: .5rem; border-bottom: 1px solid #DDE3EA;}
        .stTabs [data-baseweb="tab"] {padding: .7rem 1rem; border-radius: 9px 9px 0 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Tender Scope")
    st.markdown(
        '<div class="mission">Помогаем аналитикам находить статистически нетипичные закупки для приоритетной ручной проверки.</div>',
        unsafe_allow_html=True,
    )

    if not DEFAULT_SCORED_PATH.exists():
        st.error("Не найден scored_tenders.parquet. Сначала выполните: python -m src.pipeline --source synthetic")
        st.stop()
    data = load_scored_data(str(DEFAULT_SCORED_PATH))
    if data.empty:
        st.warning("В файле нет тендеров для отображения.")
        st.stop()

    with st.sidebar:
        st.header("Параметры выборки")
        sources = sorted(data["source"].dropna().astype(str).unique().tolist())
        selected_sources = st.multiselect("Источник", sources, default=sources)
        categories = sorted(data["category"].fillna("Не указана").astype(str).unique().tolist())
        selected_categories = st.multiselect(
            "Категория",
            categories,
            default=[],
            placeholder="Все категории",
            help="Оставьте поле пустым, чтобы не ограничивать выборку по категории.",
        )
        customer_query = st.text_input("Заказчик содержит", placeholder="Введите часть названия")
        minimum, maximum = int(data["risk_score"].min()), int(data["risk_score"].max())
        risk_range = st.slider("Диапазон риска", minimum, maximum, (minimum, maximum))
        top_max = max(1, min(500, len(data)))
        top_min = min(10, top_max)
        top_n = st.slider("Строк в таблице", top_min, top_max, min(50, top_max))

    filtered = data.copy()
    filtered["category_display"] = filtered["category"].fillna("Не указана").astype(str)
    category_mask = (
        filtered["category_display"].isin(selected_categories)
        if selected_categories
        else pd.Series(True, index=filtered.index)
    )
    filtered = filtered[
        filtered["source"].astype(str).isin(selected_sources)
        & category_mask
        & filtered["risk_score"].between(*risk_range)
    ]
    if customer_query:
        filtered = filtered[filtered["customer"].astype(str).str.contains(customer_query, case=False, na=False)]
    filtered = filtered.sort_values(["risk_score", "raw_score"], ascending=[False, True])

    overview_tab, drilldown_tab, network_tab = st.tabs(["Обзор", "Разбор тендера", "Граф связей"])

    with overview_tab:
        render_metrics(filtered)
        st.subheader("Приоритетная очередь")
        st.markdown(
            '<div class="risk-legend">🔴 80–100 высокий риск &nbsp;&nbsp; 🟡 50–79 средний &nbsp;&nbsp; ⚪ 0–49 фоновый</div>',
            unsafe_allow_html=True,
        )
        render_tender_table(filtered, top_n)

        left, right = st.columns(2)
        with left:
            histogram = px.histogram(
                filtered, x="risk_score", nbins=20, title="Распределение риск-скора",
                color_discrete_sequence=["#314E75"],
            )
            st.plotly_chart(themed_chart(histogram), width="stretch")
        with right:
            category_risk = filtered.groupby("category_display", dropna=False)["risk_score"].mean().nlargest(10).sort_values()
            category_chart = px.bar(
                category_risk,
                orientation="h",
                title="Категории с наибольшим средним риском",
                labels={"value": "Средний риск", "category_display": "Категория"},
                color_discrete_sequence=[RISK_AMBER],
            )
            st.plotly_chart(themed_chart(category_chart), width="stretch")

    with drilldown_tab:
        st.subheader("Разбор отклонений конкретного тендера")
        st.caption("Факторы описывают входные признаки относительно группы сравнения и не являются строгой атрибуцией решения модели.")
        if filtered.empty:
            st.info("В текущей выборке нет тендеров. Измените фильтры в боковой панели.")
        else:
            options = filtered["tender_id"].astype(str).tolist()
            selected_id = st.selectbox("Тендер", options)
            selected = filtered.loc[filtered["tender_id"].astype(str).eq(selected_id)].iloc[0]
            header_columns = st.columns(4)
            header_columns[0].metric("Риск", f"{int(selected['risk_score'])} / 100")
            header_columns[1].metric("Сумма", f"₸ {float(selected['amount_tg']):,.0f}".replace(",", " "))
            header_columns[2].metric("Источник", str(selected["source"]))
            header_columns[3].metric("Категория", str(selected.get("category_display", "Не указана")))
            st.markdown(f"**Заказчик:** {selected.get('customer', 'Не указан')}")
            render_factor_cards(normalize_factors(selected.get("top_factors", [])))

    with network_tab:
        st.warning("Граф предназначен для дальнейшего изучения аналитиком и не является доказательством сговора.")
        graph = build_supplier_graph(filtered)
        if graph.number_of_edges() == 0:
            st.info(
                "В выбранной выборке нет данных о победителях или связей между поставщиками. "
                "Для scraped-источника это ожидаемо: публичный реестр лотов не содержит победителей."
            )
        else:
            maximum_weight = max(int(data.get("weight", 1)) for _, _, data in graph.edges(data=True))
            control_left, control_right = st.columns([1, 3])
            with control_left:
                if maximum_weight > 1:
                    minimum_weight = st.slider(
                        "Минимальный вес связи",
                        min_value=1,
                        max_value=maximum_weight,
                        value=min(2, maximum_weight),
                        help="Скрывает слабые связи между поставщиками через общих заказчиков.",
                    )
                else:
                    minimum_weight = 1
                    st.caption("Все доступные связи имеют единичный вес; фильтр веса не требуется.")
            with control_right:
                st.caption("Показаны до 60 поставщиков с наибольшим числом связей после применения порога.")
            network_figure = supplier_graph_figure(graph, max_nodes=60, min_edge_weight=minimum_weight)
            st.plotly_chart(network_figure, width="stretch")


if __name__ == "__main__":
    main()
