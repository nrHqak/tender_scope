"""Exploratory supplier co-participation graph; never used for risk scoring."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


RISK_COLORSCALE = [
    [0.0, "#2F855A"],
    [0.49, "#68A36F"],
    [0.50, "#D69E2E"],
    [0.79, "#DD8A32"],
    [0.80, "#C2413B"],
    [1.0, "#9B2C2C"],
]


def build_supplier_graph(frame: pd.DataFrame) -> nx.Graph:
    """Project customer/winner history into supplier pairs with interpretable weights."""
    graph = nx.Graph()
    history = frame.dropna(subset=["customer", "winner_supplier"])
    supplier_stats = history.groupby("winner_supplier", dropna=False).agg(
        tender_count=("winner_supplier", "size"),
        mean_risk=("risk_score", "mean"),
    )
    for supplier, stats in supplier_stats.iterrows():
        graph.add_node(
            supplier,
            tender_count=int(stats["tender_count"]),
            mean_risk=float(stats["mean_risk"]),
        )

    for customer, customer_rows in history.groupby("customer"):
        counts = customer_rows["winner_supplier"].value_counts()
        for left, right in combinations(sorted(counts.index), 2):
            weight = int(min(counts[left], counts[right]))
            if not weight:
                continue
            existing = graph.get_edge_data(left, right, {})
            customer_weights = dict(existing.get("customer_weights", {}))
            customer_weights[str(customer)] = customer_weights.get(str(customer), 0) + weight
            graph.add_edge(
                left,
                right,
                weight=int(existing.get("weight", 0)) + weight,
                customer_weights=customer_weights,
            )
    return graph


def _visible_graph(graph: nx.Graph, max_nodes: int, min_edge_weight: int) -> nx.Graph:
    eligible_edges = [
        (left, right)
        for left, right, data in graph.edges(data=True)
        if int(data.get("weight", 1)) >= min_edge_weight
    ]
    filtered = graph.edge_subgraph(eligible_edges).copy()
    ranked_nodes = sorted(
        filtered.nodes,
        key=lambda node: (
            filtered.degree(node),
            int(filtered.nodes[node].get("tender_count", 0)),
            str(node),
        ),
        reverse=True,
    )[:max_nodes]
    return filtered.subgraph(ranked_nodes).copy()


def _edge_hover(data: dict) -> str:
    customer_weights = data.get("customer_weights", {})
    details = sorted(customer_weights.items(), key=lambda item: (-item[1], item[0]))
    customer_lines = "<br>".join(
        f"{customer}: {weight}"
        for customer, weight in details[:8]
    )
    if len(details) > 8:
        customer_lines += f"<br>…ещё {len(details) - 8}"
    return f"Вес связи: {int(data.get('weight', 1))}<br>Общие заказчики:<br>{customer_lines}"


def supplier_graph_figure(
    graph: nx.Graph,
    max_nodes: int = 60,
    min_edge_weight: int = 1,
) -> go.Figure:
    """Render a themed, bounded Plotly view of the existing supplier projection."""
    visible = _visible_graph(graph, max_nodes=max_nodes, min_edge_weight=min_edge_weight)
    if visible.number_of_nodes() == 0:
        return go.Figure().update_layout(
            title="Нет связей, соответствующих выбранному порогу",
            template="plotly_white",
            height=680,
        )

    positions = nx.spring_layout(visible, seed=42, weight="weight", k=1.15 / visible.number_of_nodes() ** 0.5)
    edge_buckets: dict[float, dict[str, list]] = defaultdict(lambda: {"x": [], "y": [], "text": []})
    weights = [int(data.get("weight", 1)) for _, _, data in visible.edges(data=True)]
    max_weight = max(weights, default=1)
    for left, right, data in visible.edges(data=True):
        weight = int(data.get("weight", 1))
        width = round(0.7 + 4.3 * (weight / max_weight) ** 0.55, 1)
        hover = _edge_hover(data)
        bucket = edge_buckets[width]
        bucket["x"].extend([positions[left][0], positions[right][0], None])
        bucket["y"].extend([positions[left][1], positions[right][1], None])
        bucket["text"].extend([hover, hover, None])

    traces: list[go.Scatter] = []
    for width, coordinates in sorted(edge_buckets.items()):
        traces.append(go.Scatter(
            x=coordinates["x"],
            y=coordinates["y"],
            mode="lines",
            text=coordinates["text"],
            hoverinfo="text",
            line={"width": width, "color": "rgba(91, 109, 130, 0.35)"},
            showlegend=False,
        ))

    nodes = list(visible.nodes)
    tender_counts = [int(visible.nodes[node].get("tender_count", 0)) for node in nodes]
    mean_risks = [float(visible.nodes[node].get("mean_risk", 0.0)) for node in nodes]
    max_tenders = max(tender_counts, default=1)
    node_sizes = [13 + 27 * (count / max_tenders) ** 0.5 for count in tender_counts]
    node_hover = [
        f"<b>{node}</b><br>Тендеров: {count}<br>Средний риск: {risk:.1f} / 100<br>Связей: {visible.degree(node)}"
        for node, count, risk in zip(nodes, tender_counts, mean_risks)
    ]
    traces.append(go.Scatter(
        x=[positions[node][0] for node in nodes],
        y=[positions[node][1] for node in nodes],
        mode="markers",
        text=node_hover,
        hoverinfo="text",
        marker={
            "size": node_sizes,
            "color": mean_risks,
            "colorscale": RISK_COLORSCALE,
            "cmin": 0,
            "cmax": 100,
            "showscale": True,
            "colorbar": {"title": "Средний<br>риск", "thickness": 13},
            "line": {"width": 1.2, "color": "#FFFFFF"},
        },
        showlegend=False,
    ))

    return go.Figure(data=traces).update_layout(
        title=f"Связи поставщиков · {visible.number_of_nodes()} узлов · {visible.number_of_edges()} связей",
        template="plotly_white",
        height=720,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 30, "t": 55, "b": 10},
    )
