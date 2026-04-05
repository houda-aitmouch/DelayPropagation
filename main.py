from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = Path("data/flights.csv")
REQUIRED_COLUMNS = {
    "flight_id",
    "origin",
    "destination",
    "departure_time",
    "weather_index",
    "congestion_index",
    "turnaround_slack_min",
    "delay_minutes",
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["departure_time"] = pd.to_datetime(df["departure_time"], errors="coerce")
    df = df.dropna(subset=["departure_time"]).copy()
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0.0)
    return df


def build_airport_graph(df: pd.DataFrame) -> nx.DiGraph:
    grouped = (
        df.groupby(["origin", "destination"], as_index=False)
        .agg(avg_delay=("delay_minutes", "mean"), flights=("flight_id", "count"))
        .sort_values(["origin", "destination"])
    )

    graph = nx.DiGraph()
    for row in grouped.itertuples(index=False):
        graph.add_edge(
            row.origin,
            row.destination,
            avg_delay=float(row.avg_delay),
            flights=int(row.flights),
        )
    return graph


def propagate_delay(graph: nx.DiGraph, source: str, initial_delay: float, decay: float, max_hops: int) -> pd.DataFrame:
    if source not in graph.nodes:
        raise ValueError(f"Unknown airport: {source}")

    queue: list[tuple[str, float, int]] = [(source, max(initial_delay, 0.0), 0)]
    best_delay: dict[str, float] = {source: max(initial_delay, 0.0)}

    # Keep strongest propagated delay per airport to avoid over-counting weak revisits.
    while queue:
        airport, delay_value, depth = queue.pop(0)
        if depth >= max_hops or delay_value <= 0:
            continue

        for neighbor in graph.successors(airport):
            edge = graph.get_edge_data(airport, neighbor)
            next_delay = delay_value * decay + float(edge.get("avg_delay", 0.0)) * 0.15
            if next_delay <= best_delay.get(neighbor, -1.0):
                continue
            best_delay[neighbor] = next_delay
            queue.append((neighbor, next_delay, depth + 1))

    result = pd.DataFrame(
        [{"airport": airport, "propagated_delay": round(value, 2)} for airport, value in best_delay.items()]
    ).sort_values("propagated_delay", ascending=False)
    return result


def render_metrics(df: pd.DataFrame) -> None:
    mean_delay = df["delay_minutes"].mean()
    p90_delay = df["delay_minutes"].quantile(0.90)
    impacted = int((df["delay_minutes"] >= 15).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Delay moyen", f"{mean_delay:.1f} min")
    c2.metric("P90 delay", f"{p90_delay:.1f} min")
    c3.metric("Vols >= 15 min", f"{impacted}")


def render_charts(df: pd.DataFrame, propagated: pd.DataFrame) -> None:
    per_airport = (
        df.groupby("origin", as_index=False)
        .agg(avg_delay=("delay_minutes", "mean"), flights=("flight_id", "count"))
        .sort_values("avg_delay", ascending=False)
    )

    fig1 = px.bar(
        per_airport,
        x="origin",
        y="avg_delay",
        color="flights",
        title="Retard moyen par aéroport de départ",
        labels={"origin": "Aéroport", "avg_delay": "Retard moyen (min)", "flights": "Vols"},
    )

    fig2 = px.bar(
        propagated.head(10),
        x="airport",
        y="propagated_delay",
        title="Top aéroports impactés par la propagation",
        labels={"airport": "Aéroport", "propagated_delay": "Retard propagé (min)"},
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)


def run_app() -> None:
    st.set_page_config(page_title="DelayPropagation", layout="wide")
    st.title("DelayPropagation - Analyse de propagation des retards")
    st.caption("Projet de démonstration livrable: génération, simulation, visualisation.")

    st.sidebar.header("Paramètres")
    data_path = Path(st.sidebar.text_input("Fichier CSV", str(DATA_FILE)))

    if not data_path.exists():
        st.error(
            "Dataset introuvable. Générez-le avec: python generate_data.py --output data/flights.csv"
        )
        return

    try:
        df = load_dataset(data_path)
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.exception(exc)
        return

    graph = build_airport_graph(df)
    airports = sorted(graph.nodes)
    source = st.sidebar.selectbox("Aéroport source", airports)
    initial_delay = st.sidebar.slider("Retard initial (min)", min_value=5, max_value=180, value=45, step=5)
    decay = st.sidebar.slider("Facteur de dissipation", min_value=0.30, max_value=0.95, value=0.65, step=0.05)
    max_hops = st.sidebar.slider("Nombre maximum de correspondances", min_value=1, max_value=8, value=4, step=1)

    propagated = propagate_delay(graph, source, float(initial_delay), float(decay), int(max_hops))

    render_metrics(df)
    render_charts(df, propagated)

    st.subheader("Données de propagation")
    st.dataframe(propagated, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run_app()
