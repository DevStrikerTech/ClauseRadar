from typing import Any, Dict, List

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def render_results_table(results: List[Dict[str, Any]]) -> None:
    """
    Display an interactive Plotly table of search results.
    Columns: Rank, Contract ID, Section/Keyword, Similarity Score (percent), Snippet (preview).
    Full snippets are shown via the "View full snippet" select box below.
    """
    ranks = list(range(1, len(results) + 1))
    contract_ids = [r["contract_id"] for r in results]
    keywords = [r["keyword"] for r in results]

    # Convert each raw score (e.g. 0.9274) into a percentage string (e.g. "92.7 %")
    percent_scores = [f"{r['score'] * 100:.1f} %" for r in results]

    # Build snippet preview (first 80 chars)
    previews: List[str] = []
    for r in results:
        full = r["snippet"].replace("\n", " ")
        if len(full) > 80:
            previews.append(full[:80].rstrip() + "…")
        else:
            previews.append(full)

    table = go.Figure(
        data=go.Table(
            header=dict(
                values=["Rank", "Contract", "Section", "Score", "Snippet"],
                fill_color="rgba(58, 71, 80, 0.8)",
                font=dict(color="white", size=14),
                align="left",
            ),
            cells=dict(
                values=[ranks, contract_ids, keywords, percent_scores, previews],
                fill_color="rgba(245, 245, 245, 0.8)",
                font=dict(color="black", size=12),
                align="left",
                height=30,
            ),
        )
    )

    row_height = 30
    header_height = 40
    total_height = header_height + row_height * len(results) + 20
    table.update_layout(
        margin=dict(l=5, r=5, t=10, b=10),
        width=900,
        height=total_height,
    )

    st.plotly_chart(table, use_container_width=True)


def render_score_bar(results: List[Dict[str, Any]]) -> None:
    """
    Display a horizontal bar chart of similarity scores (in percent) for the results.
    """
    # Build labels and numeric percentages for the chart
    labels = [f"{r['contract_id']} — {r['keyword']}" for r in results]
    numeric_percents = [r["score"] * 100 for r in results]
    text_percents = [f"{p:.1f} %" for p in numeric_percents]

    fig = px.bar(
        x=numeric_percents,
        y=labels,
        orientation="h",
        labels={"x": "Similarity (%)", "y": "Contract — Section"},
        title="Similarity Scores (percent)",
        text=text_percents,
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
        xaxis_tickformat=".1f %",  # show ticks as percentages
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
