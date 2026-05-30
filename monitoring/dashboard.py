import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Monitoring — Scoring Crédit",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Monitoring — API Scoring Crédit")
st.caption("Prêt à Dépenser — Suivi des prédictions en production")

# ─── Chargement des logs ──────────────────────────────────────
@st.cache_data
def load_logs():
    logs = []
    with open("logs/predictions.json", "r") as f:
        for line in f:
            logs.append(json.loads(line))
    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    df["heure"] = df["timestamp"].dt.hour
    return df

df = load_logs()

# ─── KPIs ─────────────────────────────────────────────────────
st.subheader("📈 Métriques clés")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total prédictions", len(df))
with col2:
    taux_refus = round(df["decision"].value_counts(normalize=True).get("REFUS", 0) * 100, 1)
    st.metric("Taux de refus", f"{taux_refus}%")
with col3:
    latence_moy = round(df["latence_ms"].mean(), 1)
    st.metric("Latence moyenne", f"{latence_moy} ms")
with col4:
    score_moy = round(df["score"].mean(), 3)
    st.metric("Score moyen", score_moy)

st.divider()

# ─── Distribution des scores ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Distribution des scores")
    fig = px.histogram(
        df, x="score", nbins=40,
        color_discrete_sequence=["#636EFA"],
        title="Distribution des probabilités de défaut"
    )
    fig.add_vline(x=0.16, line_dash="dash", line_color="red",
                  annotation_text="Seuil 0.16")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚖️ Répartition ACCORD / REFUS")
    counts = df["decision"].value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color_discrete_map={"ACCORD": "#00CC96", "REFUS": "#EF553B"},
        title="Décisions de l'API"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Évolution temporelle ─────────────────────────────────────
st.subheader("📅 Évolution temporelle du score moyen")
df_daily = df.groupby("date").agg(
    score_moyen=("score", "mean"),
    nb_predictions=("score", "count"),
    latence_moyenne=("latence_ms", "mean")
).reset_index()

fig = px.line(
    df_daily, x="date", y="score_moyen",
    title="Score moyen par jour — Détection de drift",
    markers=True
)
fig.add_hline(y=0.16, line_dash="dash", line_color="red",
              annotation_text="Seuil métier")
fig.add_hline(y=df["score"].mean(), line_dash="dot", line_color="blue",
              annotation_text="Moyenne globale")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Latence ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Distribution de la latence")
    fig = px.histogram(
        df, x="latence_ms", nbins=30,
        color_discrete_sequence=["#AB63FA"],
        title="Temps d'inférence (ms)"
    )
    fig.add_vline(x=df["latence_ms"].mean(), line_dash="dash",
                  line_color="red", annotation_text="Moyenne")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Latence par heure de la journée")
    df_heure = df.groupby("heure")["latence_ms"].mean().reset_index()
    fig = px.bar(
        df_heure, x="heure", y="latence_ms",
        title="Latence moyenne par heure",
        color_discrete_sequence=["#FFA15A"]
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Data Drift ───────────────────────────────────────────────
st.subheader("🔍 Analyse du Data Drift")

# Découpage en deux fenêtres : référence vs production récente
midpoint = len(df) // 2
df_ref = df.iloc[:midpoint]
df_prod = df.iloc[midpoint:]

col1, col2 = st.columns(2)

with col1:
    st.metric("Score moyen — Référence", round(df_ref["score"].mean(), 3))
    st.metric("Latence moyenne — Référence", f"{round(df_ref['latence_ms'].mean(), 1)} ms")

with col2:
    delta_score = round(df_prod["score"].mean() - df_ref["score"].mean(), 3)
    st.metric("Score moyen — Production", round(df_prod["score"].mean(), 3),
              delta=delta_score, delta_color="inverse")
    delta_lat = round(df_prod["latence_ms"].mean() - df_ref["latence_ms"].mean(), 1)
    st.metric("Latence moyenne — Production", f"{round(df_prod['latence_ms'].mean(), 1)} ms",
              delta=f"{delta_lat} ms")

# Comparaison distributions
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df_ref["score"], name="Référence",
    opacity=0.6, nbinsx=30,
    marker_color="#636EFA"
))
fig.add_trace(go.Histogram(
    x=df_prod["score"], name="Production récente",
    opacity=0.6, nbinsx=30,
    marker_color="#EF553B"
))
fig.update_layout(
    barmode="overlay",
    title="Comparaison distribution scores : Référence vs Production",
    xaxis_title="Score",
    yaxis_title="Nombre de prédictions"
)
fig.add_vline(x=0.16, line_dash="dash", line_color="black",
              annotation_text="Seuil 0.16")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Dashboard généré automatiquement — Prêt à Dépenser MLOps v1.0")
