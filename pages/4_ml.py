import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.alerts import calculate_irc, calculate_historical_percentiles, CORES_RISCO
from ml.model_interface import load_model, build_features, predict_risk, get_model_info, get_feature_importance


DATA_PATH = os.environ.get("DATA_PATH", "data/inmet_sp.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "")


def _load_pipeline():
    df = load_and_preprocess(DATA_PATH)
    df = calculate_accumulations(df)
    df = calculate_irc(df)
    return df


st.header("🤖 Modelo ML")

model = load_model(MODEL_PATH)
model_info = get_model_info(model, MODEL_PATH)

st.subheader("Status do Modelo")
col1, col2 = st.columns(2)

with col1:
    if model_info["carregado"]:
        st.success("Modelo carregado e pronto para predicao")
    else:
        st.warning("Modelo nao carregado — modo stub ativo")

with col2:
    st.write(f"**Tipo:** {model_info['tipo']}")
    st.write(f"**Caminho:** {model_info['caminho'] or 'Nao configurado'}")

if model_info.get("classes"):
    st.write(f"**Classes:** {model_info['classes']}")
if model_info.get("n_features"):
    st.write(f"**Features:** {model_info['n_features']}")

st.divider()

try:
    with st.spinner("Carregando dados..."):
        df = _load_pipeline()

    pct = calculate_historical_percentiles(df)

    features = build_features(df)
    predictions = predict_risk(model, features, df_original=df)

    if model is None and not predictions.empty:
        st.info("O modelo ML ainda nao foi integrado. Abaixo estao as estacoes com maior IRC calculado pelo pipeline como referencia.")

    st.subheader("Predicoes")
    if not predictions.empty:
        display_cols = [c for c in ["estacao", "timestamp", "risco_predito", "probabilidade"] if c in predictions.columns]

        def _color_risco(val):
            if isinstance(val, str):
                cor = CORES_RISCO.get(val, "#999999")
                return f"background-color: {cor}; color: white; font-weight: bold"
            return ""

        styled = predictions[display_cols].style.map(_color_risco, subset=["risco_predito"] if "risco_predito" in display_cols else [])
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Nenhuma predicao disponivel.")

    st.divider()

    st.subheader("Comparacao IRC vs ML")
    if model is not None and "probabilidade" in predictions.columns and "irc" in df.columns:
        df_str = df.copy()
        df_str["nivel_risco"] = df_str["nivel_risco"].astype(str)

        comparacao = df_str.groupby("estacao", observed=True).agg({
            "irc": "max",
            "nivel_risco": "max",
        }).reset_index().sort_values("irc", ascending=False).head(20)

        if "estacao" in predictions.columns and "probabilidade" in predictions.columns:
            pred_agg = predictions.groupby("estacao", observed=True).agg({
                "probabilidade": "max",
            }).reset_index()

            comparacao = comparacao.merge(pred_agg, on="estacao", how="left")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(
                comparacao["irc"],
                comparacao["probabilidade"],
                alpha=0.7, s=80, color="steelblue", edgecolors="white"
            )
            ax.set_xlabel("IRC (Calculado)")
            ax.set_ylabel("Probabilidade ML")
            ax.set_title("Comparacao: IRC Calculado vs Predicao ML")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Dados de predicao insuficientes para comparacao.")
    elif model is None and "irc" in df.columns:
        df_str = df.copy()
        df_str["nivel_risco"] = df_str["nivel_risco"].astype(str)

        ranking = (
            df_str.groupby("estacao", observed=True)
            .agg({"irc": "max", "nivel_risco": "max"})
            .reset_index()
            .sort_values("irc", ascending=False)
            .head(20)
        )

        def _color_nivel(val):
            cor = CORES_RISCO.get(val, "#999999")
            return f"background-color: {cor}; color: white; font-weight: bold"

        st.write("Ranking de estacoes por IRC (predicoes ML serao comparadas quando o modelo estiver disponivel):")
        styled_ranking = ranking.style.map(_color_nivel, subset=["nivel_risco"])
        st.dataframe(styled_ranking, use_container_width=True)

    st.divider()

    st.subheader("Feature Importance")
    importance_df = get_feature_importance(model)
    if importance_df is not None and not importance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
        ax.set_xlabel("Importancia")
        ax.set_title("Feature Importance do Modelo")
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("Feature importance nao disponivel — sera exibida quando o modelo for integrado.")

except FileNotFoundError:
    st.error("Arquivo de dados nao encontrado. Coloque o CSV em `data/inmet_sp.csv`.")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")