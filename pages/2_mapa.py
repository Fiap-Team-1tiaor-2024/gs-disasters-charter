import os

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.alerts import calculate_irc, NIVEIS_RISCO
from components.map_builder import build_risk_map


DATA_PATH = os.environ.get("DATA_PATH", "data/inmet_sp.csv")


def _load_pipeline():
    df = load_and_preprocess(DATA_PATH)
    df = calculate_accumulations(df)
    df = calculate_irc(df)
    return df


st.header("🗺️ Mapa Interativo")

try:
    with st.spinner("Carregando dados..."):
        df = _load_pipeline()

    with st.sidebar:
        st.subheader("Filtros do Mapa")

        ordem_risco = {"Normal": 0, "Atencao": 1, "Alto": 2, "Muito Alto": 3, "Critico": 4}
        nivel_minimo = st.selectbox(
            "Nivel de risco minimo",
            options=list(ordem_risco.keys()),
            index=0,
        )

        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            data_sel = st.date_input(
                "Data para exibicao",
                value=data_max,
                min_value=data_min,
                max_value=data_max,
            )
        else:
            data_sel = None

    nivel_risco_filtro = nivel_minimo
    data_str = str(data_sel) if data_sel else None

    risco_map = build_risk_map(
        df_estacoes=df,
        df_irc=df,
        nivel_minimo=nivel_risco_filtro,
        data_filtro=data_str,
    )

    st_folium(
        risco_map,
        width=900,
        height=600,
        returned_objects=[],
    )

except FileNotFoundError:
    st.error("Arquivo de dados nao encontrado. Coloque o CSV em `data/inmet_sp.csv`.")
    st.info("Certifique-se de que o arquivo CSV esta no caminho configurado.")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")