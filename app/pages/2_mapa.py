import os

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from app.pipeline.loader import load_and_preprocess
from app.pipeline.accumulator import calculate_accumulations
from app.pipeline.alerts import calculate_irc, NIVEIS_RISCO
from app.components.map_builder import build_risk_map


DATA_PATH = os.environ.get("DATA_PATH", "data/inmet_sp.csv")


def _load_pipeline():
    df = load_and_preprocess(DATA_PATH)
    df = calculate_accumulations(df)
    df = calculate_irc(df)
    return df


st.header("🗺️ Mapa Interativo")
st.caption("Visualize as estacoes meteorologicas coloridas pelo nivel de risco atual.")

try:
    with st.spinner("Carregando dados e calculando risco..."):
        df = _load_pipeline()

    if df.empty:
        st.warning("O dataset esta vazio apos o preprocessamento.")
        st.stop()

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
    st.error("Arquivo de dados nao encontrado.")
    st.info(f"Coloque o CSV em `{DATA_PATH}` ou configure a variavel de ambiente `DATA_PATH`.")
except pd.errors.EmptyDataError:
    st.error("O arquivo CSV esta vazio ou mal formatado.")
except Exception as e:
    st.error(f"Erro inesperado ao carregar dados: {e}")
    st.info("Verifique o formato do arquivo CSV e tente novamente.")