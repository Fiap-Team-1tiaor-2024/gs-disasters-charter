import os

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.alerts import calculate_irc, NIVEIS_RISCO
from pipeline.utils import to_pandas
from components.map_builder import build_risk_map
from ml.model_interface import get_model_package, run_batch_prediction, get_station_summary


DATA_PATH = os.environ.get("DATA_PATH", "data/dataset/inmet_sp.csv")


def _load_pipeline():
    df_pl = load_and_preprocess(DATA_PATH)
    df_pl = calculate_accumulations(df_pl)
    df_pl = calculate_irc(df_pl)
    df = to_pandas(df_pl)
    return df, df_pl


st.header("🗺️ Mapa Interativo")
st.caption("Visualize as estações meteorológicas coloridas pelo nível de risco atual.")

try:
    with st.spinner("Carregando dados e calculando risco..."):
        df, df_pl = _load_pipeline()

    if df.empty:
        st.warning("O dataset está vazio após o pré-processamento.")
        st.stop()

    df_resumo_ml = None
    pacote = get_model_package()
    if pacote is not None:
        with st.spinner("Executando predição ML..."):
            try:
                df_predito = run_batch_prediction(df_pl)
                if df_predito is not None:
                    df_resumo_ml = get_station_summary(df_predito)
            except Exception:
                df_resumo_ml = None

    with st.sidebar:
        st.subheader("Filtros do Mapa")

        ordem_risco = {"Normal": 0, "Atencao": 1, "Alto": 2, "Muito Alto": 3, "Critico": 4}
        nomes_exibicao = {"Normal": "Normal", "Atencao": "Atenção", "Alto": "Alto", "Muito Alto": "Muito Alto", "Critico": "Crítico"}
        nivel_minimo = st.selectbox(
            "Nível de risco mínimo",
            options=list(ordem_risco.keys()),
            format_func=lambda x: nomes_exibicao.get(x, x),
            index=0,
        )

        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            data_sel = st.date_input(
                "Data para exibição",
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
        df_resumo_ml=df_resumo_ml,
    )

    st_folium(
        risco_map,
        width=900,
        height=600,
        returned_objects=[],
    )

    if df_resumo_ml is not None and not df_resumo_ml.empty:
        st.subheader("Resumo ML por Estação")
        st.dataframe(df_resumo_ml, use_container_width=True)

except FileNotFoundError:
    st.error("Arquivo de dados não encontrado.")
    st.info(f"Coloque o CSV em `{DATA_PATH}` ou configure a variável de ambiente `DATA_PATH`.")
except pd.errors.EmptyDataError:
    st.error("O arquivo CSV está vazio ou mal formatado.")
except Exception as e:
    st.error(f"Erro inesperado ao carregar dados: {e}")
    st.info("Verifique o formato do arquivo CSV e tente novamente.")