import os

import streamlit as st
import pandas as pd

from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.alerts import calculate_irc, CORES_RISCO
from pipeline.utils import to_pandas
from components.chart_builder import plot_risk_distribution, plot_risk_stations

DATA_PATH = os.environ.get("DATA_PATH", "data/inmet_sp.csv")


def _load_pipeline():
    df_pl = load_and_preprocess(DATA_PATH)
    df_pl = calculate_accumulations(df_pl)
    df_pl = calculate_irc(df_pl)
    df = to_pandas(df_pl)
    return df


st.header("🏠 Dashboard")
st.caption("Resumo de alertas ativos, IRC por estacao e distribuicao de risco.")

try:
    with st.spinner("Carregando dados meteorologicos..."):
        df = _load_pipeline()

    if df.empty:
        st.warning("O dataset esta vazio apos o preprocessamento.")
        st.stop()

    st.success(f"Dados carregados: {len(df):,} registros | {df['estacao'].nunique() if 'estacao' in df.columns else 0} estacoes")

    estacoes_unicas = df["estacao"].unique() if "estacao" in df.columns else []

    with st.sidebar:
        st.subheader("Filtros")
        if "estacao" in df.columns:
            estacao_sel = st.selectbox("Estacao", options=["Todas"] + list(estacoes_unicas))
        else:
            estacao_sel = "Todas"

        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            data_sel = st.date_input(
                "Periodo",
                value=(data_min, data_max),
                min_value=data_min,
                max_value=data_max,
            )
        else:
            data_sel = None

    df_filtrado = df.copy()
    if estacao_sel != "Todas" and "estacao" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["estacao"] == estacao_sel]

    if isinstance(data_sel, tuple) and len(data_sel) == 2 and isinstance(df_filtrado.index, pd.DatetimeIndex):
        inicio = pd.Timestamp(data_sel[0])
        fim = pd.Timestamp(data_sel[1])
        mask = (df_filtrado.index >= inicio) & (df_filtrado.index <= fim)
        df_filtrado = df_filtrado[mask]

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados. Tente ampliar o periodo.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    n_estacoes_total = df["estacao"].nunique() if "estacao" in df.columns else 0
    n_estacoes_risco = 0
    if "nivel_risco" in df_filtrado.columns:
        try:
            df_ultima = df_filtrado[df_filtrado.index == df_filtrado.index.max()]
            n_estacoes_risco = df_ultima[df_ultima["nivel_risco"].isin(["Alto", "Muito Alto", "Critico"])]["estacao"].nunique()
        except Exception:
            n_estacoes_risco = 0

    periodo_str = f"{df_filtrado.index.min().strftime('%d/%m/%Y')} a {df_filtrado.index.max().strftime('%d/%m/%Y')}" if isinstance(df_filtrado.index, pd.DatetimeIndex) and len(df_filtrado) > 0 else "N/A"

    with col1:
        st.metric("Estacoes Monitoradas", value=n_estacoes_total)

    with col2:
        st.metric("Estacoes em Alerta", value=n_estacoes_risco, delta="Alto ou acima" if n_estacoes_risco > 0 else None)

    with col3:
        st.metric("Periodo dos Dados", value=periodo_str)

    st.divider()

    st.subheader("Alertas Ativos")
    if "nivel_risco" in df_filtrado.columns:
        niveis_alerta = ["Muito Alto", "Critico"]
        df_alertas = df_filtrado[df_filtrado["nivel_risco"].isin(niveis_alerta)].copy()

        if df_alertas.empty:
            niveis_alerta = ["Alto", "Muito Alto", "Critico"]
            df_alertas = df_filtrado[df_filtrado["nivel_risco"].isin(niveis_alerta)].copy()

        if not df_alertas.empty:
            df_alertas_str = df_alertas.copy()
            df_alertas_str["nivel_risco"] = df_alertas_str["nivel_risco"].astype(str)
            ranking = (
                df_alertas_str.groupby("estacao", observed=True)
                .agg({"irc": "max", "nivel_risco": "max", "precip_acc_24h": "max"})
                .reset_index()
                .sort_values("irc", ascending=False)
                .head(20)
            )

            ranking_display = ranking.rename(columns={
                "estacao": "Estacao",
                "irc": "IRC",
                "nivel_risco": "Nivel de Risco",
                "precip_acc_24h": "Acum. 24h (mm)",
            })

            def _color_nivel(val):
                cor = CORES_RISCO.get(val, "#999999")
                return f"background-color: {cor}; color: white; font-weight: bold"

            styled = ranking_display.style.map(_color_nivel, subset=["Nivel de Risco"])
            st.dataframe(styled, use_container_width=True)

            st.subheader("Ranking de Risco por Estacao")
            fig_ranking = plot_risk_stations(ranking)
            st.pyplot(fig_ranking)
        else:
            st.info("Nenhuma estacao com alerta alto ou acima no periodo selecionado.")

    st.divider()

    st.subheader("Distribuicao de Risco")
    fig_dist = plot_risk_distribution(df_filtrado)
    st.pyplot(fig_dist)

except FileNotFoundError:
    st.error("Arquivo de dados nao encontrado.")
    st.info(f"Coloque o CSV em `{DATA_PATH}` ou configure a variavel de ambiente `DATA_PATH`.")
except pd.errors.EmptyDataError:
    st.error("O arquivo CSV esta vazio ou mal formatado.")
    st.info("Verifique se o arquivo contem dados validos.")
except Exception as e:
    st.error(f"Erro inesperado ao carregar dados: {e}")
    st.info("Verifique o formato do arquivo CSV e tente novamente.")