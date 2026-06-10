import os

import streamlit as st
import pandas as pd

from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.alerts import calculate_irc, calculate_historical_percentiles, CORES_RISCO
from pipeline.utils import to_pandas
from components.chart_builder import plot_accumulation_timeseries, plot_accumulation_histogram


DATA_PATH = os.environ.get("DATA_PATH", "data/dataset/inmet_sp.csv")

JANELAS_LABELS = {
    "1h": "1 hora",
    "6h": "6 horas",
    "24h": "24 horas",
    "72h": "72 horas",
}


def _load_pipeline():
    df_pl = load_and_preprocess(DATA_PATH)
    df_pl = calculate_accumulations(df_pl)
    df_pl = calculate_irc(df_pl)
    pct_pl = calculate_historical_percentiles(df_pl)
    df = to_pandas(df_pl)
    pct = pct_pl.to_pandas()
    return df, pct


st.header("📈 Histórico")
st.caption("Análise de acumulados, IRC e percentis históricos por estação e período.")

try:
    with st.spinner("Carregando dados..."):
        df, pct = _load_pipeline()

    if df.empty:
        st.warning("O dataset está vazio após o pré-processamento.")
        st.stop()

    estacoes = sorted(df["estacao"].unique()) if "estacao" in df.columns else []

    if not estacoes:
        st.warning("Nenhuma estação encontrada nos dados.")
        st.stop()

    with st.sidebar:
        st.subheader("Filtros")

        estacao_sel = st.selectbox("Estação", options=estacoes, index=0)

        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            periodo = st.date_input(
                "Período",
                value=(data_min, data_max),
                min_value=data_min,
                max_value=data_max,
            )
        else:
            periodo = None

        janela_sel = st.selectbox(
            "Janela Temporal",
            options=list(JANELAS_LABELS.keys()),
            format_func=lambda x: JANELAS_LABELS[x],
            index=2,
        )

    df_estacao = df[df["estacao"] == estacao_sel].copy()

    if periodo and len(periodo) == 2:
        inicio = pd.Timestamp(periodo[0])
        fim = pd.Timestamp(periodo[1])
        mask = (df_estacao.index >= inicio) & (df_estacao.index <= fim)
        df_estacao = df_estacao[mask]

    if df_estacao.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados. Tente ampliar o período.")
        st.stop()

    mes_selecionado = st.sidebar.selectbox(
        "Mês (para percentis)",
        options=list(range(1, 13)),
        format_func=lambda m: [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
        ][m - 1],
        index=df_estacao.index[0].month - 1 if isinstance(df_estacao.index, pd.DatetimeIndex) else 0,
    )

    st.subheader(f"Série Temporal — {estacao_sel}")
    fig_ts = plot_accumulation_timeseries(
        df_estacao, janela=janela_sel, percentiles=pct, mes=mes_selecionado
    )
    st.pyplot(fig_ts)

    st.divider()

    st.subheader(f"Distribuição Histórica — {JANELAS_LABELS[janela_sel]}")
    fig_hist = plot_accumulation_histogram(
        df_estacao, janela=janela_sel, percentiles=pct, mes=mes_selecionado
    )
    st.pyplot(fig_hist)

    st.divider()

    st.subheader("Top 10 Eventos Críticos")
    if "irc" in df_estacao.columns:
        top_eventos = (
            df_estacao.nlargest(10, "irc")[
                ["estacao", "precip_acc_1h", "precip_acc_6h", "precip_acc_24h", "precip_acc_72h", "irc", "nivel_risco"]
            ]
            .copy()
        )

        top_display = top_eventos.rename(columns={
            "estacao": "Estação",
            "precip_acc_1h": "Acum. 1h (mm)",
            "precip_acc_6h": "Acum. 6h (mm)",
            "precip_acc_24h": "Acum. 24h (mm)",
            "precip_acc_72h": "Acum. 72h (mm)",
            "irc": "IRC",
            "nivel_risco": "Nível de Risco",
        })

        def _color_nivel(val):
            cor = CORES_RISCO.get(val, "#999999")
            return f"background-color: {cor}; color: white; font-weight: bold"

        styled = top_display.style.map(_color_nivel, subset=["Nível de Risco"])
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("IRC não calculado para esta estação.")

except FileNotFoundError:
    st.error("Arquivo de dados não encontrado.")
    st.info(f"Coloque o CSV em `{DATA_PATH}` ou configure a variável de ambiente `DATA_PATH`.")
except pd.errors.EmptyDataError:
    st.error("O arquivo CSV está vazio ou mal formatado.")
except Exception as e:
    st.error(f"Erro inesperado ao carregar dados: {e}")
    st.info("Verifique o formato do arquivo CSV e tente novamente.")