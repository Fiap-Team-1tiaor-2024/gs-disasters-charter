from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from app.pipeline.alerts import CORES_RISCO, NIVEIS_RISCO


def plot_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    """Gera grafico de barras com a distribuicao de estacoes por nivel de risco.

    Args:
        df: DataFrame com coluna 'nivel_risco'.

    Returns:
        Figura matplotlib.
    """
    if "nivel_risco" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sem dados de risco", ha="center", va="center")
        return fig

    contagem = df["nivel_risco"].value_counts()
    niveis_ordenados = [n for n in NIVEIS_RISCO if n in contagem.index]
    valores = [contagem.get(n, 0) for n in niveis_ordenados]
    cores = [CORES_RISCO.get(n, "#999999") for n in niveis_ordenados]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(niveis_ordenados, valores, color=cores, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Nivel de Risco")
    ax.set_ylabel("Numero de Registros")
    ax.set_title("Distribuicao de Risco")
    fig.tight_layout()
    return fig


def plot_risk_stations(ranking: pd.DataFrame) -> plt.Figure:
    """Gera grafico de barras horizontais com ranking de estacoes por IRC.

    Args:
        ranking: DataFrame com colunas 'estacao' e 'irc'.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(ranking) * 0.4)))

    if ranking.empty:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax.transAxes)
        return fig

    cores = [CORES_RISCO.get(n, "#999999") for n in ranking["nivel_risco"]]
    bars = ax.barh(ranking["estacao"], ranking["irc"], color=cores, edgecolor="white")

    ax.set_xlabel("IRC (0-1)")
    ax.set_title("Estacoes com Maior Risco")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_accumulation_timeseries(
    df_estacao: pd.DataFrame,
    janela: str = "24h",
    percentiles: Optional[pd.DataFrame] = None,
    mes: Optional[int] = None,
) -> plt.Figure:
    """Gera grafico de serie temporal do acumulado e IRC com eixo duplo.

    Args:
        df_estacao: DataFrame filtrado para uma estacao, com colunas de acumulado e irc.
        janela: Janela temporal para o acumulado ('1h', '6h', '24h', '72h').
        percentiles: DataFrame de percentis para marcar linhas no grafico.
        mes: Mes para filtrar percentis.

    Returns:
        Figura matplotlib.
    """
    col_acum = f"precip_acc_{janela}"

    fig, ax1 = plt.subplots(figsize=(12, 5))

    if col_acum not in df_estacao.columns:
        ax1.text(0.5, 0.5, f"Coluna {col_acum} nao encontrada", ha="center", transform=ax1.transAxes)
        return fig

    ax1.fill_between(df_estacao.index, df_estacao[col_acum], alpha=0.3, color="steelblue")
    ax1.plot(df_estacao.index, df_estacao[col_acum], color="steelblue", linewidth=1, label=f"Acumulado {janela}")
    ax1.set_ylabel(f"Acumulado {janela} (mm)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    if percentiles is not None and mes is not None:
        estacao_nome = df_estacao["estacao"].iloc[0] if "estacao" in df_estacao.columns else None
        pct_row = None
        if estacao_nome is not None:
            mask = (percentiles["estacao"] == estacao_nome) & (percentiles["mes"] == mes)
            if mask.any():
                pct_row = percentiles[mask].iloc[0]

        if pct_row is not None:
            for p, cor in [(75, "#2ecc71"), (90, "#f1c40f"), (95, "#e67e22"), (99, "#e74c3c")]:
                col_pct = f"P{p}_{janela}"
                if col_pct in pct_row.index:
                    ax1.axhline(y=pct_row[col_pct], color=cor, linestyle="--",
                                alpha=0.7, label=f"P{p}")

    if "irc" in df_estacao.columns:
        ax2 = ax1.twinx()
        ax2.plot(df_estacao.index, df_estacao["irc"], color="red", linewidth=1.2, label="IRC")
        ax2.set_ylabel("IRC (0-1)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(loc="upper left", fontsize=8)

    ax1.set_title(f"Serie Temporal — Acumulado {janela} e IRC")
    fig.tight_layout()
    return fig


def plot_accumulation_histogram(
    df_estacao: pd.DataFrame,
    janela: str = "24h",
    percentiles: Optional[pd.DataFrame] = None,
    mes: Optional[int] = None,
) -> plt.Figure:
    """Gera histograma do acumulado com marcacoes dos percentis.

    Args:
        df_estacao: DataFrame filtrado para uma estacao.
        janela: Janela temporal ('1h', '6h', '24h', '72h').
        percentiles: DataFrame de percentis.
        mes: Mes para filtrar percentis.

    Returns:
        Figura matplotlib.
    """
    col_acum = f"precip_acc_{janela}"

    fig, ax = plt.subplots(figsize=(10, 5))

    if col_acum not in df_estacao.columns:
        ax.text(0.5, 0.5, f"Coluna {col_acum} nao encontrada", ha="center", transform=ax.transAxes)
        return fig

    dados = df_estacao[col_acum].dropna()
    ax.hist(dados, bins=50, color="steelblue", alpha=0.7, edgecolor="white")

    if percentiles is not None and mes is not None:
        estacao_nome = df_estacao["estacao"].iloc[0] if "estacao" in df_estacao.columns else None
        pct_row = None
        if estacao_nome is not None:
            mask = (percentiles["estacao"] == estacao_nome) & (percentiles["mes"] == mes)
            if mask.any():
                pct_row = percentiles[mask].iloc[0]

        if pct_row is not None:
            for p, cor in [(75, "#2ecc71"), (90, "#f1c40f"), (95, "#e67e22"), (99, "#e74c3c")]:
                col_pct = f"P{p}_{janela}"
                if col_pct in pct_row.index:
                    ax.axvline(x=pct_row[col_pct], color=cor, linestyle="--",
                               linewidth=2, label=f"P{p}")

    ax.set_xlabel(f"Acumulado {janela} (mm)")
    ax.set_ylabel("Frequencia")
    ax.set_title(f"Distribuicao Historica — Acumulado {janela}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig