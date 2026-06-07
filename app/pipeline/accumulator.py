import time
from typing import List, Optional

import polars as pl
import streamlit as st


JANELAS_PADRAO = [1, 6, 24, 72]


@st.cache_data(show_spinner="Calculando acumulados de precipitacao...")
def calculate_accumulations(
    df: pl.DataFrame,
    janelas_horas: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Calcula acumulados de precipitacao em janelas temporais por estacao.

    Usa Polars rolling_sum com over("estacao") para garantir que nao
    ha vazamento de dados entre estacoes diferentes.

    O DataFrame deve estar ordenado por timestamp antes desta chamada.
    A funcao ordena por (estacao, timestamp) internamente para o rolling,
    e restaura a ordenacao por timestamp ao final.

    Args:
        df: pl.DataFrame com colunas 'precipitacao', 'estacao' e 'timestamp'.
        janelas_horas: Lista de tamanhos de janela em horas. Padrao: [1, 6, 24, 72].

    Returns:
        pl.DataFrame com colunas adicionais de acumulado
        (precip_acc_1h, precip_acc_6h, precip_acc_24h, precip_acc_72h).
    """
    inicio = time.time()

    if janelas_horas is None:
        janelas_horas = JANELAS_PADRAO

    required = {"precipitacao", "estacao"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    df = df.sort(["estacao", "timestamp"])

    new_cols = []
    for janela in janelas_horas:
        col_nome = f"precip_acc_{janela}h"
        new_cols.append(
            pl.col("precipitacao")
            .rolling_sum(window_size=janela, min_periods=1)
            .over("estacao")
            .cast(pl.Float32)
            .alias(col_nome)
        )

    df = df.with_columns(new_cols)
    df = df.sort("timestamp")

    elapsed = time.time() - inicio
    print(f"[accumulator] Acumulados calculados em {elapsed:.1f}s para janelas {janelas_horas}")

    return df