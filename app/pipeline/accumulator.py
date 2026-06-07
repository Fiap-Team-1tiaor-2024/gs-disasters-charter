import time
from typing import List, Optional

import pandas as pd
import streamlit as st


JANELAS_PADRAO = [1, 6, 24, 72]


@st.cache_data(show_spinner="Calculando acumulados de precipitacao...")
def calculate_accumulations(
    df: pd.DataFrame,
    janelas_horas: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calcula acumulados de precipitacao em janelas temporais por estacao.

    Tempo estimado: ~30-60s para ~4.6M registros com 4 janelas.
    Carregamentos subsequentes usam cache do Streamlit.

    Args:
        df: DataFrame com colunas 'precipitacao' e 'estacao', indexado por timestamp.
        janelas_horas: Lista de tamanhos de janela em horas. Padrao: [1, 6, 24, 72].

    Returns:
        DataFrame com colunas adicionais de acumulado (precip_acc_1h, precip_acc_6h,
        precip_acc_24h, precip_acc_72h).
    """
    inicio = time.time()

    if janelas_horas is None:
        janelas_horas = JANELAS_PADRAO

    required = {"precipitacao", "estacao"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("O indice do DataFrame precisa ser um DatetimeIndex")

    df = df.copy()
    df = df.sort_values(by=["estacao", df.index.name or "timestamp"])

    for janela in janelas_horas:
        col_nome = f"precip_acc_{janela}h"
        df[col_nome] = (
            df.groupby("estacao", observed=True)["precipitacao"]
            .rolling(window=janela, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[col_nome] = df[col_nome].astype("float32")

    df = df.sort_index()

    elapsed = time.time() - inicio
    print(f"[accumulator] Acumulados calculados em {elapsed:.1f}s para janelas {janelas_horas}")

    return df