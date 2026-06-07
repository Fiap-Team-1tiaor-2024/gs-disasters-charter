from typing import List, Optional

import pandas as pd


JANELAS_PADRAO = [1, 6, 24, 72]


def calculate_accumulations(
    df: pd.DataFrame,
    janelas_horas: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calcula acumulados de precipitação em janelas temporais por estação.

    Args:
        df: DataFrame com colunas 'precipitacao' e 'estacao', indexado por timestamp.
        janelas_horas: Lista de tamanhos de janela em horas. Padrão: [1, 6, 24, 72].

    Returns:
        DataFrame com colunas adicionais de acumulado (precip_acc_1h, etc.).
    """
    if janelas_horas is None:
        janelas_horas = JANELAS_PADRAO

    required = {"precipitacao", "estacao"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("O índice do DataFrame precisa ser um DatetimeIndex")

    df = df.copy()
    df = df.sort_values(by=["estacao", df.index.name or "timestamp"])

    for janela in janelas_horas:
        col_nome = f"precip_acc_{janela}h"
        df[col_nome] = (
            df.groupby("estacao")["precipitacao"]
            .rolling(window=janela, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    return df