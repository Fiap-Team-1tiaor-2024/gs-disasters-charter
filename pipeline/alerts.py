import time
from typing import Dict, Optional

import pandas as pd
import streamlit as st


PERCENTIS = [75, 90, 95, 99]

NIVEIS_RISCO = ["Normal", "Atencao", "Alto", "Muito Alto", "Critico"]

CORES_RISCO = {
    "Normal": "#2ecc71",
    "Atencao": "#f1c40f",
    "Alto": "#e67e22",
    "Muito Alto": "#e74c3c",
    "Critico": "#1a1a1a",
}

PESOS_PADRAO = {"1h": 0.35, "6h": 0.30, "24h": 0.20, "72h": 0.15}

JANELAS_MAP = {
    "1h": "precip_acc_1h",
    "6h": "precip_acc_6h",
    "24h": "precip_acc_24h",
    "72h": "precip_acc_72h",
}


def calculate_historical_percentiles(
    df: pd.DataFrame,
    janelas: Optional[Dict[str, str]] = None,
    percentis: Optional[list] = None,
) -> pd.DataFrame:
    """Calcula percentis historicos de acumulado de chuva por (estacao, mes).

    Para cada combinacao de estacao e mes, calcula os percentis P75, P90, P95, P99
    de cada janela temporal de acumulado.

    Args:
        df: DataFrame com colunas de acumulado e colunas 'estacao' e 'mes'.
        janelas: Dicionario mapeando nome da janela para nome da coluna.
            Padrao: {"1h": "precip_acc_1h", "6h": "precip_acc_6h", ...}
        percentis: Lista de percentis a calcular. Padrao: [75, 90, 95, 99].

    Returns:
        DataFrame com colunas: estacao, mes, P75_{janela}, P90_{janela}, etc.
        Para cada janela temporal.
    """
    if janelas is None:
        janelas = JANELAS_MAP
    if percentis is None:
        percentis = PERCENTIS

    required = {"estacao", "mes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    colunas_acumulo = [col for col in janelas.values() if col in df.columns]
    if not colunas_acumulo:
        raise ValueError(f"Nenhuma coluna de acumulado encontrada. Esperadas: {list(janelas.values())}")

    dfs_percentis = []
    grouped = df.groupby(["estacao", "mes"], observed=True)

    for col_acum in colunas_acumulo:
        nome_janela = [k for k, v in janelas.items() if v == col_acum][0]
        stats = grouped[col_acum].quantile([p / 100 for p in percentis]).unstack()
        stats.columns = [f"P{int(p)}_{nome_janela}" for p in percentis]
        dfs_percentis.append(stats)

    resultado = pd.concat(dfs_percentis, axis=1).reset_index()
    print(f"[alerts] Percentis calculados para {len(resultado)} combinacoes (estacao, mes)")

    return resultado


def calculate_risk_scores(
    df: pd.DataFrame,
    percentiles: pd.DataFrame,
    janelas: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Determina o score de risco de cada registro com base nos percentis historicos.

    Para cada registro, verifica em qual faixa de percentil o acumulado se encontra
    e atribui um score de 0 a 4:
        0 = Normal (< P75)
        1 = Atencao (P75 <= x < P90)
        2 = Alto (P90 <= x < P95)
        3 = Muito Alto (P95 <= x < P99)
        4 = Critico (>= P99)

    Args:
        df: DataFrame com colunas de acumulado, estacao e mes.
        percentiles: DataFrame de percentis calculado por calculate_historical_percentiles.
        janelas: Dicionario mapeando nome da janela para nome da coluna.

    Returns:
        DataFrame original com colunas adicionais:
        score_{janela} (0-4) e nivel_{janela} (categorico) para cada janela.
    """
    if janelas is None:
        janelas = JANELAS_MAP

    df = df.copy()

    percentile_lookup = {}
    for _, row in percentiles.iterrows():
        estacao = row["estacao"]
        mes = row["mes"]
        key = (estacao, mes)
        percentile_lookup[key] = row

    for nome_janela, col_acum in janelas.items():
        if col_acum not in df.columns:
            continue

        p75_col = f"P75_{nome_janela}"
        p90_col = f"P90_{nome_janela}"
        p95_col = f"P95_{nome_janela}"
        p99_col = f"P99_{nome_janela}"

        scores = pd.Series(0, index=df.index, dtype="int8")
        niveis = pd.Series(
            pd.Categorical(["Normal"] * len(df), categories=NIVEIS_RISCO, ordered=True),
            index=df.index,
        )

        for (estacao, mes), pct_row in percentile_lookup.items():
            if p75_col not in pct_row.index:
                continue

            mask = (df["estacao"] == estacao) & (df["mes"] == mes)
            if not mask.any():
                continue

            valores = df.loc[mask, col_acum]
            p75 = pct_row[p75_col]
            p90 = pct_row[p90_col]
            p95 = pct_row[p95_col]
            p99 = pct_row[p99_col]

            scores[mask & (df[col_acum] >= p99)] = 4
            scores[mask & (df[col_acum] >= p95) & (df[col_acum] < p99)] = 3
            scores[mask & (df[col_acum] >= p90) & (df[col_acum] < p95)] = 2
            scores[mask & (df[col_acum] >= p75) & (df[col_acum] < p90)] = 1
            scores[mask & (df[col_acum] < p75)] = 0

            niveis[mask & (df[col_acum] >= p99)] = "Critico"
            niveis[mask & (df[col_acum] >= p95) & (df[col_acum] < p99)] = "Muito Alto"
            niveis[mask & (df[col_acum] >= p90) & (df[col_acum] < p95)] = "Alto"
            niveis[mask & (df[col_acum] >= p75) & (df[col_acum] < p90)] = "Atencao"
            niveis[mask & (df[col_acum] < p75)] = "Normal"

        df[f"score_{nome_janela}"] = scores
        df[f"nivel_{nome_janela}"] = niveis

    return df


@st.cache_data(show_spinner="Calculando IRC e niveis de risco...")
def calculate_irc(
    df: pd.DataFrame,
    percentiles: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    janelas: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Calcula o Indice de Risco Composto (IRC) para cada registro.

    O IRC combina os scores de risco de cada janela temporal em um indice
    normalizado entre 0 e 1:

        IRC = sum(w_janela * score_janela) / sum(w_janela * 4)

    Os pesos padrao sao: 1h=0.35, 6h=0.30, 24h=0.20, 72h=0.15.

    Tempo estimado: ~5-15s para ~4.6M registros.
    Carregamentos subsequentes usam cache do Streamlit.

    Args:
        df: DataFrame com colunas de acumulado, estacao e mes.
        percentiles: DataFrame de percentis. Se None, sera calculado internamente.
        weights: Pesos para cada janela. Padrao: {"1h": 0.35, "6h": 0.30, "24h": 0.20, "72h": 0.15}.
        janelas: Dicionario mapeando nome da janela para nome da coluna.

    Returns:
        DataFrame original com colunas adicionais:
        score_{janela}, nivel_{janela}, irc (0 a 1), nivel_risco (categorico).
    """
    inicio = time.time()

    if weights is None:
        weights = PESOS_PADRAO
    if janelas is None:
        janelas = JANELAS_MAP

    if percentiles is None:
        percentiles = calculate_historical_percentiles(df, janelas=janelas)

    df = calculate_risk_scores(df, percentiles, janelas=janelas)

    score_cols = []
    for nome_janela in janelas:
        col = f"score_{nome_janela}"
        if col in df.columns:
            score_cols.append((nome_janela, col))

    irc_num = pd.Series(0.0, index=df.index, dtype="float32")
    irc_den = 0.0

    for nome_janela, col in score_cols:
        peso = weights.get(nome_janela, 0)
        irc_num += df[col].astype("float32") * peso
        irc_den += 4 * peso

    df["irc"] = (irc_num / irc_den).clip(0, 1).astype("float32")

    def _classificar_risco(irc_val):
        if irc_val < 0.25:
            return "Normal"
        elif irc_val < 0.50:
            return "Atencao"
        elif irc_val < 0.75:
            return "Alto"
        elif irc_val < 0.95:
            return "Muito Alto"
        else:
            return "Critico"

    df["nivel_risco"] = df["irc"].apply(_classificar_risco).astype("category")

    elapsed = time.time() - inicio
    print(f"[alerts] IRC calculado em {elapsed:.1f}s")
    dist = df["nivel_risco"].value_counts()
    print(f"[alerts] Distribuicao de risco: {dict(dist)}")

    return df