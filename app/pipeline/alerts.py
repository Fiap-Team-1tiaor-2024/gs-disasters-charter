import time
from typing import Dict, Optional

import polars as pl
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
    "3h": "precip_acc_3h",
    "6h": "precip_acc_6h",
    "12h": "precip_acc_12h",
    "24h": "precip_acc_24h",
    "48h": "precip_acc_48h",
    "72h": "precip_acc_72h",
}


def calculate_historical_percentiles(
    df: pl.DataFrame,
    janelas: Optional[Dict[str, str]] = None,
    percentis: Optional[list] = None,
) -> pl.DataFrame:
    """Calcula percentis historicos de acumulado de chuva por (estacao, mes).

    Usa Polars group_by com expressoes de quantile para cada janela temporal.

    Args:
        df: pl.DataFrame com colunas de acumulado, 'estacao' e 'mes'.
        janelas: Dicionario mapeando nome da janela para nome da coluna.
        percentis: Lista de percentis a calcular. Padrao: [75, 90, 95, 99].

    Returns:
        pl.DataFrame com colunas: estacao, mes, P75_{janela}, P90_{janela}, etc.
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

    agg_exprs = []
    for col_acum in colunas_acumulo:
        nome_janela = [k for k, v in janelas.items() if v == col_acum][0]
        for p in percentis:
            quantile_val = p / 100.0
            agg_exprs.append(
                pl.col(col_acum).quantile(quantile_val, interpolation="linear").alias(f"P{p}_{nome_janela}")
            )

    resultado = df.group_by(["estacao", "mes"]).agg(agg_exprs)
    print(f"[alerts] Percentis calculados para {len(resultado)} combinacoes (estacao, mes)")

    return resultado


def calculate_risk_scores(
    df: pl.DataFrame,
    percentiles: pl.DataFrame,
    janelas: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """Determina o score de risco de cada registro com base nos percentis historicos.

    Para cada registro, verifica em qual faixa de percentil o acumulado se encontra
    e atribui um score de 0 a 4:
        0 = Normal (< P75)
        1 = Atencao (P75 <= x < P90)
        2 = Alto (P90 <= x < P95)
        3 = Muito Alto (P95 <= x < P99)
        4 = Critico (>= P99)

    Args:
        df: pl.DataFrame com colunas de acumulado, 'estacao' e 'mes'.
        percentiles: pl.DataFrame de percentis calculado por calculate_historical_percentiles.
        janelas: Dicionario mapeando nome da janela para nome da coluna.

    Returns:
        pl.DataFrame original com colunas adicionais:
        score_{janela} (0-4) e nivel_{janela} (categorico) para cada janela.
    """
    if janelas is None:
        janelas = JANELAS_MAP

    df = df.join(percentiles, on=["estacao", "mes"], how="left")

    for nome_janela, col_acum in janelas.items():
        if col_acum not in df.columns:
            continue

        p75_col = f"P75_{nome_janela}"
        p90_col = f"P90_{nome_janela}"
        p95_col = f"P95_{nome_janela}"
        p99_col = f"P99_{nome_janela}"

        if p75_col not in df.columns:
            continue

        score_expr = (
            pl.when(pl.col(col_acum) >= pl.col(p99_col)).then(pl.lit(4))
            .when(pl.col(col_acum) >= pl.col(p95_col)).then(pl.lit(3))
            .when(pl.col(col_acum) >= pl.col(p90_col)).then(pl.lit(2))
            .when(pl.col(col_acum) >= pl.col(p75_col)).then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias(f"score_{nome_janela}")
        )

        nivel_expr = (
            pl.when(pl.col(col_acum) >= pl.col(p99_col)).then(pl.lit("Critico"))
            .when(pl.col(col_acum) >= pl.col(p95_col)).then(pl.lit("Muito Alto"))
            .when(pl.col(col_acum) >= pl.col(p90_col)).then(pl.lit("Alto"))
            .when(pl.col(col_acum) >= pl.col(p75_col)).then(pl.lit("Atencao"))
            .otherwise(pl.lit("Normal"))
            .alias(f"nivel_{nome_janela}")
        )

        df = df.with_columns([score_expr, nivel_expr])

    pct_cols = [c for c in df.columns if c.startswith("P") and "_" in c and c != "precipitacao"]
    if pct_cols:
        df = df.drop(pct_cols)

    return df


def adjust_irc_with_sensor(
    df: pl.DataFrame,
    df_sensor: Optional[pl.DataFrame] = None,
    umidade_p90_threshold: float = 85.0,
) -> pl.DataFrame:
    """Ajusta o IRC com base nos dados do sensor ESP32.

    Formula documentada:
        irc_ajustado = irc_base * fator_umidade
        fator_umidade = 1 + (umidade_relativa - umidade_p90_threshold) / 100
                     se umidade > umidade_p90_threshold
                     1.0 caso contrario

    Args:
        df: pl.DataFrame com coluna 'irc' ja calculada.
        df_sensor: pl.DataFrame do sensor com coluna 'umidade'. Se None, nao aplica ajuste.
        umidade_p90_threshold: Limiar de umidade para ativacao do fator (padrao: 85%).

    Returns:
        pl.DataFrame com colunas 'irc' ajustada e 'irc_base' (valor original),
        e 'nivel_risco' recalculado com base no IRC ajustado.
    """
    if df_sensor is None or df_sensor.is_empty() or "umidade" not in df_sensor.columns:
        print("[alerts] Sem dados do sensor — IRC base utilizado sem ajuste")
        return df

    df = df.clone()
    df = df.with_columns(pl.col("irc").alias("irc_base"))

    umidade_media = df_sensor["umidade"].mean()

    if umidade_media is not None and umidade_media > umidade_p90_threshold:
        fator_umidade = 1.0 + (umidade_media - umidade_p90_threshold) / 100.0
        df = df.with_columns([
            (pl.col("irc") * pl.lit(fator_umidade)).clip(0.0, 1.0).cast(pl.Float32).alias("irc"),
            pl.lit(fator_umidade).alias("fator_umidade"),
        ])
        print(f"[alerts] IRC ajustado pelo sensor: umidade_media={umidade_media:.1f}%, fator={fator_umidade:.3f}")
    else:
        df = df.with_columns(pl.lit(1.0).alias("fator_umidade"))
        if umidade_media is not None:
            print(f"[alerts] Umidade ({umidade_media:.1f}%) abaixo do limiar ({umidade_p90_threshold}%) — sem ajuste")

    df = df.with_columns(
        pl.when(pl.col("irc") < 0.25).then(pl.lit("Normal"))
        .when(pl.col("irc") < 0.50).then(pl.lit("Atencao"))
        .when(pl.col("irc") < 0.75).then(pl.lit("Alto"))
        .when(pl.col("irc") < 0.95).then(pl.lit("Muito Alto"))
        .otherwise(pl.lit("Critico"))
        .alias("nivel_risco")
    )

    return df


@st.cache_data(show_spinner="Calculando IRC e niveis de risco...")
def calculate_irc(
    df: pl.DataFrame,
    percentiles: Optional[pl.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    janelas: Optional[Dict[str, str]] = None,
    df_sensor: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Calcula o Indice de Risco Composto (IRC) para cada registro.

    O IRC combina os scores de risco de cada janela temporal em um indice
    normalizado entre 0 e 1:

        IRC = sum(w_janela * score_janela) / sum(w_janela * 4)

    Os pesos padrao sao: 1h=0.35, 6h=0.30, 24h=0.20, 72h=0.15.

    Args:
        df: pl.DataFrame com colunas de acumulado, 'estacao' e 'mes'.
        percentiles: pl.DataFrame de percentis. Se None, sera calculado internamente.
        weights: Pesos para cada janela. Padrao: {"1h": 0.35, "6h": 0.30, "24h": 0.20, "72h": 0.15}.
        janelas: Dicionario mapeando nome da janela para nome da coluna.
        df_sensor: pl.DataFrame do sensor ESP32 com coluna 'umidade'. Se None, nao aplica ajuste.

    Returns:
        pl.DataFrame original com colunas adicionais:
        score_{janela}, nivel_{janela}, irc (0 a 1), nivel_risco (categorico).
        Se df_sensor for fornecido, inclui irc_base e fator_umidade.
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

    irc_num = sum(
        pl.col(col).cast(pl.Float32) * pl.lit(weights.get(nome, 0))
        for nome, col in score_cols
    )
    irc_den = sum(
        4 * weights.get(nome, 0)
        for nome, col in score_cols
    )

    df = df.with_columns(
        (irc_num / pl.lit(irc_den)).clip(0.0, 1.0).cast(pl.Float32).alias("irc")
    )

    df = df.with_columns(
        pl.when(pl.col("irc") < 0.25).then(pl.lit("Normal"))
        .when(pl.col("irc") < 0.50).then(pl.lit("Atencao"))
        .when(pl.col("irc") < 0.75).then(pl.lit("Alto"))
        .when(pl.col("irc") < 0.95).then(pl.lit("Muito Alto"))
        .otherwise(pl.lit("Critico"))
        .alias("nivel_risco")
    )

    if df_sensor is not None:
        df = adjust_irc_with_sensor(df, df_sensor)

    elapsed = time.time() - inicio
    print(f"[alerts] IRC calculado em {elapsed:.1f}s")

    dist = df.group_by("nivel_risco").len().sort("nivel_risco")
    print(f"[alerts] Distribuicao de risco: {dict(zip(dist['nivel_risco'].to_list(), dist['len'].to_list()))}")

    return df