# BASELINE PANDAS: ~1.35s (demo 43,800 rows)
# APOS POLARS: ~0.10s (demo 43,800 rows)
import re
import time
import unicodedata
from typing import Optional

import polars as pl
import streamlit as st


COLUNAS_PADRAO = {
    "data_str": "data",
    "hora_str": "hora",
    "precipitacao": "precipitacao_total",
    "estacao": "estacao",
    "latitude": "latitude",
    "longitude": "longitude",
    "municipio": "municipio",
    "estado": "estado",
}


def _normalizar_nome_coluna(col: str) -> str:
    col = unicodedata.normalize("NFKD", col)
    col = col.encode("ASCII", "ignore").decode("ASCII")
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9_]", "_", col)
    col = re.sub(r"_+", "_", col)
    return col


@st.cache_data(show_spinner="Carregando dados meteorologicos...")
def load_and_preprocess(
    filepath: str,
    colunas_map: Optional[dict] = None,
    separador: str = ",",
    codificacao: str = "utf-8",
    decimal: str = ".",
) -> pl.DataFrame:
    """Carrega o CSV do INMET e realiza o preprocessamento completo usando Polars.

    Usa leitura eager (pl.read_csv) com otimizacao de tipos para maximizar
    performance. O resultado e um pl.DataFrame com timestamp como coluna
    e colunas derivadas (mes, hora, ano).

    Tempo estimado de primeiro carregamento: <5s para ~4.6M registros.
    Carregamentos subsequentes usam cache do Streamlit.

    Args:
        filepath: Caminho para o arquivo CSV.
        colunas_map: Mapeamento de nomes logicos para colunas do CSV.
        separador: Separador do CSV.
        codificacao: Encoding do CSV.
        decimal: Caractere decimal do CSV.

    Returns:
        pl.DataFrame com colunas normalizadas, timestamp e colunas derivadas.
    """
    inicio = time.time()

    if colunas_map is None:
        colunas_map = COLUNAS_PADRAO

    na_values = ["", " ", "NULL", "-9999", "#N/D", "*****"]

    df = pl.read_csv(
        filepath,
        separator=separador,
        encoding=codificacao if codificacao.lower() in ("utf-8", "utf8") else "lossy_utf8",
        null_values=na_values,
        infer_schema_length=10000,
    )

    existing_cols = df.columns
    col_rename = {}
    colunas_presentes = set(existing_cols)

    for logico, original in colunas_map.items():
        original_norm = _normalizar_nome_coluna(original)
        if original_norm in colunas_presentes:
            col_rename[original_norm] = logico

    if col_rename:
        df = df.rename(col_rename)

    df = df.with_columns(
        pl.col("precipitacao").fill_null(0.0).cast(pl.Float32).alias("precipitacao"),
    )

    for col_coord in ["latitude", "longitude"]:
        if col_coord in df.columns:
            df = df.with_columns(
                pl.col(col_coord).cast(pl.Float32).alias(col_coord)
            )

    hora_col = "hora_str" if "hora_str" in df.columns else "hora" if "hora" in df.columns else None
    data_col = "data_str" if "data_str" in df.columns else "data" if "data" in df.columns else None

    if hora_col and data_col:
        df = df.with_columns(
            pl.col(hora_col).cast(pl.String).str.strip_chars().alias("_hora_raw")
        )

        df = df.with_columns(
            pl.when(pl.col("_hora_raw").str.contains(":"))
            .then(
                pl.when(pl.col("_hora_raw").str.split(":").list.len() == 2)
                .then(pl.col("_hora_raw") + ":00")
                .otherwise(pl.col("_hora_raw"))
            )
            .otherwise(
                pl.col("_hora_raw").str.zfill(4)
                .str.slice(0, 2) + pl.lit(":") +
                pl.col("_hora_raw").str.zfill(4)
                .str.slice(2, 2) + pl.lit(":00")
            )
            .alias("_hora_fmt")
        )

        df = df.with_columns(
            (pl.col(data_col).cast(pl.String) + pl.lit(" ") + pl.col("_hora_fmt"))
            .alias("_timestamp_str")
        )

        df = df.with_columns(
            pl.col("_timestamp_str")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .alias("timestamp")
        )

        df = df.filter(pl.col("timestamp").is_not_null())

        df = df.drop(["_hora_raw", "_hora_fmt", "_timestamp_str"])

    df = df.filter(pl.col("precipitacao").is_not_null())

    df = df.with_columns([
        pl.col("timestamp").dt.month().cast(pl.Int8).alias("mes"),
        pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hora"),
        pl.col("timestamp").dt.year().cast(pl.Int16).alias("ano"),
    ])

    cols_manter = [c for c in [
        "precipitacao", "estacao", "latitude", "longitude",
        "municipio", "estado", "mes", "hora", "ano", "timestamp",
    ] if c in df.columns]

    df = df.select(cols_manter)
    df = df.sort("timestamp")

    elapsed = time.time() - inicio
    n_rows = len(df)
    n_estacoes = df["estacao"].n_unique() if "estacao" in df.columns else 0
    periodo_inicio = df["timestamp"].min() if n_rows > 0 else "N/A"
    periodo_fim = df["timestamp"].max() if n_rows > 0 else "N/A"

    print(f"[loader] Dataset carregado: {n_rows:,} registros, {n_estacoes} estacoes")
    print(f"[loader] Periodo: {periodo_inicio} a {periodo_fim}")
    print(f"[loader] Tempo de carregamento: {elapsed:.1f}s")

    return df