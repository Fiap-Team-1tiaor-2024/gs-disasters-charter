import re
import unicodedata
from datetime import datetime, date
from typing import Optional

import pandas as pd


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
    """Normaliza nome de coluna para snake_case sem acentos."""
    col = unicodedata.normalize("NFKD", col)
    col = col.encode("ASCII", "ignore").decode("ASCII")
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9_]", "_", col)
    col = re.sub(r"_+", "_", col)
    return col


def _formatar_hora(hora_val) -> str:
    """Converte valor de hora para string formatada HH:MM:SS."""
    hora_str = str(int(hora_val)) if isinstance(hora_val, (int, float)) else str(hora_val)
    hora_str = hora_str.strip()

    if ":" in hora_str:
        partes = hora_str.split(":")
        if len(partes) == 2:
            return f"{partes[0].zfill(2)}:{partes[1].zfill(2)}:00"
        return hora_str

    hora_str = hora_str.zfill(4)
    if len(hora_str) == 4:
        return f"{hora_str[:2]}:{hora_str[2:]}:00"

    return hora_str


def load_and_preprocess(
    filepath: str,
    colunas_map: Optional[dict] = None,
    separador: str = ",",
    codificacao: str = "utf-8",
    decimal: str = ".",
) -> pd.DataFrame:
    """Carrega o CSV do INMET e realiza o pré-processamento completo.

    Args:
        filepath: Caminho para o arquivo CSV.
        colunas_map: Mapeamento de nomes lógicos para colunas do CSV.
        separador: Separador do CSV.
        codificacao: Encoding do CSV.
        decimal: Caractere decimal do CSV.

    Returns:
        DataFrame com timestamp como índice e colunas normalizadas.
    """
    if colunas_map is None:
        colunas_map = COLUNAS_PADRAO

    na_values = ["", " ", "NULL", "-9999", "#N/D", "*****"]

    df = pd.read_csv(
        filepath,
        sep=separador,
        encoding=codificacao,
        decimal=decimal,
        na_values=na_values,
    )

    df.columns = [_normalizar_nome_coluna(c) for c in df.columns]

    col_rename = {}
    colunas_presentes = set(df.columns)
    for logico, original in colunas_map.items():
        original_norm = _normalizar_nome_coluna(original)
        if original_norm in colunas_presentes:
            col_rename[original_norm] = logico

    df = df.rename(columns=col_rename)

    if "precipitacao" in df.columns:
        df["precipitacao"] = pd.to_numeric(df["precipitacao"], errors="coerce").fillna(0)

    for col_coord in ["latitude", "longitude"]:
        if col_coord in df.columns:
            if df[col_coord].dtype == object:
                df[col_coord] = df[col_coord].str.replace(",", ".").astype(float)
            else:
                df[col_coord] = pd.to_numeric(df[col_coord], errors="coerce")

    timestamps = []
    for _, row in df.iterrows():
        if "data_str" in df.columns and "hora_str" in df.columns:
            hora_fmt = _formatar_hora(row["hora_str"])
            ts_str = f"{row['data_str']} {hora_fmt}"
            try:
                timestamps.append(pd.Timestamp(ts_str))
            except (ValueError, TypeError):
                timestamps.append(pd.NaT)
        else:
            timestamps.append(pd.NaT)

    df["timestamp"] = timestamps
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    if "precipitacao" in df.columns:
        df = df.dropna(subset=["precipitacao"])

    df["mes"] = df.index.month
    df["hora"] = df.index.hour
    df["ano"] = df.index.year

    n_estacoes = df["estacao"].nunique() if "estacao" in df.columns else 0
    periodo = f"{df.index.min()} a {df.index.max()}" if len(df) > 0 else "N/A"

    print(f"[loader] Dataset carregado: {len(df):,} registros, {n_estacoes} estações")
    print(f"[loader] Período: {periodo}")

    return df