import re
import time
import unicodedata
from typing import Optional

import pandas as pd
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

DTYPE_OTIMIZADOS = {
    "precipitacao": "float32",
    "latitude": "float32",
    "longitude": "float32",
    "estacao": "category",
    "municipio": "category",
    "estado": "category",
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


def _formatar_hora_series(serie: pd.Series) -> pd.Series:
    """Vetoriza a formatação de hora para toda uma Series."""
    horas = serie.astype(str).str.strip()

    horas_com_dois_pontos = horas.str.contains(":", na=False)
    resultado = horas.copy()

    horas_com_ponto = horas[horas_com_dois_pontos]
    partes = horas_com_ponto.str.split(":", expand=True)
    if partes.shape[1] >= 2:
        resultado[horas_com_dois_pontos & (partes.shape[1] == 2)] = (
            partes[0].str.zfill(2) + ":" + partes[1].str.zfill(2) + ":00"
        )

    horas_sem_ponto = horas[~horas_com_dois_pontos]
    horas_4dig = horas_sem_ponto.str.zfill(4)
    resultado[~horas_com_dois_pontos] = (
        horas_4dig.str[:2] + ":" + horas_4dig.str[2:4] + ":00"
    )

    return resultado


@st.cache_data(show_spinner="Carregando dados meteorologicos...")
def load_and_preprocess(
    filepath: str,
    colunas_map: Optional[dict] = None,
    separador: str = ",",
    codificacao: str = "utf-8",
    decimal: str = ".",
) -> pd.DataFrame:
    """Carrega o CSV do INMET e realiza o preprocessamento completo.

    Tempo estimado de primeiro carregamento: ~15-30s para ~4.6M registros.
    Carregamentos subsequentes usam cache do Streamlit.

    Args:
        filepath: Caminho para o arquivo CSV.
        colunas_map: Mapeamento de nomes logicos para colunas do CSV.
        separador: Separador do CSV.
        codificacao: Encoding do CSV.
        decimal: Caractere decimal do CSV.

    Returns:
        DataFrame com timestamp como indice e colunas normalizadas.
        Colunas derivadas: mes, hora, ano.
    """
    inicio = time.time()

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
                df[col_coord] = df[col_coord].str.replace(",", ".").astype("float32")
            else:
                df[col_coord] = pd.to_numeric(df[col_coord], errors="coerce").astype("float32")

    for col_cat in ["estacao", "municipio", "estado"]:
        if col_cat in df.columns:
            df[col_cat] = df[col_cat].astype("category")

    df["precipitacao"] = df["precipitacao"].astype("float32")

    if "data_str" in df.columns and "hora_str" in df.columns:
        horas_formatadas = _formatar_hora_series(df["hora_str"])
        timestamps_str = df["data_str"].astype(str) + " " + horas_formatadas
        df["timestamp"] = pd.to_datetime(timestamps_str, errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    if "precipitacao" in df.columns:
        df = df.dropna(subset=["precipitacao"])

    df["mes"] = df.index.month.astype("int8")
    df["hora"] = df.index.hour.astype("int8")
    df["ano"] = df.index.year.astype("int16")

    cols_manter = []
    for c in ["precipitacao", "estacao", "latitude", "longitude", "municipio", "estado", "mes", "hora", "ano"]:
        if c in df.columns:
            cols_manter.append(c)
    df = df[cols_manter]

    elapsed = time.time() - inicio
    n_estacoes = df["estacao"].nunique() if "estacao" in df.columns else 0
    periodo_inicio = df.index.min() if len(df) > 0 else "N/A"
    periodo_fim = df.index.max() if len(df) > 0 else "N/A"

    print(f"[loader] Dataset carregado: {len(df):,} registros, {n_estacoes} estacoes")
    print(f"[loader] Periodo: {periodo_inicio} a {periodo_fim}")
    print(f"[loader] Tempo de carregamento: {elapsed:.1f}s")

    return df