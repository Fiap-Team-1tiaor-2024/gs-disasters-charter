import json
import os
from typing import Optional

import pandas as pd


CAMPOS_ESPERADOS = ["temperatura", "umidade", "nivel_chuva", "timestamp"]


def load_sensor_data(filepath: str) -> pd.DataFrame:
    """Le o arquivo de output do ESP32 simulado (JSON) e retorna DataFrame padronizado.

    Contrato esperado do JSON: lista de objetos com campos
    temperatura, umidade, nivel_chuva, timestamp.

    Caso o arquivo nao exista ou esteja malformado, retorna DataFrame vazio
    com aviso (nao quebra a aplicacao).

    Args:
        filepath: Caminho para o arquivo JSON do sensor ESP32.

    Returns:
        DataFrame com colunas: temperatura, umidade, nivel_chuva, timestamp.
        Coluna timestamp como datetime, demais como float.
    """
    colunas_saida = ["temperatura", "umidade", "nivel_chuva", "timestamp"]

    if not os.path.exists(filepath):
        print(f"[sensor_reader] Arquivo nao encontrado: {filepath}")
        return pd.DataFrame(columns=colunas_saida)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[sensor_reader] Erro ao ler JSON: {e}")
        return pd.DataFrame(columns=colunas_saida)

    if isinstance(dados, dict):
        dados = [dados]

    if not isinstance(dados, list) or len(dados) == 0:
        print(f"[sensor_reader] JSON vazio ou formato inesperado")
        return pd.DataFrame(columns=colunas_saida)

    df = pd.DataFrame(dados)

    campos_ausentes = set(CAMPOS_ESPERADOS) - set(df.columns)
    if campos_ausentes:
        print(f"[sensor_reader] Campos ausentes no JSON: {campos_ausentes}")
        for campo in campos_ausentes:
            df[campo] = float("nan")

    for campo in CAMPOS_ESPERADOS:
        if campo not in df.columns:
            df[campo] = float("nan")

    df = df[["timestamp", "temperatura", "umidade", "nivel_chuva"]]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df["umidade"] = pd.to_numeric(df["umidade"], errors="coerce")
    df["nivel_chuva"] = pd.to_numeric(df["nivel_chuva"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print(f"[sensor_reader] Dados do sensor carregados: {len(df)} registros")

    return df


def normalizar_nivel_chuva(df: pd.DataFrame, escala_max: float = 100.0) -> pd.DataFrame:
    """Normaliza a coluna nivel_chuva para o intervalo 0-100%.

    O potenciometro do ESP32 gera valores brutos que podem variar
    de 0 a 4095 (ADC 12-bit) ou 0 a 1023 (ADC 10-bit).
    Esta funcao converte para percentual 0-100.

    Args:
        df: DataFrame com coluna 'nivel_chuva'.
        escala_max: Valor maximo do ADC para normalizacao (padrao: 4095).

    Returns:
        DataFrame com coluna 'nivel_chuva_pct' adicionada (0-100).
    """
    df = df.copy()

    if "nivel_chuva" not in df.columns:
        df["nivel_chuva_pct"] = float("nan")
        return df

    valores = df["nivel_chuva"].dropna()
    if valores.empty:
        df["nivel_chuva_pct"] = float("nan")
        return df

    valor_max = valores.max()

    if valor_max <= 1.0:
        df["nivel_chuva_pct"] = df["nivel_chuva"] * 100
    elif valor_max <= escala_max:
        df["nivel_chuva_pct"] = (df["nivel_chuva"] / escala_max) * 100
    else:
        df["nivel_chuva_pct"] = (df["nivel_chuva"] / valor_max) * 100

    df["nivel_chuva_pct"] = df["nivel_chuva_pct"].clip(0, 100)

    return df


def ultima_leitura(df: pd.DataFrame) -> Optional[pd.Series]:
    """Retorna a ultima leitura do sensor.

    Args:
        df: DataFrame com dados do sensor.

    Returns:
        Series com a ultima leitura, ou None se DataFrame vazio.
    """
    if df.empty:
        return None
    return df.iloc[-1]