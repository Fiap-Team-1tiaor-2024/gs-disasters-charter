import os
from typing import Any, Optional

import pandas as pd
import polars as pl


def load_model(model_path: str) -> Optional[Any]:
    """Carrega um modelo de ML a partir de arquivo (joblib/pickle).

    Suporta modelos serializados com joblib ou pickle.
    Retorna None se o modelo nao estiver disponivel, sem quebrar a aplicacao.

    Args:
        model_path: Caminho para o arquivo do modelo (.pkl, .joblib, .pickle).

    Returns:
        Objeto do modelo carregado, ou None se o arquivo nao existir.
    """
    if not model_path or not os.path.exists(model_path):
        print(f"[ml] Modelo nao encontrado em: {model_path}")
        return None

    ext = os.path.splitext(model_path)[1].lower()

    try:
        if ext in (".joblib", ".jl"):
            import joblib
            model = joblib.load(model_path)
            print(f"[ml] Modelo carregado via joblib: {model_path}")
            return model
        elif ext in (".pkl", ".pickle", ".pk"):
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"[ml] Modelo carregado via pickle: {model_path}")
            return model
        else:
            print(f"[ml] Extensao nao suportada: {ext}. Tentando joblib...")
            import joblib
            model = joblib.load(model_path)
            print(f"[ml] Modelo carregado via joblib (fallback): {model_path}")
            return model
    except Exception as e:
        print(f"[ml] Erro ao carregar modelo: {e}")
        return None


def _to_pandas(df):
    """Converte pl.DataFrame ou pd.DataFrame para pd.DataFrame."""
    if isinstance(df, pl.DataFrame):
        if "timestamp" in df.columns:
            return df.to_pandas().set_index("timestamp").sort_index()
        return df.to_pandas()
    return df


def build_features(df, df_sensor=None):
    """Prepara features a partir do pipeline para o modelo de ML.

    Aceita pl.DataFrame ou pd.DataFrame e converte internamente para pandas,
    pois modelos sklearn esperam pd.DataFrame como entrada.

    Features geradas:
    - Acumulados de precipitacao (1h, 6h, 24h, 72h)
    - Scores de risco por janela (se disponiveis)
    - Percentis do mes atual para cada janela
    - IRC calculado (se disponivel)
    - Dados do sensor (se disponiveis): temperatura, umidade, nivel_chuva

    Args:
        df: DataFrame (pl.DataFrame ou pd.DataFrame) com acumulados, scores e IRC.
        df_sensor: DataFrame do sensor ESP32 (opcional).

    Returns:
        pd.DataFrame com features prontas para o modelo.
    """
    df_pd = _to_pandas(df)

    feature_cols = []

    colunas_acum = ["precip_acc_1h", "precip_acc_6h", "precip_acc_24h", "precip_acc_72h"]
    for col in colunas_acum:
        if col in df_pd.columns:
            feature_cols.append(col)

    score_cols = ["score_1h", "score_6h", "score_24h", "score_72h"]
    for col in score_cols:
        if col in df_pd.columns:
            feature_cols.append(col)

    if "irc" in df_pd.columns:
        feature_cols.append("irc")

    if "mes" in df_pd.columns:
        feature_cols.append("mes")
    if "hora" in df_pd.columns:
        feature_cols.append("hora")

    if "precipitacao" in df_pd.columns:
        feature_cols.append("precipitacao")

    features = df_pd[feature_cols].copy()

    if df_sensor is not None and not df_sensor.empty:
        df_sensor_pd = _to_pandas(df_sensor) if isinstance(df_sensor, pl.DataFrame) else df_sensor
        sensor_features = {
            "sensor_temperatura": df_sensor_pd["temperatura"].mean(),
            "sensor_umidade": df_sensor_pd["umidade"].mean(),
            "sensor_nivel_chuva_pct": df_sensor_pd["nivel_chuva_pct"].mean()
            if "nivel_chuva_pct" in df_sensor_pd.columns
            else df_sensor_pd["nivel_chuva"].mean(),
        }
        for col, val in sensor_features.items():
            features[col] = val

    return features


def predict_risk(
    model: Optional[Any],
    features: pd.DataFrame,
    df_original=None,
) -> pd.DataFrame:
    """Executa predicao de risco usando o modelo de ML.

    Se o modelo nao estiver disponivel (None), retorna DataFrame com
    aviso "Modelo nao carregado" em modo stub.

    Args:
        model: Modelo de ML carregado (ou None para stub).
        features: DataFrame com features preparadas por build_features.
        df_original: DataFrame original com colunas estacao e timestamp
            para enriquecer a saida. Aceita pl.DataFrame ou pd.DataFrame.

    Returns:
        pd.DataFrame com colunas: estacao, timestamp, risco_predito, probabilidade.
    """
    resultado_vazio = pd.DataFrame(
        columns=["estacao", "timestamp", "risco_predito", "probabilidade"]
    )

    if df_original is not None:
        df_original_pd = _to_pandas(df_original)
    else:
        df_original_pd = None

    if model is None:
        print("[ml] Modelo nao carregado — retornando predicoes stub")

        if df_original_pd is not None and "irc" in df_original_pd.columns:
            df_original_str = df_original_pd.copy()
            df_original_str["nivel_risco"] = df_original_str["nivel_risco"].astype(str) if "nivel_risco" in df_original_str.columns else "Normal"

            ranking = (
                df_original_str.groupby("estacao", observed=True)
                .agg({"irc": "max"})
                .reset_index()
                .sort_values("irc", ascending=False)
                .head(20)
            )

            ranking["risco_predito"] = "Modelo nao carregado"
            ranking["probabilidade"] = float("nan")
            ranking["timestamp"] = df_original_pd.index.max() if isinstance(df_original_pd.index, pd.DatetimeIndex) else pd.NaT

            return ranking[["estacao", "timestamp", "risco_predito", "probabilidade"]]

        return resultado_vazio

    try:
        predictions = model.predict(features)
        resultado = pd.DataFrame()
        resultado["risco_predito"] = predictions

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features)
            resultado["probabilidade"] = probas.max(axis=1)
        else:
            resultado["probabilidade"] = float("nan")

        if df_original_pd is not None:
            resultado["estacao"] = df_original_pd["estacao"].values[:len(resultado)]
            resultado["timestamp"] = df_original_pd.index[:len(resultado)]
        else:
            resultado["estacao"] = "Desconhecida"
            resultado["timestamp"] = pd.NaT

        return resultado[["estacao", "timestamp", "risco_predito", "probabilidade"]]

    except Exception as e:
        print(f"[ml] Erro ao executar predicao: {e}")
        return resultado_vazio


def get_model_info(model: Optional[Any], model_path: str = "") -> dict:
    """Retorna informacoes sobre o modelo carregado.

    Args:
        model: Modelo de ML carregado (ou None).
        model_path: Caminho do arquivo do modelo.

    Returns:
        Dicionario com status, versao, tipo e metricas do modelo.
    """
    info = {
        "carregado": model is not None,
        "caminho": model_path,
        "tipo": type(model).__name__ if model is not None else "N/A",
        "versao": "N/A",
        "metricas": {},
    }

    if model is None:
        info["status"] = "Modelo nao carregado — modo stub ativo"
        return info

    info["status"] = "Modelo carregado e pronto para predicao"

    if hasattr(model, "classes_"):
        info["classes"] = list(model.classes_)

    if hasattr(model, "n_features_in_"):
        info["n_features"] = model.n_features_in_

    if hasattr(model, "feature_names_in_"):
        info["feature_names"] = list(model.feature_names_in_)

    return info


def get_feature_importance(model: Optional[Any]) -> Optional[pd.DataFrame]:
    """Extrai feature importance do modelo, se disponivel.

    Args:
        model: Modelo de ML carregado (ou None).

    Returns:
        DataFrame com colunas 'feature' e 'importance' ordenado por importancia,
        ou None se o modelo nao fornecer feature importance.
    """
    if model is None:
        return None

    feature_names = None
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_).flatten() if model.coef_.ndim > 1 else abs(model.coef_)

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if importances is None:
        return None

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df_importance = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances[:len(feature_names)],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df_importance