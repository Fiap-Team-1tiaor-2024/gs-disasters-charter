import os
from typing import Any, Optional

import pandas as pd
import polars as pl
import streamlit as st


MODELO_PADRAO = "data/ml/pkl/modelo_risco_climatico.pkl"
ENCODER_PADRAO = "data/ml/pkl/label_encoder_risco.pkl"
MODELO_PRED_PDR = "data/ml/pkl/modelo_preditivo_3h.pkl"
MODELO_ANOM_PDR = "data/ml/pkl/modelo_anomalias.pkl"


@st.cache_resource
def get_model_package() -> Optional[dict]:
    """Carrega o pacote do modelo ML com cache do Streamlit.

    Wrapper de ml_predict.carregar_modelo() que retorna None
    graciosamente quando os arquivos .pkl nao existem, em vez de
    lancar FileNotFoundError.

    Returns:
        dict com modelo, features, metricas etc., ou None se ausente.
    """
    try:
        from ml.ml_predict import carregar_modelo
        pacote = carregar_modelo(
            caminho_modelo=MODELO_PADRAO,
            caminho_encoder=ENCODER_PADRAO,
        )
        print(f"[ml] Modelo carregado: {pacote.get('nome_modelo', 'N/A')}")
        return pacote
    except FileNotFoundError:
        print("[ml] Modelos .pkl nao encontrados — modo stub ativo")
        return None
    except Exception as e:
        print(f"[ml] Erro ao carregar modelo: {e}")
        return None


def preparar_df_para_ml(df) -> pd.DataFrame:
    """Converte DataFrame do pipeline para pd.DataFrame com DatetimeIndex.

    Renomeia colunas precip_acc_Nh -> prec_acum_Nh (formato esperado
    pelo ml_model) e seta data_hora como indice, conforme exigido
    por engenharia_de_features().

    Args:
        df: pl.DataFrame ou pd.DataFrame do pipeline com colunas de
            acumulado e timestamp.

    Returns:
        pd.DataFrame com DatetimeIndex nomeado 'data_hora' e colunas
        renomeadas para o formato esperado por ml_model.
    """
    import polars as pl

    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        pdf = df.copy()
    else:
        raise TypeError(f"Esperado pl.DataFrame ou pd.DataFrame, recebido {type(df)}")

    rename_map = {}
    for n in [1, 3, 6, 12, 24, 48, 72]:
        old = f"precip_acc_{n}h"
        new = f"prec_acum_{n}h"
        if old in pdf.columns:
            rename_map[old] = new

    pdf = pdf.rename(columns=rename_map)

    if "timestamp" in pdf.columns:
        pdf["data_hora"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("data_hora")
        pdf = pdf.sort_index()

    return pdf


def run_batch_prediction(df: pl.DataFrame) -> Optional[pd.DataFrame]:
    """Executa predicao em lote sobre o DataFrame do pipeline.

    Converte o pl.DataFrame para pandas, aplica engenharia de features
    e chama prever_lote() do ml_predict.

    Args:
        df: pl.DataFrame do pipeline com acumulados calculados.

    Returns:
        pd.DataFrame com colunas risco_ml, confianca_ml, risco_regras,
        ou None se o modelo nao estiver disponivel.
    """
    pacote = get_model_package()
    if pacote is None:
        return None

    from ml.ml_predict import prever_lote

    pdf = preparar_df_para_ml(df)

    try:
        df_predito = prever_lote(pdf)
        print(f"[ml] Predicao em lote: {len(df_predito):,} registros")
        return df_predito
    except Exception as e:
        print(f"[ml] Erro na predicao em lote: {e}")
        return None


def get_station_summary(df_predito: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Agrega predicoes por estacao para uso no mapa.

    Wrapper direto de ml_predict.resumo_risco_por_estacao().

    Args:
        df_predito: DataFrame retornado por run_batch_prediction().

    Returns:
        pd.DataFrame com resumo por estacao, ou None se erro.
    """
    if df_predito is None or df_predito.empty:
        return None

    try:
        from ml.ml_predict import resumo_risco_por_estacao
        return resumo_risco_por_estacao(df_predito)
    except Exception as e:
        print(f"[ml] Erro ao gerar resumo: {e}")
        return None


def get_model_info() -> dict:
    """Retorna informacoes sobre o modelo ML carregado.

    Returns:
        dict com carregado, nome_modelo, acuracia, f1, roc_auc, etc.
    """
    pacote = get_model_package()

    if pacote is None:
        return {
            "carregado": False,
            "caminho": MODELO_PADRAO,
            "tipo": "N/A",
            "status": "Modelo não carregado — modo stub ativo. Copie os .pkl para data/ml/pkl/",
            "versao": "N/A",
            "metricas": {},
        }

    return {
        "carregado": True,
        "caminho": MODELO_PADRAO,
        "tipo": pacote.get("nome_modelo", "N/A"),
        "status": "Modelo carregado e pronto para predição",
        "features": pacote.get("features", []),
        "n_features": len(pacote.get("features", [])),
        "classes": pacote.get("ordem_classes", []),
        "acuracia": pacote.get("acuracia"),
        "f1": pacote.get("f1"),
        "roc_auc": pacote.get("roc_auc"),
        "metricas": {
            "acuracia": pacote.get("acuracia"),
            "f1": pacote.get("f1"),
            "roc_auc": pacote.get("roc_auc"),
        },
    }


def get_feature_importance() -> Optional[pd.DataFrame]:
    """Extrai feature importance do modelo ML, se disponivel.

    Returns:
        pd.DataFrame com colunas 'feature' e 'importance', ou None.
    """
    pacote = get_model_package()
    if pacote is None:
        return None

    modelo = pacote.get("modelo")
    if modelo is None:
        return None

    feature_names = None
    importances = None

    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importances = abs(modelo.coef_).flatten() if modelo.coef_.ndim > 1 else abs(modelo.coef_)

    if hasattr(modelo, "feature_names_in_"):
        feature_names = list(modelo.feature_names_in_)
    elif "features" in pacote:
        feature_names = pacote["features"]

    if importances is None:
        return None

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df_importance = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances[:len(feature_names)],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df_importance