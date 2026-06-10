import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.alerts import calculate_irc, calculate_historical_percentiles
from pipeline.utils import to_pandas
from ml.model_interface import (
    get_model_package,
    run_batch_prediction,
    get_station_summary,
    get_model_info,
    get_feature_importance,
)

DATA_PATH = os.environ.get("DATA_PATH", "data/dataset/inmet_sp.csv")

ML_IMAGES_DIR = "data/ml/images"
ML_PDF_PATH = "data/docs/relatorio_ml_gs2026.pdf"


def _load_pipeline():
    df_pl = load_and_preprocess(DATA_PATH)
    df_pl = calculate_accumulations(df_pl)
    pct_pl = calculate_historical_percentiles(df_pl)
    df_pl = calculate_irc(df_pl, percentiles=pct_pl)
    df = to_pandas(df_pl)
    pct = pct_pl.to_pandas()
    return df, pct


st.header("🤖 Modelo ML")
st.caption("Predição de risco com XGBoost/Random Forest, detecção de anomalias e predição antecipada.")

pacote = get_model_package()
model_info = get_model_info()

# ── Seção 1 — Status do modelo ──────────────────────────────────────────────
st.subheader("1. Status do Modelo")

if not model_info["carregado"]:
    st.warning(
        "Modelos não encontrados em `data/ml/pkl/`. "
        "Copie os arquivos `.pkl` gerados pelo treinamento para `data/ml/pkl/` "
        "e as imagens para `data/ml/images/`."
    )
    st.json({
        "caminho_modelo": model_info["caminho"],
        "status": model_info["status"],
    })
    st.stop()

st.success("Modelo carregado e pronto para predição")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Modelo", value=model_info.get("tipo", "N/A"))

with col2:
    f1_val = model_info.get("metricas", {}).get("f1")
    st.metric("F1 (pond.)", value=f"{f1_val:.4f}" if f1_val else "N/A")

with col3:
    roc_val = model_info.get("metricas", {}).get("roc_auc")
    st.metric("ROC-AUC", value=f"{roc_val:.4f}" if roc_val else "N/A")

if model_info.get("classes"):
    st.write(f"**Classes:** {', '.join(model_info['classes'])}")
if model_info.get("n_features"):
    st.write(f"**Features:** {model_info['n_features']}")

st.divider()

# ── Seção 2 — Predição manual ────────────────────────────────────────────────
st.subheader("2. Predição Manual")

with st.expander("Formulário de predição manual", expanded=True):
    from ml.ml_predict import prever_risco_manual, detectar_anomalia_pontual

    c1, c2, c3 = st.columns(3)
    with c1:
        precipitacao = st.number_input("Precipitação (mm)", min_value=0.0, value=10.0, step=1.0)
        prec_acum_1h = st.number_input("Acumulado 1h (mm)", min_value=0.0, value=10.0, step=1.0)
        prec_acum_3h = st.number_input("Acumulado 3h (mm)", min_value=0.0, value=15.0, step=1.0)
    with c2:
        prec_acum_6h = st.number_input("Acumulado 6h (mm)", min_value=0.0, value=20.0, step=1.0)
        prec_acum_12h = st.number_input("Acumulado 12h (mm)", min_value=0.0, value=30.0, step=1.0)
        prec_acum_24h = st.number_input("Acumulado 24h (mm)", min_value=0.0, value=40.0, step=1.0)
    with c3:
        prec_acum_48h = st.number_input("Acumulado 48h (mm)", min_value=0.0, value=50.0, step=1.0)
        prec_acum_72h = st.number_input("Acumulado 72h (mm)", min_value=0.0, value=60.0, step=1.0)
        hora_do_dia = st.number_input("Hora do dia", min_value=0, max_value=23, value=12)
        mes = st.number_input("Mês", min_value=1, max_value=12, value=2)

    if st.button("Executar Predição Manual"):
        try:
            resultado = prever_risco_manual(
                precipitacao=precipitacao,
                prec_acum_1h=prec_acum_1h,
                prec_acum_3h=prec_acum_3h,
                prec_acum_6h=prec_acum_6h,
                prec_acum_12h=prec_acum_12h,
                prec_acum_24h=prec_acum_24h,
                prec_acum_48h=prec_acum_48h,
                prec_acum_72h=prec_acum_72h,
                hora_do_dia=hora_do_dia,
                mes=mes,
            )

            estilo = resultado.get("estilo", {})
            emoji = estilo.get("emoji", "")
            st.subheader(f"{emoji} Nível de Risco ML: {resultado['nivel_risco']}")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Nível ML", value=f"{emoji} {resultado['nivel_risco']}")
            with col_b:
                st.metric("Confiança", value=f"{resultado['confianca']*100:.1f}%")
            with col_c:
                concorda = resultado.get("concordancia_regras", None)
                if concorda is not None:
                    st.metric("Concorda com Regras", value="Sim" if concorda else "Não")

            if resultado.get("nivel_regras"):
                st.info(f"Nível do sistema de regras (IRC): **{resultado['nivel_regras']}**")

            probs = resultado.get("probabilidades", {})
            if probs:
                st.subheader("Probabilidades por Classe")
                for cls, prob in probs.items():
                    st.progress(prob, text=f"{cls}: {prob*100:.1f}%")

            anom = detectar_anomalia_pontual(
                prec_acum_24h=prec_acum_24h,
                prec_acum_72h=prec_acum_72h,
                precipitacao=precipitacao,
            )
            if anom.get("anomalia"):
                st.warning(f"Detecção de anomalia: {anom['mensagem']} (score: {anom['score']})")
            else:
                st.info(f"Detecção de anomalia: {anom['mensagem']}")

        except FileNotFoundError:
            st.warning("Modelos .pkl não encontrados. A predição manual requer os arquivos treinados.")
        except Exception as e:
            st.error(f"Erro na predição manual: {e}")

st.divider()

# ── Seção 3 — Predição antecipada (3h) ──────────────────────────────────────
st.subheader("3. Predição Antecipada (3h)")

try:
    from ml.ml_predict import prever_risco_antecipado, ESTILOS

    caminho_pred = os.environ.get("ML_PRED_PATH", "data/ml/pkl/modelo_preditivo_3h.pkl")

    if os.path.exists(caminho_pred):
        with st.expander("Predição antecipada 3h", expanded=False):
            st.write("Usando os mesmos valores do formulário acima para prever o risco nas próximas 3 horas.")

            if st.button("Prever Risco 3h à Frente"):
                try:
                    resultado_pred = prever_risco_antecipado(
                        precipitacao=precipitacao,
                        prec_acum_1h=prec_acum_1h,
                        prec_acum_3h=prec_acum_3h,
                        prec_acum_6h=prec_acum_6h,
                        prec_acum_12h=prec_acum_12h,
                        prec_acum_24h=prec_acum_24h,
                        prec_acum_48h=prec_acum_48h,
                        prec_acum_72h=prec_acum_72h,
                        hora_do_dia=hora_do_dia,
                        mes=mes,
                        caminho_preditivo=caminho_pred,
                    )
                    if resultado_pred:
                        emoji = resultado_pred.get("estilo", {}).get("emoji", "")
                        st.metric(
                            "Risco previsto em 3h",
                            value=f"{emoji} {resultado_pred['nivel_risco']}",
                        )
                        st.metric("Confiança", value=f"{resultado_pred['confianca']*100:.1f}%")
                    else:
                        st.info("Modelo preditivo 3h não disponível.")
                except Exception as e:
                    st.error(f"Erro na predição antecipada: {e}")
    else:
        st.info("Modelo preditivo 3h não disponível. Coloque o arquivo em `data/ml/pkl/modelo_preditivo_3h.pkl`.")
except FileNotFoundError:
    st.info("Modelo preditivo 3h não disponível.")

st.divider()

# ── Seção 4 — Visualizações do treinamento ──────────────────────────────────
st.subheader("4. Visualizações do Treinamento")

imagens_expect = [
    ("comparativo_modelos.png", "Comparativo de Modelos"),
    ("importancia_random_forest.png", "Importância das Features (Random Forest)"),
    ("shap_summary_random_forest.png", "SHAP Summary (Random Forest)"),
    ("matriz_confusao_random_forest.png", "Matriz de Confusão (Random Forest)"),
    ("matriz_confusao_xgboost.png", "Matriz de Confusão (XGBoost)"),
    ("roc_random_forest.png", "Curva ROC (Random Forest)"),
    ("roc_xgboost.png", "Curva ROC (XGBoost)"),
    ("precision_recall_random_forest.png", "Precision-Recall (Random Forest)"),
    ("precision_recall_xgboost.png", "Precision-Recall (XGBoost)"),
    ("anomalias_distribuicao.png", "Distribuição de Anomalias"),
    ("anomalias_serie_temporal.png", "Série Temporal de Anomalias"),
    ("matriz_confusao_pred_3h_ahead.png", "Matriz de Confusão — Predição 3h"),
    ("risco_por_municipio.png", "Risco por Município"),
    ("risco_sazonal.png", "Risco Sazonal"),
]

imagens_encontradas = []
for nome_arquivo, titulo in imagens_expect:
    caminho = os.path.join(ML_IMAGES_DIR, nome_arquivo)
    if os.path.exists(caminho):
        imagens_encontradas.append((caminho, titulo))

if imagens_encontradas:
    for caminho, titulo in imagens_encontradas:
        st.subheader(titulo)
        st.image(caminho, use_container_width=True)
else:
    st.info("Nenhuma imagem de treinamento encontrada em `data/ml/images/`. Execute o treinamento para gerar.")

if os.path.exists(ML_PDF_PATH):
    with open(ML_PDF_PATH, "rb") as f:
        st.download_button(
            label="📊 Baixar Relatório PDF",
            data=f.read(),
            file_name="relatorio_ml_gs2026.pdf",
            mime="application/pdf",
        )
else:
    st.info("Relatório PDF não encontrado em `data/docs/relatorio_ml_gs2026.pdf`.")

st.divider()

# ── Seção 5 — Nota sobre os dois sistemas ────────────────────────────────────
st.subheader("5. Dois Sistemas de Classificação")

st.info(
    "Este sistema usa **dois mecanismos paralelos** de classificação de risco:\n\n"
    "- **IRC (Índice de Risco Composto)**: baseado em percentis históricos, "
    "classifica em 5 níveis (Normal, Atenção, Alto, Muito Alto, Crítico)\n\n"
    "- **Modelo ML (XGBoost/Random Forest)**: treinado sobre os dados históricos, "
    "classifica em 4 níveis (NORMAL, ATENÇÃO, ALERTA, ALERTA MÁXIMO)\n\n"
    "Os níveis não são diretamente comparáveis, mas se complementam: "
    "o IRC oferece uma visão histórica adaptativa, enquanto o ML identifica "
    "padrões não-lineares e prediz risco 3 horas à frente."
)

# ── Seção extra — Importance (se modelo carregado) ───────────────────────────
importance_df = get_feature_importance()
if importance_df is not None and not importance_df.empty:
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.set_xlabel("Importância")
    ax.set_title("Feature Importance do Modelo")
    ax.invert_yaxis()
    st.pyplot(fig)