# =============================================================================
# ml_integracao_colab.py — Integração com o Notebook Original
# Global Solution 2026.1
# Autora: Gabriela da Cunha Rocha — RM561041
#
# Cole este arquivo como células ao final do notebook original,
# logo após a Etapa D (geração de alertas).
#
# Pré-requisito: subir ml_model.py e ml_predict.py na mesma pasta do notebook.
# =============================================================================

# ── CÉLULA 0: Instalar dependências (rode uma vez) ──────────────────────────
# !pip install xgboost shap fpdf2 --quiet

# ── CÉLULA 1: Importar módulos ───────────────────────────────────────────────
import sys, os
sys.path.insert(0, DRIVE_BASE_PATH)   # variável já definida no notebook original

from ml_model import treinar_e_salvar, gerar_labels, LIMIARES, ORDEM_CLASSES
from ml_predict import (
    carregar_modelo, prever_risco_manual, prever_risco_antecipado,
    prever_lote, resumo_risco_por_estacao, detectar_anomalia_pontual, ESTILOS,
)

PASTA_ML = os.path.join(CAMINHO_PARA_SALVAR_OUTPUTS, "ml_outputs")

# =============================================================================
# BLOCO A — TREINAMENTO COMPLETO
# Roda após a Etapa D do pipeline original.
# 'dados_com_prec_acumulada' é a variável gerada na Etapa C.
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-A — Treinamento do Modelo de Classificação de Risco")
print("="*65)

pacote_modelo = treinar_e_salvar(
    df_acumulados = dados_com_prec_acumulada,
    caminho_modelo  = os.path.join(DRIVE_BASE_PATH, "modelo_risco_climatico.pkl"),
    caminho_encoder = os.path.join(DRIVE_BASE_PATH, "label_encoder_risco.pkl"),
    caminho_outputs = PASTA_ML,
    amostra_max = 500_000,
)
# Ao final deste bloco serão gerados em PASTA_ML:
#   • comparativo_modelos.png
#   • matriz_confusao_*.png
#   • roc_*.png  /  precision_recall_*.png
#   • importancia_*.png
#   • shap_summary_*.png
#   • anomalias_distribuicao.png  /  anomalias_serie_temporal.png
#   • risco_por_municipio.png  /  risco_sazonal.png
#   • modelo_anomalias.pkl  /  modelo_preditivo_3h.pkl
#   • relatorio_ml_gs2026.pdf        ← pronto para o PDF da entrega


# =============================================================================
# BLOCO B — PREDIÇÃO EM LOTE
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-B — Predição de Risco em Lote")
print("="*65)

df_predito = prever_lote(dados_com_prec_acumulada)
print("\nAmostra dos resultados:")
cols_exibir = ['estacao', 'prec_acum_24h', 'risco_ml', 'risco_regras', 'confianca_ml']
cols_exibir = [c for c in cols_exibir if c in df_predito.columns]
print(df_predito[cols_exibir].head(10).to_string())


# =============================================================================
# BLOCO C — RESUMO POR ESTAÇÃO (para o mapa do Streamlit)
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-C — Resumo de Risco por Estação")
print("="*65)

resumo = resumo_risco_por_estacao(df_predito)
print(resumo[['estacao', 'risco_max', 'confianca_media', 'emoji']].to_string(index=False))


# =============================================================================
# BLOCO D — PREDIÇÃO MANUAL
# Simula o formulário de entrada do Streamlit.
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-D — Predição Manual (Bertioga 19/02/2023 — evento real)")
print("="*65)

resultado = prever_risco_manual(
    precipitacao  = 45.0,
    prec_acum_1h  = 45.0,
    prec_acum_3h  = 85.0,
    prec_acum_6h  = 130.0,
    prec_acum_12h = 200.0,
    prec_acum_24h = 292.6,  # valor real registrado em Bertioga
    prec_acum_48h = 310.0,
    prec_acum_72h = 295.4,  # valor real do acumulado 72h
    hora_do_dia   = 15,
    mes           = 2,
)
estilo = resultado['estilo']
print(f"\n  {estilo.get('emoji','')} Nível ML     : {resultado['nivel_risco']}")
print(f"  Nível Regras  : {resultado.get('nivel_regras','')}")
print(f"  Concordância  : {resultado.get('concordancia_regras','')}")
print(f"  Confiança     : {resultado['confianca']*100:.1f}%")
print(f"  Descrição     : {estilo.get('descricao','')}")
print(f"\n  Probabilidades por classe:")
for cls in ORDEM_CLASSES:
    p = resultado['probabilidades'].get(cls, 0)
    barra = '█' * int(p * 30)
    print(f"    {cls:<15} {p*100:5.1f}%  {barra}")


# =============================================================================
# BLOCO E — PREDIÇÃO ANTECIPADA (3h à frente)
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-E — Predição Antecipada (3h à Frente)")
print("="*65)

resultado_pred = prever_risco_antecipado(
    precipitacao  = 20.0,
    prec_acum_1h  = 20.0,
    prec_acum_3h  = 40.0,
    prec_acum_6h  = 60.0,
    prec_acum_12h = 80.0,
    prec_acum_24h = 95.0,   # ainda abaixo do limiar de 100mm
    prec_acum_48h = 110.0,
    prec_acum_72h = 130.0,
    hora_do_dia   = 12,
    mes           = 2,
    caminho_preditivo = os.path.join(PASTA_ML, 'modelo_preditivo_3h.pkl'),
)
if resultado_pred:
    e = resultado_pred['estilo']
    print(f"\n  Risco previsto em 3h: {e.get('emoji','')} {resultado_pred['nivel_risco']}")
    print(f"  Confiança: {resultado_pred['confianca']*100:.1f}%")
    print(f"  {e.get('descricao','')}")
else:
    print("  Modelo preditivo não encontrado.")


# =============================================================================
# BLOCO F — DETECÇÃO DE ANOMALIA PONTUAL
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-F — Detecção de Anomalia Pontual")
print("="*65)

anom_evento_real = detectar_anomalia_pontual(
    prec_acum_24h = 292.6,
    prec_acum_72h = 295.4,
    precipitacao  = 45.0,
    caminho_anomalia = os.path.join(PASTA_ML, 'modelo_anomalias.pkl'),
)
print(f"\n  Bertioga 19/02/2023: {anom_evento_real['mensagem']}")
print(f"  Score de anomalia   : {anom_evento_real['score']}")

anom_normal = detectar_anomalia_pontual(
    prec_acum_24h = 15.0,
    prec_acum_72h = 30.0,
    precipitacao  = 2.0,
    caminho_anomalia = os.path.join(PASTA_ML, 'modelo_anomalias.pkl'),
)
print(f"\n  Dia comum (15mm/24h): {anom_normal['mensagem']}")
print(f"  Score de anomalia   : {anom_normal['score']}")


# =============================================================================
# BLOCO G — COMPARAÇÃO REGRAS vs ML no evento de São Sebastião
# =============================================================================
print("\n" + "="*65)
print("ETAPA ML-G — Validação: Regras vs ML (Bertioga fev/2023)")
print("="*65)

periodo = df_predito[
    (df_predito['estacao'] == 'Bertioga') &
    (df_predito.index >= '2023-02-17') &
    (df_predito.index <= '2023-02-22')
].copy()

if not periodo.empty:
    concordancia = (periodo['risco_ml'] == periodo['risco_regras']).mean()
    comp = periodo[['prec_acum_24h', 'risco_regras', 'risco_ml', 'confianca_ml']].copy()
    comp.index = comp.index.strftime('%d/%m %Hh')
    print(comp.to_string())
    print(f"\n  Concordância Regras vs ML : {concordancia*100:.1f}%")
else:
    print("  Dados de Bertioga não encontrados no período.")

print("\n✅ Módulo ML integrado com sucesso.")
print(f"   Relatório PDF disponível em: {PASTA_ML}/relatorio_ml_gs2026.pdf")
