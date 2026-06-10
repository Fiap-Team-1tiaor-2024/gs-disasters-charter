# =============================================================================
# ml_predict.py — Interface de Predição para o Streamlit
# Global Solution 2026.1 — Monitoramento de Eventos Pluviais Extremos em SP
# Autora: Gabriela da Cunha Rocha — RM561041
#
# Funções exportadas para uso no Streamlit do grupo:
#   carregar_modelo()              → carrega modelo treinado (com cache)
#   prever_risco_manual(...)       → predição por formulário
#   prever_lote(df)                → predição em batch sobre DataFrame
#   prever_risco_antecipado(...)   → predição 3h à frente
#   resumo_risco_por_estacao(df)   → agrega por estação para o mapa
#   detectar_anomalia_pontual(...) → verifica se um registro é anômalo
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

from ml.ml_model import (
    ORDEM_CLASSES, LIMIARES,
    engenharia_de_features, gerar_labels, FEATURES_BASE, _slug
)

# ── Estilos visuais por nível (Streamlit) ──────────────────────────────────
ESTILOS = {
    'NORMAL':        {'cor': '#2ecc71', 'hex_bg': '#d5f5e3', 'emoji': '🟢',
                      'descricao': 'Sem risco significativo'},
    'ATENÇÃO':       {'cor': '#f1c40f', 'hex_bg': '#fef9e7', 'emoji': '🟡',
                      'descricao': 'Atenção — precipitação acima do padrão'},
    'ALERTA':        {'cor': '#e67e22', 'hex_bg': '#fdebd0', 'emoji': '🟠',
                      'descricao': 'Alerta — risco moderado de alagamentos/deslizamentos'},
    'ALERTA MÁXIMO': {'cor': '#e74c3c', 'hex_bg': '#fadbd8', 'emoji': '🔴',
                      'descricao': 'Alerta Máximo — risco alto de desastre'},
}

# ── Cache em memória ───────────────────────────────────────────────────────
_CACHE = {'classificador': None, 'preditivo': None, 'anomalia': None}


# =============================================================================
# 1. CARREGAR MODELOS
# =============================================================================

def carregar_modelo(caminho_modelo:  str = 'data/ml/pkl/modelo_risco_climatico.pkl',
                    caminho_encoder: str = 'data/ml/pkl/label_encoder_risco.pkl') -> dict:
    """Carrega o modelo principal com cache. Lança FileNotFoundError se ausente."""
    if _CACHE['classificador'] is not None:
        return _CACHE['classificador']

    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{caminho_modelo}'. "
            "Execute ml_model.treinar_e_salvar() primeiro."
        )
    pacote = joblib.load(caminho_modelo)
    if pacote.get('usa_encoder') and os.path.exists(caminho_encoder):
        pacote['encoder'] = joblib.load(caminho_encoder)

    _CACHE['classificador'] = pacote
    print(f"[ML] Modelo carregado: {pacote['nome_modelo']} "
          f"(F1={pacote['f1']:.4f})")
    return pacote


def carregar_modelo_preditivo(caminho: str = 'data/ml/pkl/modelo_preditivo_3h.pkl') -> dict | None:
    if _CACHE['preditivo'] is not None:
        return _CACHE['preditivo']
    if not os.path.exists(caminho):
        return None
    pacote = joblib.load(caminho)
    _CACHE['preditivo'] = pacote
    return pacote


def carregar_modelo_anomalia(caminho: str = 'data/ml/pkl/modelo_anomalias.pkl'):
    if _CACHE['anomalia'] is not None:
        return _CACHE['anomalia']
    if not os.path.exists(caminho):
        return None
    modelo = joblib.load(caminho)
    _CACHE['anomalia'] = modelo
    return modelo


# =============================================================================
# 2. PREDIÇÃO MANUAL (formulário Streamlit)
# =============================================================================

def prever_risco_manual(
    precipitacao:  float,
    prec_acum_1h:  float,
    prec_acum_3h:  float,
    prec_acum_6h:  float,
    prec_acum_12h: float,
    prec_acum_24h: float,
    prec_acum_48h: float,
    prec_acum_72h: float,
    hora_do_dia:   int   = 12,
    mes:           int   = 2,
) -> dict:
    """
    Prediz o nível de risco para valores informados manualmente.

    Retorna dict com:
        nivel_risco    (str)
        probabilidades (dict classe→prob)
        confianca      (float 0-1)
        estilo         (dict cor/emoji/descrição)
        nivel_regras   (str) — comparação com o sistema de regras original
    """
    pacote = carregar_modelo()

    linha = {
        'precipitacao':        precipitacao,
        'prec_acum_1h':        prec_acum_1h,
        'prec_acum_3h':        prec_acum_3h,
        'prec_acum_6h':        prec_acum_6h,
        'prec_acum_12h':       prec_acum_12h,
        'prec_acum_24h':       prec_acum_24h,
        'prec_acum_48h':       prec_acum_48h,
        'prec_acum_72h':       prec_acum_72h,
        'hora_do_dia':         hora_do_dia,
        'mes':                 mes,
        'estacao_do_ano':      _mes_para_estacao(mes),
        'dia_semana':          0,
        'variacao_1h_3h':      max(0, prec_acum_3h  - prec_acum_1h),
        'variacao_3h_6h':      max(0, prec_acum_6h  - prec_acum_3h),
        'variacao_6h_24h':     max(0, prec_acum_24h - prec_acum_6h),
        'variacao_24h_72h':    max(0, prec_acum_72h - prec_acum_24h),
        'aceleracao_chuva':    max(0, (prec_acum_3h - prec_acum_1h) - (prec_acum_6h - prec_acum_3h)),
        'intensidade_relativa': (prec_acum_1h / prec_acum_24h) if prec_acum_24h > 0 else 0.0,
        'media_movel_6h':      prec_acum_6h / 6,
        'desvio_movel_6h':     0.0,
    }

    resultado = _executar_predicao(linha, pacote)

    # Comparar com sistema de regras
    resultado['nivel_regras'] = _nivel_por_regras(linha)
    resultado['concordancia_regras'] = (resultado['nivel_risco'] == resultado['nivel_regras'])

    return resultado


# =============================================================================
# 3. PREDIÇÃO ANTECIPADA (3h à frente)
# =============================================================================

def prever_risco_antecipado(
    precipitacao:  float,
    prec_acum_1h:  float,
    prec_acum_3h:  float,
    prec_acum_6h:  float,
    prec_acum_12h: float,
    prec_acum_24h: float,
    prec_acum_48h: float,
    prec_acum_72h: float,
    hora_do_dia:   int = 12,
    mes:           int = 2,
    caminho_preditivo: str = 'data/ml/pkl/modelo_preditivo_3h.pkl',
) -> dict | None:
    """
    Prediz o risco que ocorrerá nas próximas 3 horas.
    Retorna None se o modelo preditivo não estiver disponível.
    """
    pacote = carregar_modelo_preditivo(caminho_preditivo)
    if pacote is None:
        return None

    linha = {
        'precipitacao': precipitacao, 'prec_acum_1h': prec_acum_1h,
        'prec_acum_3h': prec_acum_3h, 'prec_acum_6h': prec_acum_6h,
        'prec_acum_12h': prec_acum_12h, 'prec_acum_24h': prec_acum_24h,
        'prec_acum_48h': prec_acum_48h, 'prec_acum_72h': prec_acum_72h,
        'hora_do_dia': hora_do_dia, 'mes': mes,
        'estacao_do_ano': _mes_para_estacao(mes), 'dia_semana': 0,
        'variacao_1h_3h':   max(0, prec_acum_3h  - prec_acum_1h),
        'variacao_3h_6h':   max(0, prec_acum_6h  - prec_acum_3h),
        'variacao_6h_24h':  max(0, prec_acum_24h - prec_acum_6h),
        'variacao_24h_72h': max(0, prec_acum_72h - prec_acum_24h),
        'aceleracao_chuva': max(0, (prec_acum_3h - prec_acum_1h) - (prec_acum_6h - prec_acum_3h)),
        'intensidade_relativa': (prec_acum_1h / prec_acum_24h) if prec_acum_24h > 0 else 0.0,
        'media_movel_6h': prec_acum_6h / 6, 'desvio_movel_6h': 0.0,
    }
    return _executar_predicao(linha, pacote)


# =============================================================================
# 4. DETECÇÃO DE ANOMALIA PONTUAL
# =============================================================================

def detectar_anomalia_pontual(
    prec_acum_24h: float,
    prec_acum_72h: float,
    precipitacao:  float = 0,
    caminho_anomalia: str = 'data/ml/pkl/modelo_anomalias.pkl',
) -> dict:
    """
    Verifica se um conjunto de valores é anômalo em relação ao histórico.
    Útil para exibir aviso no Streamlit quando um evento não tem precedente.
    """
    iso = carregar_modelo_anomalia(caminho_anomalia)
    if iso is None:
        return {'anomalia': False, 'score': None, 'mensagem': 'Modelo de anomalia não disponível'}

    # Construir vetor mínimo compatível com o Isolation Forest treinado
    # (usa as mesmas features que o classificador)
    pacote = carregar_modelo()
    feats  = pacote['features']
    linha  = {f: 0.0 for f in feats}
    linha.update({
        'precipitacao':    precipitacao,
        'prec_acum_1h':    precipitacao,
        'prec_acum_24h':   prec_acum_24h,
        'prec_acum_72h':   prec_acum_72h,
    })
    X = pd.DataFrame([linha])[feats].fillna(0)

    pred  = iso.predict(X)[0]
    score = float(iso.score_samples(X)[0])
    eh_anomalia = (pred == -1)

    return {
        'anomalia': eh_anomalia,
        'score':    round(score, 4),
        'mensagem': (
            '⚠️ Evento anômalo — sem precedente histórico próximo' if eh_anomalia
            else '✅ Evento dentro do padrão histórico'
        )
    }


# =============================================================================
# 5. PREDIÇÃO EM LOTE (DataFrame completo)
# =============================================================================

def prever_lote(df_acumulados: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o modelo sobre um DataFrame completo com acumulados.
    Adiciona colunas: risco_ml, confianca_ml, risco_regras.

    Uso no Streamlit:
        df_pred = prever_lote(dados_com_prec_acumulada)
    """
    pacote = carregar_modelo()
    modelo = pacote['modelo']

    df = engenharia_de_features(df_acumulados)
    feats_ok = [f for f in pacote['features'] if f in df.columns]
    X = df[feats_ok].fillna(0)

    # Predição
    if pacote.get('usa_encoder') and 'encoder' in pacote:
        y_enc  = modelo.predict(X)
        y_pred = pacote['encoder'].inverse_transform(y_enc)
    else:
        y_pred = modelo.predict(X)

    # Confiança
    confianca = (modelo.predict_proba(X).max(axis=1)
                 if hasattr(modelo, 'predict_proba') else np.ones(len(X)))

    df_out = df_acumulados.copy()
    df_out['risco_ml']      = y_pred
    df_out['confianca_ml']  = confianca
    df_out['risco_regras']  = gerar_labels(df, LIMIARES).values

    print(f"[ML] Predição em lote: {len(df_out):,} registros.")
    dist = pd.Series(y_pred).value_counts()
    for c in ORDEM_CLASSES:
        n = dist.get(c, 0)
        print(f"       {c:<15}: {n:>8,}  ({100*n/len(y_pred):.1f}%)")

    return df_out


# =============================================================================
# 6. RESUMO POR ESTAÇÃO (para o mapa Streamlit)
# =============================================================================

def resumo_risco_por_estacao(df_predito: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega predições por estação. Retorna o nível mais severo e confiança média.
    Pronto para colorir marcadores do mapa Folium/Streamlit.
    """
    prioridade = {n: i for i, n in enumerate(ORDEM_CLASSES)}

    resumo = (
        df_predito.groupby('estacao')
        .agg(
            risco_max      =('risco_ml', lambda s: max(s, key=lambda x: prioridade.get(x, 0))),
            confianca_media=('confianca_ml', 'mean'),
            latitude       =('latitude',  'first'),
            longitude      =('longitude', 'first'),
            municipio      =('municipio', 'first') if 'municipio' in df_predito else ('estacao', 'first'),
        )
        .reset_index()
    )

    resumo['cor']        = resumo['risco_max'].map(lambda r: ESTILOS.get(r, {}).get('cor', 'gray'))
    resumo['emoji']      = resumo['risco_max'].map(lambda r: ESTILOS.get(r, {}).get('emoji', '⚪'))
    resumo['descricao']  = resumo['risco_max'].map(lambda r: ESTILOS.get(r, {}).get('descricao', ''))
    return resumo


# =============================================================================
# 7. AUXILIARES
# =============================================================================

def _mes_para_estacao(mes: int) -> int:
    return {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}.get(mes, 0)


def _nivel_por_regras(linha: dict) -> str:
    """Aplica os limiares fixos sobre um registro dict."""
    prioridade = {n: i for i, n in enumerate(ORDEM_CLASSES)}
    nivel = 'NORMAL'
    for lim in LIMIARES:
        col, val, nv = lim['coluna_acumulado'], lim['valor_mm'], lim['nivel_alerta']
        if linha.get(col, 0) > val and prioridade[nv] > prioridade[nivel]:
            nivel = nv
    return nivel


def _executar_predicao(linha: dict, pacote: dict) -> dict:
    modelo  = pacote['modelo']
    features = pacote['features']
    df_in   = pd.DataFrame([linha])
    feats_ok = [f for f in features if f in df_in.columns]
    X = df_in[feats_ok].fillna(0)

    # Predição
    if pacote.get('usa_encoder') and 'encoder' in pacote:
        nivel = pacote['encoder'].inverse_transform(modelo.predict(X))[0]
    else:
        nivel = str(modelo.predict(X)[0])

    # Probabilidades
    probs = {}
    confianca = 1.0
    if hasattr(modelo, 'predict_proba'):
        p_arr = modelo.predict_proba(X)[0]
        if pacote.get('usa_encoder') and 'encoder' in pacote:
            cls_str = list(pacote['encoder'].inverse_transform(range(len(p_arr))))
        else:
            cls_str = list(modelo.classes_)
        probs = {str(c): round(float(p), 4) for c, p in zip(cls_str, p_arr)}
        confianca = float(max(p_arr))

    return {
        'nivel_risco':    nivel,
        'probabilidades': probs,
        'confianca':      round(confianca, 4),
        'estilo':         ESTILOS.get(nivel, {}),
    }


# =============================================================================
# TESTE RÁPIDO
# =============================================================================

if __name__ == "__main__":
    print("\n[Teste] Predição manual — Bertioga 19/02/2023 (evento real)")
    r = prever_risco_manual(
        precipitacao=45, prec_acum_1h=45, prec_acum_3h=85,
        prec_acum_6h=130, prec_acum_12h=200, prec_acum_24h=292.6,
        prec_acum_48h=310, prec_acum_72h=295.4, hora_do_dia=15, mes=2,
    )
    e = r['estilo']
    print(f"\n  {e.get('emoji','')} Nível ML    : {r['nivel_risco']}")
    print(f"  Nível Regras  : {r.get('nivel_regras','')}")
    print(f"  Concordância  : {r.get('concordancia_regras','')}")
    print(f"  Confiança     : {r['confianca']*100:.1f}%")
    print(f"  Probabilidades:")
    for cls, p in r['probabilidades'].items():
        barra = '█' * int(p * 25)
        print(f"    {cls:<15} {p*100:5.1f}%  {barra}")
