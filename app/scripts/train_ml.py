"""Gera os artefatos ML (.pkl, graficos, PDF) a partir do dataset INMET.

Uso:
    cd app && python scripts/train_ml.py
"""
import os
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

import pandas as pd
from pipeline.loader import load_and_preprocess
from pipeline.accumulator import calculate_accumulations
from pipeline.utils import to_pandas
from ml.ml_model import treinar_e_salvar

DATA_PATH = os.environ.get("DATA_PATH", "data/dataset/inmet_sp.csv")
OUTPUTS_DIR = "data/ml/images"
ML_DIR = "data/ml/pkl"
DOCS_DIR = "data/docs"
AMOSTRA_MAX = 500_000


def _sample_stratified(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Amostragem estratificada por estacao, preservando todas as colunas."""
    frac = min(1.0, n / len(df))
    sampled = df.groupby("estacao", group_keys=False).sample(frac=frac, random_state=42)
    return sampled


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Dataset não encontrado em: {os.path.abspath(DATA_PATH)}")
        print("Coloque o CSV em app/data/dataset/ ou configure a variável DATA_PATH.")
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(ML_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    print("=" * 65)
    print("TREINAMENTO ML — Global Solution 2026.1")
    print(f"Working dir: {os.getcwd()}")
    print(f"Dataset: {os.path.abspath(DATA_PATH)}")
    print("=" * 65)

    print("\n[1/3] Carregando e preprocessando dados...")
    df_pl = load_and_preprocess(DATA_PATH)
    df_pl = calculate_accumulations(df_pl)

    rename_map = {}
    for n in [1, 3, 6, 12, 24, 48, 72]:
        old = f"precip_acc_{n}h"
        new = f"prec_acum_{n}h"
        if old in df_pl.columns:
            rename_map[old] = new
    df_pl = df_pl.rename(rename_map)

    df = to_pandas(df_pl)

    print(f"  Shape original: {df.shape}")
    print(f"  'estacao' in columns: {'estacao' in df.columns}")

    if len(df) > AMOSTRA_MAX:
        print(f"\n  Pre-amostrando {AMOSTRA_MAX:,} registros de {len(df):,} (estratificado por estacao)...")
        df = _sample_stratified(df, AMOSTRA_MAX)
        print(f"  Apos amostragem: {len(df):,} registros")
        print(f"  'estacao' in columns: {'estacao' in df.columns}")

    assert "estacao" in df.columns, f"'estacao' nao encontrado! Colunas: {df.columns.tolist()}"
    assert "prec_acum_24h" in df.columns, f"'prec_acum_24h' nao encontrado!"

    print(f"  Shape final: {df.shape}")
    print(f"  Colunas prec_acum: {[c for c in df.columns if 'prec_acum' in c]}")

    cmap_modelo = os.path.join(ML_DIR, "modelo_risco_climatico.pkl")
    cmap_encoder = os.path.join(ML_DIR, "label_encoder_risco.pkl")

    print("\n[3/3] Treinando modelos (isso pode demorar varios minutos)...")
    print(f"  Outputs em: {os.path.abspath(OUTPUTS_DIR)}")
    print(f"  Modelo em: {os.path.abspath(cmap_modelo)}")

    pacote = treinar_e_salvar(
        df_acumulados=df,
        caminho_modelo=cmap_modelo,
        caminho_encoder=cmap_encoder,
        caminho_outputs=OUTPUTS_DIR,
        amostra_max=AMOSTRA_MAX,
    )

    for src_name in ["modelo_preditivo_3h.pkl", "modelo_anomalias.pkl"]:
        src = os.path.join(OUTPUTS_DIR, src_name)
        if os.path.exists(src):
            dst = os.path.join(ML_DIR, src_name)
            shutil.copy2(src, dst)
            print(f"  Copiado: {src_name}")

    relatorio = os.path.join(OUTPUTS_DIR, "relatorio_ml_gs2026.pdf")
    if os.path.exists(relatorio):
        dst_relatorio = os.path.join(DOCS_DIR, "relatorio_ml_gs2026.pdf")
        shutil.copy2(relatorio, dst_relatorio)
        print(f"  Relatório PDF copiado para: {dst_relatorio}")

    print("\n" + "=" * 65)
    print("TREINAMENTO CONCLUIDO!")
    print(f"  Modelo: {pacote.get('nome_modelo', 'N/A')}")
    print(f"  Acuracia: {pacote.get('acuracia', 'N/A')}")
    print(f"  F1: {pacote.get('f1', 'N/A')}")
    if pacote.get('roc_auc'):
        print(f"  ROC-AUC: {pacote['roc_auc']}")
    print("=" * 65)


if __name__ == "__main__":
    main()