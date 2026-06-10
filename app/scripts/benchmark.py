"""
Benchmark do pipeline completo: leitura -> acumulados -> IRC.

Uso:
    cd app && python scripts/benchmark.py

Mede o tempo de execucao do pipeline Pandas (baseline) e Polars,
e compara os resultados.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.environ.get("DATA_PATH", "data/dataset/inmet_sp_demo.csv")


def run_polars_pipeline():
    from pipeline.loader import load_and_preprocess
    from pipeline.accumulator import calculate_accumulations
    from pipeline.alerts import calculate_irc, calculate_historical_percentiles
    from pipeline.utils import to_pandas

    print("=" * 60)
    print("BENCHMARK — Pipeline Polars")
    print("=" * 60)

    t0 = time.time()
    df_pl = load_and_preprocess(DATA_PATH)
    t_load = time.time() - t0
    print(f"[load] {t_load:.2f}s — {len(df_pl):,} registros, type={type(df_pl).__name__}")

    t1 = time.time()
    df_pl = calculate_accumulations(df_pl)
    t_acc = time.time() - t1
    print(f"[accum] {t_acc:.2f}s")

    t2 = time.time()
    pct = calculate_historical_percentiles(df_pl)
    df_pl = calculate_irc(df_pl, percentiles=pct)
    t_irc = time.time() - t2
    print(f"[irc] {t_irc:.2f}s")

    t3 = time.time()
    df = to_pandas(df_pl)
    t_conv = time.time() - t3
    print(f"[to_pandas] {t_conv:.2f}s")

    t_total = time.time() - t0
    print(f"\nRESULTADO:")
    print(f"  load={t_load:.2f}s accum={t_acc:.2f}s irc={t_irc:.2f}s to_pandas={t_conv:.2f}s total={t_total:.2f}s")
    print(f"  Shape: {df.shape}")
    print(f"  IRC range: [{df['irc'].min():.4f}, {df['irc'].max():.4f}]")
    print(f"  Levels: {sorted(df['nivel_risco'].unique().tolist())}")
    print("=" * 60)

    return {
        "load": t_load,
        "accum": t_acc,
        "irc": t_irc,
        "to_pandas": t_conv,
        "total": t_total,
    }


if __name__ == "__main__":
    run_polars_pipeline()