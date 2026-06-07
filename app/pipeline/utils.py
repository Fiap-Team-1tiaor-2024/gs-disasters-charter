import polars as pl
import pandas as pd


def to_pandas(df: pl.DataFrame, index_col: str = "timestamp") -> pd.DataFrame:
    """Converte pl.DataFrame para pd.DataFrame com timestamp como indice.

    Funcao de conversao para compatibilidade com Matplotlib, Folium e
    componentes Streamlit que exigem pd.DataFrame.

    Args:
        df: pl.DataFrame do pipeline.
        index_col: Nome da coluna a ser usada como indice do pd.DataFrame.
            Se a coluna nao existir, o indice sera RangeIndex.

    Returns:
        pd.DataFrame com timestamp como indice DatetimeIndex.
    """
    pdf = df.to_pandas()

    if index_col in pdf.columns:
        pdf[index_col] = pd.to_datetime(pdf[index_col])
        pdf = pdf.set_index(index_col)
        pdf = pdf.sort_index()

    for col in ["estacao", "municipio", "estado", "nivel_risco"]:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype("category")

    return pdf