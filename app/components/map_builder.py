from typing import List, Optional

import folium
import pandas as pd

from pipeline.alerts import CORES_RISCO


NOMES_EXIBICAO = {
    "Normal": "Normal",
    "Atencao": "Atenção",
    "Alto": "Alto",
    "Muito Alto": "Muito Alto",
    "Critico": "Crítico",
}


def build_risk_map(
    df_estacoes: pd.DataFrame,
    df_irc: pd.DataFrame,
    nivel_minimo: str = "Normal",
    data_filtro: Optional[str] = None,
    df_resumo_ml: Optional[pd.DataFrame] = None,
) -> folium.Map:
    """Cria mapa Folium com estacoes meteorologicas coloridas por nivel de risco.

    Quando df_resumo_ml e fornecido (resultado de resumo_risco_por_estacao),
    a cor do marcador reflete o nivel_risco do ML e o popup exibe ambos os sistemas.
    Quando None, usa a cor IRC como antes (fallback).

    Args:
        df_estacoes: DataFrame com colunas latitude, longitude, estacao, municipio.
        df_irc: DataFrame com colunas estacao, irc, nivel_risco, precip_acc_24h.
        nivel_minimo: Filtra estacoes com nivel de risco >= este nivel.
        data_filtro: Data para filtrar o IRC (string YYYY-MM-DD). Se None, usa o mais recente.
        df_resumo_ml: DataFrame retornado por resumo_risco_por_estacao(). Opcional.

    Returns:
        Objeto folium.Map com marcadores e layer control.
    """
    ordem_risco = {"Normal": 0, "Atencao": 1, "Alto": 2, "Muito Alto": 3, "Critico": 4}
    limite = ordem_risco.get(nivel_minimo, 0)

    if not isinstance(df_irc.index, pd.DatetimeIndex):
        m = folium.Map(location=[-23.5505, -46.6333], zoom_start=7)
        return m

    if data_filtro:
        data_ts = pd.Timestamp(data_filtro)
        df_periodo = df_irc[df_irc.index.normalize() == data_ts.normalize()]
        if df_periodo.empty:
            df_periodo = df_irc
    else:
        df_periodo = df_irc

    ranking = (
        df_periodo.groupby("estacao", observed=True)
        .agg({
            "irc": "max",
            "nivel_risco": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Normal",
            "precip_acc_24h": "max",
        })
        .reset_index()
    )

    cols_estacao = ["estacao"]
    if "municipio" in df_estacoes.columns:
        cols_estacao.append("municipio")
    if "latitude" in df_estacoes.columns:
        cols_estacao.append("latitude")
    if "longitude" in df_estacoes.columns:
        cols_estacao.append("longitude")

    df_mapa = ranking.merge(
        df_estacoes[cols_estacao].drop_duplicates(subset=["estacao"]),
        on="estacao",
        how="left",
    )

    df_mapa["nivel_risco"] = df_mapa["nivel_risco"].astype(str)

    if df_resumo_ml is not None and not df_resumo_ml.empty and "estacao" in df_resumo_ml.columns:
        ml_cols = ["estacao"]
        if "risco_max" in df_resumo_ml.columns:
            ml_cols.append("risco_max")
        if "confianca_media" in df_resumo_ml.columns:
            ml_cols.append("confianca_media")
        if "cor" in df_resumo_ml.columns:
            ml_cols.append("cor")
        if "emoji" in df_resumo_ml.columns:
            ml_cols.append("emoji")
        if "descricao" in df_resumo_ml.columns:
            ml_cols.append("descricao")
        ml_merge = df_resumo_ml[ml_cols].copy()
        ml_merge = ml_merge.rename(columns={"risco_max": "risco_ml", "confianca_media": "confianca_ml"})
        df_mapa = df_mapa.merge(ml_merge, on="estacao", how="left")

    df_mapa = df_mapa[df_mapa["nivel_risco"].map(lambda x: ordem_risco.get(x, 0)) >= limite]

    if df_mapa.empty:
        m = folium.Map(location=[-23.5505, -46.6333], zoom_start=7)
        return m

    centro_lat = df_mapa["latitude"].mean()
    centro_lon = df_mapa["longitude"].mean()

    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=7)

    grupos = {}
    for nivel, cor in CORES_RISCO.items():
        nome_exibicao = NOMES_EXIBICAO.get(nivel, nivel)
        grupos[nivel] = folium.FeatureGroup(name=f"IRC: {nome_exibicao}", show=True)

    for _, row in df_mapa.iterrows():
        nivel_irc = row["nivel_risco"]
        cor_irc = CORES_RISCO.get(nivel_irc, "#999999")
        irc_val = row["irc"]
        acum_24h = row.get("precip_acc_24h", 0)
        municipio = row.get("municipio", "N/A")
        estacao = row["estacao"]

        risco_ml = row.get("risco_ml", None)
        confianca_ml = row.get("confianca_ml", None)

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 220px;">
            <h4 style="margin: 0 0 8px 0;">{estacao}</h4>
            <table style="width: 100%;">
                <tr><td><b>IRC:</b></td><td style="color: {cor_irc}; font-weight: bold;">{nivel_irc} ({irc_val:.3f})</td></tr>
        """

        if risco_ml is not None:
            cor_ml = row.get("cor", "#999999")
            emoji_ml = row.get("emoji", "")
            popup_html += f'<tr><td><b>ML:</b></td><td style="color: {cor_ml}; font-weight: bold;">{emoji_ml} {risco_ml}'
            if confianca_ml is not None:
                popup_html += f' ({confianca_ml*100:.1f}% confiança)'
            popup_html += '</td></tr>'

        popup_html += f"""
                <tr><td><b>Acum. 24h:</b></td><td>{acum_24h:.1f} mm</td></tr>
                <tr><td><b>Município:</b></td><td>{municipio}</td></tr>
            </table>
        </div>
        """

        grupo = grupos.get(nivel_irc, grupos.get("Normal", list(grupos.values())[0]))

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=cor_irc,
            fill=True,
            fill_color=cor_irc,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{estacao}: IRC={nivel_irc}" + (f" | ML={risco_ml}" if risco_ml else ""),
        ).add_to(grupo)

    for nivel, grupo in grupos.items():
        grupo.add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    _adicionar_legenda(m)

    return m


def _adicionar_legenda(m: folium.Map) -> None:
    """Adiciona legenda HTML ao mapa com as cores de risco IRC."""
    legenda_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 5px;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
        <p style="margin: 0 0 5px 0; color: #000000"><b>IRC — Nível de Risco</b></p>
    """
    for nivel, cor in CORES_RISCO.items():
        nome_exibicao = NOMES_EXIBICAO.get(nivel, nivel)
        legenda_html += f"""
        <p style="margin: 2px 0; color: #000000">
            <span style="display:inline-block; width:12px; height:12px;
                  background-color:{cor}; border-radius:50%; margin-right:5px;"></span>
            {nome_exibicao}
        </p>
        """

    legenda_html += """
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #666;">
            Clique nos marcadores para detalhes
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legenda_html))