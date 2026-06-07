from typing import List, Optional

import folium
import pandas as pd

from pipeline.alerts import CORES_RISCO


def build_risk_map(
    df_estacoes: pd.DataFrame,
    df_irc: pd.DataFrame,
    nivel_minimo: str = "Normal",
    data_filtro: Optional[str] = None,
) -> folium.Map:
    """Cria mapa Folium com estacoes meteorologicas coloridas por nivel de risco.

    Args:
        df_estacoes: DataFrame com colunas latitude, longitude, estacao, municipio.
        df_irc: DataFrame com colunas estacao, irc, nivel_risco, precip_acc_24h.
        nivel_minimo: Filtra estacoes com nivel de risco >= este nivel.
        data_filtro: Data para filtrar o IRC (string YYYY-MM-DD). Se None, usa o mais recente.

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
    df_mapa = df_mapa[df_mapa["nivel_risco"].map(lambda x: ordem_risco.get(x, 0)) >= limite]

    if df_mapa.empty:
        m = folium.Map(location=[-23.5505, -46.6333], zoom_start=7)
        return m

    centro_lat = df_mapa["latitude"].mean()
    centro_lon = df_mapa["longitude"].mean()

    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=7)

    grupos = {}
    for nivel, cor in CORES_RISCO.items():
        grupos[nivel] = folium.FeatureGroup(name=f"Risco: {nivel}", show=True)

    for _, row in df_mapa.iterrows():
        nivel = row["nivel_risco"]
        cor = CORES_RISCO.get(nivel, "#999999")
        irc_val = row["irc"]
        acum_24h = row.get("precip_acc_24h", 0)
        municipio = row.get("municipio", "N/A")
        estacao = row["estacao"]

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 8px 0;">{estacao}</h4>
            <table style="width: 100%;">
                <tr><td><b>Municipio:</b></td><td>{municipio}</td></tr>
                <tr><td><b>IRC:</b></td><td>{irc_val:.3f}</td></tr>
                <tr><td><b>Acum. 24h:</b></td><td>{acum_24h:.1f} mm</td></tr>
                <tr><td><b>Nivel:</b></td><td style="color: {cor}; font-weight: bold;">{nivel}</td></tr>
            </table>
        </div>
        """

        grupo = grupos.get(nivel)
        if grupo is None:
            grupo = grupos.get("Normal", grupos.get(list(grupos.keys())[0]))

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=cor,
            fill=True,
            fill_color=cor,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{estacao}: {nivel} (IRC={irc_val:.2f})",
        ).add_to(grupo)

    for nivel, grupo in grupos.items():
        grupo.add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    _adicionar_legenda(m)

    return m


def _adicionar_legenda(m: folium.Map) -> None:
    """Adiciona legenda HTML ao mapa com as cores de risco."""
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
        <p style="margin: 0 0 5px 0; color: #000000"><b>IRC - Nivel de Risco</b></p>
    """
    for nivel, cor in CORES_RISCO.items():
        legenda_html += f"""
        <p style="margin: 2px 0; color: #000000">
            <span style="display:inline-block; width:12px; height:12px;
                  background-color:{cor}; border-radius:50%; margin-right:5px;"></span>
            {nivel}
        </p>
        """
    legenda_html += """
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #666;">
            Clique nos marcadores para detalhes
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legenda_html))