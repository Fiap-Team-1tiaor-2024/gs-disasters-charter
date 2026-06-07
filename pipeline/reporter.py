import os
import base64
import io
import re
import unicodedata
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from PIL import Image

EXTENSOES_IMAGEM_VALIDAS = (".jpg", ".jpeg", ".png")

MEUS_LIMIARES_DE_ALERTA = [
    {"coluna_acumulado": "prec_acum_1h", "valor_mm": 25, "nivel_alerta": "ALERTA MAXIMO", "mensagem": "Chuva MUITO FORTE em 1h (>25mm)"},
    {"coluna_acumulado": "prec_acum_1h", "valor_mm": 15, "nivel_alerta": "ALERTA", "mensagem": "Chuva FORTE em 1h (>15mm)"},
    {"coluna_acumulado": "prec_acum_24h", "valor_mm": 100, "nivel_alerta": "ALERTA MAXIMO", "mensagem": "Acumulado >100mm em 24h - RISCO ALTO"},
    {"coluna_acumulado": "prec_acum_24h", "valor_mm": 70, "nivel_alerta": "ALERTA", "mensagem": "Acumulado >70mm em 24h - RISCO MODERADO"},
    {"coluna_acumulado": "prec_acum_24h", "valor_mm": 40, "nivel_alerta": "ATENCAO", "mensagem": "Acumulado >40mm em 24h - Atencao"},
    {"coluna_acumulado": "prec_acum_72h", "valor_mm": 150, "nivel_alerta": "ALERTA", "mensagem": "Acumulado >150mm em 72h (solo saturado)"},
]

MAPEAMENTO_REGIONAL_RELEVANCIA = {
    "sao sebastiao": ["bertioga", "sao sebastiao", "caraguatatuba", "ilhabela"],
    "bertioga": ["bertioga", "sao sebastiao", "guaruja", "santos"],
    "caraguatatuba": ["caraguatatuba", "sao sebastiao", "ubatuba"],
    "ubatuba": ["ubatuba", "caraguatatuba"],
    "guaruja": ["guaruja", "santos", "bertioga"],
    "santos": ["santos", "guaruja", "sao vicente"],
}

PALAVRAS_CHAVE_LOCAIS_IMAGEM = [
    "são sebastião", "bertioga", "caraguatatuba", "ubatuba",
    "ilhabela", "guarujá", "santos", "são paulo",
]


def normalizar_texto_para_comparacao(texto: str) -> str:
    """Normaliza texto removendo acentos, convertendo para minúsculas e mantendo apenas alfanuméricos."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ASCII", "ignore").decode("ASCII")
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto


def extrair_data_de_nome_arquivo(nome_arquivo: str) -> Optional[date]:
    """Extrai data do nome de arquivo usando regex (formatos ISO e BR)."""
    padrao_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", nome_arquivo)
    if padrao_iso:
        try:
            return date(int(padrao_iso.group(1)), int(padrao_iso.group(2)), int(padrao_iso.group(3)))
        except ValueError:
            return None

    padrao_br = re.search(r"(\d{2})-(\d{2})-(\d{4})", nome_arquivo)
    if padrao_br:
        try:
            return date(int(padrao_br.group(3)), int(padrao_br.group(2)), int(padrao_br.group(1)))
        except ValueError:
            return None

    return None


def criar_thumbnail_base64(caminho_completo_imagem: str, largura_max: int = 200, qualidade: int = 75) -> str:
    """Cria thumbnail em base64 a partir de caminho de imagem."""
    try:
        img = Image.open(caminho_completo_imagem)
        img = img.convert("RGB")
        ratio = largura_max / float(img.width)
        novo_tamanho = (largura_max, int(img.height * ratio))
        img = img.resize(novo_tamanho, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=qualidade)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f'<img src="data:image/jpeg;base64,{img_base64}" width="{largura_max}">'
    except Exception as e:
        return f"<i>Erro ao carregar imagem: {e}</i>"


def gerar_alertas_precipitacao(
    df_dados: pd.DataFrame,
    limiares: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Gera alertas de precipitação comparando acumulados com limiares.

    Args:
        df_dados: DataFrame com acumulados de precipitação calculados.
        limiares: Lista de dicionários com limiares de alerta.

    Returns:
        DataFrame de alertas com colunas: timestamp, estacao, nivel_alerta,
        descricao_alerta, coluna_trigger, valor_observado_mm, limiar_mm.
    """
    if limiares is None:
        limiares = MEUS_LIMIARES_DE_ALERTA

    alertas_gerados = []
    cols_estacao = {}
    if "latitude" in df_dados.columns:
        cols_estacao["latitude"] = "latitude"
    if "longitude" in df_dados.columns:
        cols_estacao["longitude"] = "longitude"
    if "municipio" in df_dados.columns:
        cols_estacao["municipio"] = "municipio"
    if "estado" in df_dados.columns:
        cols_estacao["estado"] = "estado"

    for idx, linha in df_dados.iterrows():
        timestamp = idx if isinstance(idx, pd.Timestamp) else pd.NaT
        nome_estacao = linha.get("estacao", "Desconhecida")

        dados_extras = {}
        for logico, col in cols_estacao.items():
            if col in linha.index and pd.notna(linha[col]):
                dados_extras[logico] = linha[col]

        for limiar in limiares:
            coluna_gatilho = limiar["coluna_acumulado"]
            valor_limite = limiar["valor_mm"]
            nivel = limiar["nivel_alerta"]
            msg_template = limiar["mensagem"]

            if coluna_gatilho in linha.index:
                valor_observado = linha[coluna_gatilho]
                if pd.notna(valor_observado) and valor_observado > valor_limite:
                    alerta = {
                        "timestamp": timestamp,
                        "estacao": nome_estacao,
                        "nivel_alerta": nivel,
                        "descricao_alerta": msg_template,
                        "coluna_trigger": coluna_gatilho,
                        "valor_observado_mm": valor_observado,
                        "limiar_mm": valor_limite,
                    }
                    alerta.update(dados_extras)
                    alertas_gerados.append(alerta)

    if not alertas_gerados:
        return pd.DataFrame()

    df_alertas = pd.DataFrame(alertas_gerados)

    ordem_niveis = ["ATENCAO", "ALERTA", "ALERTA MAXIMO"]
    df_alertas["nivel_alerta"] = pd.Categorical(
        df_alertas["nivel_alerta"], categories=ordem_niveis, ordered=True
    )

    df_alertas = df_alertas.sort_values(
        ["timestamp", "estacao", "nivel_alerta"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return df_alertas


def plotar_analise_chuva_estacao(
    df_acumulados: pd.DataFrame,
    nome_estacao_analise: str,
    data_inicio_plot: str,
    data_fim_plot: str,
    limiares_alerta: Optional[List[Dict]] = None,
    nomes_imagens_evento: Optional[List[str]] = None,
    caminho_para_salvar_plot: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Gera gráfico de análise de chuva para uma estação específica.

    Args:
        df_acumulados: DataFrame com dados de precipitação e acumulados.
        nome_estacao_analise: Nome da estação para filtrar.
        data_inicio_plot: Data de início para o plot (string).
        data_fim_plot: Data de fim para o plot (string).
        limiares_alerta: Lista de limiares para linhas horizontais.
        nomes_imagens_evento: Lista de nomes de imagens para marcar no gráfico.
        caminho_para_salvar_plot: Diretório para salvar o gráfico.

    Returns:
        Figura matplotlib ou None se não houver dados.
    """
    if limiares_alerta is None:
        limiares_alerta = MEUS_LIMIARES_DE_ALERTA

    df_estacao = df_acumulados[df_acumulados["estacao"] == nome_estacao_analise].copy()
    if df_estacao.empty:
        return None

    if not isinstance(df_estacao.index, pd.DatetimeIndex):
        return None

    inicio = pd.Timestamp(data_inicio_plot)
    fim = pd.Timestamp(data_fim_plot)
    df_filtrado = df_estacao.loc[inicio:fim]

    if df_filtrado.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.bar(df_filtrado.index, df_filtrado["precipitacao"], color="deepskyblue", alpha=0.7, label="Precipitação horária (mm)")
    ax1.set_xlabel("Data/Hora")
    ax1.set_ylabel("Precipitação horária (mm)", color="deepskyblue")
    ax1.tick_params(axis="y", labelcolor="deepskyblue")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    if "prec_acum_24h" in df_filtrado.columns:
        ax2 = ax1.twinx()
        ax2.plot(df_filtrado.index, df_filtrado["prec_acum_24h"], color="red", marker="o", markersize=2, label="Acumulado 24h (mm)")
        ax2.set_ylabel("Acumulado 24h (mm)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        cores_limiar = {"ATENCAO": "gold", "ALERTA": "darkorange", "ALERTA MAXIMO": "crimson"}
        for limiar in limiares_alerta:
            if limiar["coluna_acumulado"] == "prec_acum_24h":
                cor = cores_limiar.get(limiar["nivel_alerta"], "gray")
                ax2.axhline(y=limiar["valor_mm"], color=cor, linestyle="--", alpha=0.7, label=f'{limiar["nivel_alerta"]}: {limiar["valor_mm"]}mm')

    if nomes_imagens_evento:
        for nome_img in nomes_imagens_evento:
            data_img = extrair_data_de_nome_arquivo(nome_img)
            if data_img and inicio.date() <= data_img <= fim.date():
                ax1.axvline(x=pd.Timestamp(data_img), color="green", linestyle=":", alpha=0.7)

    ax1.set_title(f"Análise de Chuva — {nome_estacao_analise}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if "ax2" in dir():
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(lines1, labels1, loc="upper left", fontsize=8)

    fig.tight_layout()

    if caminho_para_salvar_plot:
        os.makedirs(caminho_para_salvar_plot, exist_ok=True)
        inicio_str = inicio.strftime("%Y%m%d")
        fim_str = fim.strftime("%Y%m%d")
        estacao_safe = re.sub(r"[^a-zA-Z0-9]", "_", nome_estacao_analise)
        caminho = os.path.join(caminho_para_salvar_plot, f"plot_chuva_{estacao_safe}_{inicio_str}_a_{fim_str}.png")
        fig.savefig(caminho, dpi=150, bbox_inches="tight")

    return fig


def plotar_mapa_estacoes(
    df_dados: pd.DataFrame,
    caminho_para_salvar_mapa: Optional[str] = None,
) -> folium.Map:
    """Gera mapa Folium com a localização de todas as estações meteorológicas.

    Args:
        df_dados: DataFrame com colunas latitude, longitude, estacao, municipio.
        caminho_para_salvar_mapa: Diretório para salvar o mapa HTML.

    Returns:
        Objeto folium.Map.
    """
    required = {"latitude", "longitude", "estacao"}
    if not required.issubset(df_dados.columns):
        raise ValueError(f"Colunas obrigatórias ausentes: {required - set(df_dados.columns)}")

    df_estacoes = df_dados.drop_duplicates(subset=["estacao"])[["latitude", "longitude", "estacao", "municipio"]].copy()
    df_estacoes = df_estacoes.dropna(subset=["latitude", "longitude"])

    centro_lat = df_estacoes["latitude"].mean()
    centro_lon = df_estacoes["longitude"].mean()

    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=7)

    for _, row in df_estacoes.iterrows():
        popup_html = f"<b>{row['estacao']}</b><br>Município: {row.get('municipio', 'N/A')}<br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_html,
            tooltip=row["estacao"],
        ).add_to(m)

    if caminho_para_salvar_mapa:
        os.makedirs(caminho_para_salvar_mapa, exist_ok=True)
        m.save(os.path.join(caminho_para_salvar_mapa, "mapa_todas_estacoes.html"))

    return m


def plotar_mapa_alertas_dia(
    df_alertas: pd.DataFrame,
    data_especifica: str,
    lista_imagens: Optional[List[str]] = None,
    caminho_pasta_imagens: Optional[str] = None,
    palavras_chave_locais: Optional[List[str]] = None,
    mapeamento_relevancia: Optional[Dict[str, List[str]]] = None,
    caminho_para_salvar_mapa: Optional[str] = None,
) -> folium.Map:
    """Gera mapa de alertas para um dia específico com thumbnails de imagens de satélite.

    Args:
        df_alertas: DataFrame de alertas.
        data_especifica: Data para filtrar (string YYYY-MM-DD).
        lista_imagens: Lista de nomes de arquivos de imagem.
        caminho_pasta_imagens: Caminho para a pasta de imagens.
        palavras_chave_locais: Lista de palavras-chave para inferir localidade.
        mapeamento_relevancia: Mapeamento de relevância regional.
        caminho_para_salvar_mapa: Diretório para salvar o mapa.

    Returns:
        Objeto folium.Map.
    """
    if palavras_chave_locais is None:
        palavras_chave_locais = PALAVRAS_CHAVE_LOCAIS_IMAGEM
    if mapeamento_relevancia is None:
        mapeamento_relevancia = MAPEAMENTO_REGIONAL_RELEVANCIA

    cores_alerta = {"ATENCAO": "yellow", "ALERTA": "orange", "ALERTA MAXIMO": "red"}

    df_dia = df_alertas[df_alertas["timestamp"].dt.strftime("%Y-%m-%d") == data_especifica].copy()
    if df_dia.empty:
        m = folium.Map(location=[-23.5505, -46.6333], zoom_start=7)
        return m

    resumo = (
        df_dia.groupby("estacao")
        .agg({"nivel_alerta": "max", "latitude": "first", "longitude": "first", "municipio": "first"})
        .reset_index()
    )

    centro_lat = resumo["latitude"].mean() if not resumo.empty else -23.5505
    centro_lon = resumo["longitude"].mean() if not resumo.empty else -46.6333
    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=8)

    for _, row in resumo.iterrows():
        nivel = row["nivel_alerta"]
        cor = cores_alerta.get(str(nivel), "gray")

        popup_html = f"<b>{row['estacao']}</b><br>Município: {row.get('municipio', 'N/A')}<br>Nível: {nivel}<br>Data: {data_especifica}"

        alertas_estacao = df_dia[df_dia["estacao"] == row["estacao"]]
        for _, alerta in alertas_estacao.iterrows():
            popup_html += f"<br><small>{alerta['descricao_alerta']} ({alerta['valor_observado_mm']:.1f}mm)</small>"

        if lista_imagens and caminho_pasta_imagens:
            for nome_img in lista_imagens:
                data_img = extrair_data_de_nome_arquivo(nome_img)
                if data_img and str(data_img) == data_especifica:
                    caminho_img = os.path.join(caminho_pasta_imagens, nome_img)
                    img_local = normalizar_texto_para_comparacao(nome_img)
                    estacao_local = normalizar_texto_para_comparacao(row.get("municipio", ""))
                    for kw in palavras_chave_locais:
                        kw_norm = normalizar_texto_para_comparacao(kw)
                        if kw_norm in img_local and (kw_norm == estacao_local or estacao_local in mapeamento_relevancia.get(kw_norm, [])):
                            thumb = criar_thumbnail_base64(caminho_img, largura_max=250)
                            popup_html += f"<br>{thumb}"
                            break

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10,
            color=cor,
            fill=True,
            fill_color=cor,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=400),
            tooltip=f"{row['estacao']}: {nivel}",
        ).add_to(m)

    if caminho_para_salvar_mapa:
        os.makedirs(caminho_para_salvar_mapa, exist_ok=True)
        data_safe = data_especifica.replace("-", "")
        m.save(os.path.join(caminho_para_salvar_mapa, f"mapa_alertas_{data_safe}.html"))

    return m


def gerar_relatorio_para_imagem(
    nome_arquivo_imagem: str,
    df_todos_alertas: pd.DataFrame,
    lista_imagens: Optional[List[str]] = None,
    caminho_para_salvar_relatorio: Optional[str] = None,
    municipio_foco_manual: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Gera relatório textual para a data de uma imagem de satélite.

    Args:
        nome_arquivo_imagem: Nome do arquivo de imagem.
        df_todos_alertas: DataFrame com todos os alertas.
        lista_imagens: Lista de nomes de imagens.
        caminho_para_salvar_relatorio: Diretório para salvar o relatório.
        municipio_foco_manual: Município de foco manual.
        top_n: Número de alertas top para exibir.

    Returns:
        String com o relatório formatado.
    """
    data_img = extrair_data_de_nome_arquivo(nome_arquivo_imagem)
    if data_img is None:
        return "Não foi possível extrair a data do nome do arquivo."

    data_str = data_img.strftime("%Y-%m-%d")
    df_dia = df_todos_alertas[df_todos_alertas["timestamp"].dt.strftime("%Y-%m-%d") == data_str].copy()

    if df_dia.empty:
        return f"Nenhum alerta registrado em {data_str}."

    linhas = []
    linhas.append(f"=" * 60)
    linhas.append(f"RELATÓRIO DE EVENTO — {data_str}")
    linhas.append(f"Imagem: {nome_arquivo_imagem}")
    linhas.append(f"=" * 60)
    linhas.append(f"Total de alertas: {len(df_dia)}")

    contagem = df_dia["nivel_alerta"].value_counts()
    for nivel in ["ATENCAO", "ALERTA", "ALERTA MAXIMO"]:
        if nivel in contagem.index:
            linhas.append(f"  {nivel}: {contagem[nivel]}")

    n_estacoes = df_dia["estacao"].nunique()
    linhas.append(f"Estações com alerta: {n_estacoes}")

    if "municipio" in df_dia.columns:
        n_municipios = df_dia["municipio"].nunique()
        linhas.append(f"Municípios com alerta: {n_municipios}")

    df_top = df_dia.sort_values(["nivel_alerta", "valor_observado_mm"], ascending=[True, False]).head(top_n)
    linhas.append(f"\nTop {top_n} alertas mais significativos:")
    linhas.append("-" * 40)
    for _, alerta in df_top.iterrows():
        linhas.append(
            f"  [{alerta.get('nivel_alerta', 'N/A')}] {alerta.get('estacao', 'N/A')} "
            f"— {alerta.get('descricao_alerta', 'N/A')} "
            f"— {alerta.get('valor_observado_mm', 0):.1f}mm"
        )

    relatorio = "\n".join(linhas)

    if caminho_para_salvar_relatorio:
        os.makedirs(caminho_para_salvar_relatorio, exist_ok=True)
        nome_safe = re.sub(r"[^a-zA-Z0-9]", "_", nome_arquivo_imagem)
        caminho = os.path.join(caminho_para_salvar_relatorio, f"relatorio_{nome_safe}.txt")
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(relatorio)

    return relatorio


def obter_nomes_imagens(caminho_pasta: str) -> List[str]:
    """Retorna lista de nomes de arquivos de imagem em uma pasta."""
    if not os.path.isdir(caminho_pasta):
        return []
    return sorted([
        f for f in os.listdir(caminho_pasta)
        if f.lower().endswith(EXTENSOES_IMAGEM_VALIDAS)
    ])