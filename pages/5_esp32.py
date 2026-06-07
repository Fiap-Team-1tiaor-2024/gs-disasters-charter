import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from components.sensor_reader import load_sensor_data, normalizar_nivel_chuva, ultima_leitura


ESP32_DATA_PATH = os.environ.get("ESP32_DATA_PATH", "data/esp32_simulado.json")

st.header("📡 Painel ESP32")

try: 
    with st.spinner("Carregando dados do ESP32..."):
        df_sensor = load_sensor_data(ESP32_DATA_PATH)

    if df_sensor.empty:
        st.warning("Nenhum dado do sensor ESP32 disponivel.")
        st.info("Verifique se o arquivo `data/esp32_simulado.json` existe e esta no formato correto.")
        st.subheader("Formato esperado do JSON")
        st.json([
            {
                "temperatura": 28.5,
                "umidade": 78.2,
                "nivel_chuva": 2048,
                "timestamp": "2025-06-01T10:00:00"
            }
        ])
        st.stop()

    df_sensor = normalizar_nivel_chuva(df_sensor, escala_max=4095)

    leitura = ultima_leitura(df_sensor)

    st.subheader("Leitura Atual")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Temperatura", value=f"{leitura['temperatura']:.1f} °C")

    with col2:
        st.metric("Umidade", value=f"{leitura['umidade']:.1f} %")

    with col3:
        st.metric("Nivel de Chuva", value=f"{leitura['nivel_chuva_pct']:.1f} %")

    st.divider()

    st.subheader("Status do Sensor")
    ultima_ts = leitura["timestamp"]
    st.success(f"Conectado — ultima leitura: {ultima_ts:%d/%m/%Y %H:%M}")

    st.divider()

    st.subheader("Series Temporais")

    with st.spinner("Gerando graficos..."):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        timestamps = df_sensor["timestamp"]

        axes[0].plot(timestamps, df_sensor["temperatura"], color="#e74c3c", linewidth=1.5, marker="o", markersize=3)
        axes[0].set_ylabel("Temperatura (°C)")
        axes[0].set_title("Temperatura ao Longo do Tempo")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(timestamps, df_sensor["umidade"], color="#3498db", linewidth=1.5, marker="o", markersize=3)
        axes[1].set_ylabel("Umidade (%)")
        axes[1].set_title("Umidade ao Longo do Tempo")
        axes[1].grid(True, alpha=0.3)

        axes[2].fill_between(timestamps, df_sensor["nivel_chuva_pct"], color="#2ecc71", alpha=0.4)
        axes[2].plot(timestamps, df_sensor["nivel_chuva_pct"], color="#2ecc71", linewidth=1.5, marker="o", markersize=3)
        axes[2].set_ylabel("Nivel de Chuva (%)")
        axes[2].set_title("Nivel de Chuva ao Longo do Tempo")
        axes[2].set_xlabel("Tempo")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

        st.divider()

        st.subheader("Integracao com IRC")
        st.info(
            "Os dados do sensor ESP32 podem ser integrados ao calculo do IRC como features adicionais. "
            "Quando a umidade do sensor esta acima do P90 historico, um fator multiplicativo e aplicado "
            "ao IRC base, aumentando o nivel de risco. "
            "Esta integracao sera ativada quando os dados do sensor estiverem disponiveis em tempo real."
        )

    if not df_sensor.empty:
        umidade_max = df_sensor["umidade"].max()
        umidade_media = df_sensor["umidade"].mean()
        st.write(f"Umidade maxima registrada: **{umidade_max:.1f}%** | Media: **{umidade_media:.1f}%**")
except Exception as e:
    st.error(f"Erro ao carregar ou processar os dados do ESP32: {e}")
    st.info("Verifique o arquivo de dados e tente novamente.")