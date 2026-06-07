import streamlit as st

st.set_page_config(
    page_title="Disasters Charter",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Disasters Charter")
st.subheader("Monitoramento e Previsão de Desastres Naturais")

pg = st.navigation([
    st.Page("pages/1_dashboard.py", title="Dashboard", icon="🏠"),
    st.Page("pages/2_mapa.py", title="Mapa Interativo", icon="🗺️"),
    st.Page("pages/3_historico.py", title="Histórico", icon="📈"),
    st.Page("pages/4_ml.py", title="Modelo ML", icon="🤖"),
    st.Page("pages/5_esp32.py", title="Painel ESP32", icon="📡"),
])
pg.run()