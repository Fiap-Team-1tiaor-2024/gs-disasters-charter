# 🌦️ Disasters Charter — Global Solution 2026.06

> **Monitoramento e Previsão de Desastres Naturais com IA e Dados Espaciais**

A exploração espacial deixou de ser apenas científica e passou a representar uma das maiores oportunidades tecnológicas e estratégicas da atualidade. Satélites monitoram o clima, auxiliam sistemas de segurança e produzem grandes volumes de dados utilizados por governos e centros de pesquisa. Tecnologias originalmente desenvolvidas para missões espaciais impulsionaram avanços em inteligência artificial, automação e computação distribuída.

O **Disasters Charter** é uma plataforma de monitoramento e previsão de desastres naturais que aplica técnicas de **inteligência artificial** sobre dados meteorológicos reais do **INMET** (Instituto Nacional de Meteorologia). O sistema integra uma pipeline de dados em tempo real, um modelo de classificação de risco via machine learning, um sensor ESP32 simulado e visualizações interativas — resolvendo problemas reais de prevenção e resposta a eventos pluviais extremos no estado de São Paulo.

---

## 🎥 Vídeo Demonstrativo

🔗 [Vídeo](https://youtu.be/eodtobXaQhs)

---

## ✅ Funcionalidades Principais

- **Dashboard** — Resumo de alertas ativos, IRC por estação, ranking de risco e distribuição de níveis
- **Mapa Interativo** — Visualização georreferenciada das estações meteorológicas com legenda de risco IRC, filtros por nível e data, e popups com detalhes de cada estação
- **Histórico** — Análise temporal de acumulados, percentis históricos dinâmicos por estação e mês, e ranking dos 10 eventos mais críticos
- **Modelo ML** — Predição de risco com Random Forest/XGBoost, detecção de anomalias via Isolation Forest, predição antecipada 3h, gráficos de treino (matriz de confusão, ROC, SHAP) e download do relatório PDF
- **Painel ESP32** — Integração com sensor simulado (temperatura, umidade, nível de chuva), com séries temporais em tempo real e ajuste dinâmico do IRC
- **Pipeline Polars** — Carregamento, acumulados e cálculo de IRC em ~0,10s (13x mais rápido que Pandas puro)
- **Treinamento Automatizado** — Script `train_ml.py` que gera modelos, gráficos e relatório PDF a partir do dataset completo

---

## 🛠️ Tecnologias Utilizadas

| Categoria | Tecnologia |
|-----------|------------|
| **Interface** | Streamlit, Folium, Streamlit-Folium |
| **Processamento** | Polars (pipeline), Pandas (compatibilidade) |
| **Machine Learning** | Scikit-learn, XGBoost, SHAP, Isolation Forest |
| **Visualização** | Matplotlib, Folium |
| **Sensor** | ESP32 (simulação Wokwi), JSON |
| **Linguagem** | Python 3.13 |
| **Versionamento** | Git |

---

## 📁 Estrutura do Projeto

```
gs-disasters-charter/
├── PRD.md                            # Documento de requisitos
├── README.md
├── requirements.txt
├── .env.example
│
├── esp32/                            # Simulação Wokwi (fora do app)
│   ├── diagram.json
│   ├── sketch.ino
│   └── README.md
│
└── app/                              # Aplicação Streamlit
    ├── app.py                        # Entry point
    ├── .streamlit/config.toml        # Tema escuro customizado
    │
    ├── data/
    │   ├── dataset/                  # CSV e JSON (não versionados)
    │   │   ├── inmet_sp.csv
    │   │   ├── inmet_sp_demo.csv
    │   │   └── esp32_simulado.json
    │   ├── ml/
    │   │   ├── pkl/                  # Modelos treinados (.pkl)
    │   │   └── images/               # Gráficos e outputs do treino
    │   └── docs/                     # Relatório PDF gerado
    │
    ├── pipeline/
    │   ├── loader.py                 # Carregamento Polars com cache
    │   ├── accumulator.py            # Acumulados por estação (7 janelas)
    │   ├── alerts.py                 # Percentis, scores e IRC
    │   ├── reporter.py               # Relatórios e visualizações
    │   └── utils.py                  # Conversão Polars → Pandas
    │
    ├── ml/
    │   ├── ml_model.py               # Treinamento, SHAP, IsolationForest
    │   ├── ml_predict.py             # Predição manual, lote e antecipada
    │   └── model_interface.py         # Interface Streamlit ↔ modelo ML
    │
    ├── components/
    │   ├── map_builder.py             # Mapa Folium com legenda IRC
    │   ├── chart_builder.py           # Gráficos de distribuição e séries
    │   └── sensor_reader.py           # Leitura do sensor ESP32
    │
    ├── pages/
    │   ├── 1_dashboard.py
    │   ├── 2_mapa.py
    │   ├── 3_historico.py
    │   ├── 4_ml.py
    │   └── 5_esp32.py
    │
    └── scripts/
        ├── train_ml.py               # Treinamento completo dos modelos
        └── benchmark.py              # Benchmark Pipeline Polars vs Pandas
```

---

## ⚙️ Configuração e Instalação

### Pré-requisitos

- Python 3.11+
- Dataset INMET-SP (arquivo CSV)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/<usuario>/gs-disasters-charter.git
cd gs-disasters-charter

# Crie e ative o ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### Execução

```bash
cd app
streamlit run app.py
```

> O comando `streamlit run app.py` deve ser executado de dentro do diretório `app/`.

### Dataset INMET

Coloque o arquivo CSV em `app/data/dataset/inmet_sp.csv` ou configure a variável de ambiente:

```bash
export DATA_PATH=data/dataset/inmet_sp.csv   # Linux/macOS
set DATA_PATH=data/dataset/inmet_sp.csv       # Windows
```

O CSV deve conter as colunas: `data`, `hora`, `precipitacao_total`, `estacao`, `latitude`, `longitude`, `municipio`, `estado`.

### Dataset de Demonstração

Para executar sem o dataset completo:

```bash
cd app
export DATA_PATH=data/dataset/inmet_sp_demo.csv
streamlit run app.py
```

O demo contém 43.800 registros (5 estações, 12 meses) e carrega em ~2 segundos.

### 🌱 Sensor ESP32

Coloque o arquivo JSON em `app/data/dataset/esp32_simulado.json` ou configure:

```bash
export ESP32_DATA_PATH=data/dataset/esp32_simulado.json
```

Formato esperado:

```json
[
  {"temperatura": 28.5, "umidade": 78.2, "nivel_chuva": 2048, "timestamp": "2025-06-01T10:00:00"}
]
```

### 🤖 Treinamento do Modelo ML

Para treinar os modelos a partir do dataset:

```bash
cd app
python scripts/train_ml.py
```

Isso gera os artefatos em `app/data/`:

| Artefato | Caminho |
|----------|---------|
| Classificador principal | `data/ml/pkl/modelo_risco_climatico.pkl` |
| Modelo preditivo 3h | `data/ml/pkl/modelo_preditivo_3h.pkl` |
| Detector de anomalias | `data/ml/pkl/modelo_anomalias.pkl` |
| Gráficos de treino | `data/ml/images/*.png` |
| Relatório PDF | `data/docs/relatorio_ml_gs2026.pdf` |

> Se os modelos não forem encontrados, a página de ML opera em modo **stub** com dados de referência.

---

## 👨‍💻 Autores

- Gabrielle Barao Halasc Frateschi - RM560147@fiap.com.br
- Gabriela da Cunha Rocha - RM561041@fiap.com.br
- Gustavo Segantini Rossignolli - RM560111@fiap.com.br
- Vitor Lopes Romão - RM559858@fiap.com.br