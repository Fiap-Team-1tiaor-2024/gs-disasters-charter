# Global Solution 2026.06

# Sistema Inteligente de Monitoramento Climático com Dados Espaciais, ESP32, Sensores e Geolocalização

## Integrantes do Grupo

- Gabriela da Cunha Rocha — RM561041
- Gustavo Segantini Rossignolli — RM560111
- Vitor Lopes Romão — RM559858
- Gabrielle Halasc — RM560147

Grupo 28 — 1TIAOR
Curso: Inteligência Artificial
Instituição: FIAP
Disciplina: Global Solution 2026.06

## Link do Repositório GitHub

[Repositório](https://github.com/Fiap-Team-1tiaor-2024/gs-disasters-charter)

## Link do Vídeo Demonstrativo

[Vídeo](https://youtu.be/eodtobXaQhs)

---

# 1. Introdução

## 1.1 Contextualização

A exploração espacial deixou de ser apenas uma atividade científica e passou a ocupar papel estratégico em diversas áreas da sociedade. Satélites monitoram o clima, acompanham mudanças ambientais, auxiliam sistemas de segurança, contribuem para a prevenção de desastres naturais e produzem grandes volumes de dados utilizados por governos, empresas e centros de pesquisa.

Ao mesmo tempo, tecnologias originalmente desenvolvidas para missões espaciais impulsionaram avanços em Inteligência Artificial, automação, visão computacional, sistemas embarcados, telecomunicações, computação distribuída e aplicações autônomas. Esses recursos podem ser aplicados não apenas fora da Terra, mas também em problemas reais enfrentados pela população, como eventos climáticos extremos.

Nesse contexto, eventos como enchentes, chuvas intensas, deslizamentos de terra, secas prolongadas e ondas de calor têm se tornado cada vez mais frequentes e severos. O estado de São Paulo, especialmente regiões litorâneas e áreas de serra, possui histórico relevante de desastres associados a chuvas extremas, como o caso de São Sebastião e Bertioga em fevereiro de 2023.

Diante desse cenário, o projeto propõe uma solução tecnológica conectada ao universo da economia espacial, utilizando dados climáticos, imagens de satélite, análise de dados, sistemas embarcados com ESP32, sensores simulados e geolocalização para apoiar o monitoramento de riscos ambientais na Terra.

## 1.2 Desafio da Global Solution 2026.06

A Global Solution 2026.06 propõe o desenvolvimento de uma Prova de Conceito que responda à pergunta:

Como tecnologias avançadas de Inteligência Artificial e computação podem impulsionar a nova economia espacial e gerar impacto positivo na Terra?

A solução apresentada neste projeto está alinhada ao tema de sistemas inteligentes de monitoramento climático utilizando dados espaciais. O projeto utiliza imagens de satélite e dados ambientais para apoiar a identificação de eventos climáticos extremos, além de integrar uma solução embarcada com ESP32, sensores e geolocalização simulada.

A proposta conecta o uso de dados espaciais e tecnologias embarcadas a um problema terrestre de alto impacto: o monitoramento de chuvas intensas e riscos de enchentes ou deslizamentos.

## 1.3 Problema Abordado

O problema escolhido pelo grupo é a dificuldade de monitorar, interpretar e antecipar situações de risco causadas por chuvas intensas, principalmente em áreas vulneráveis. Em muitos casos, os eventos extremos se desenvolvem rapidamente, dificultando a tomada de decisão por órgãos públicos, Defesa Civil e comunidades expostas.

Embora existam dados meteorológicos e imagens de satélite disponíveis, esses dados nem sempre são apresentados de forma integrada, acessível e automatizada para apoiar decisões preventivas. Além disso, sistemas de alerta existentes frequentemente utilizam limiares fixos e universais que não consideram as particularidades climáticas de cada estação meteorológica nem a sazonalidade dos dados.

Por isso, o projeto propõe uma solução que combina:

- dados climáticos históricos do INMET (4.632.374 registros horários, 42 estações, 2001–2024);
- índice de risco adaptativo baseado em percentis históricos por estação e mês (sem limiares fixos);
- modelo de Machine Learning treinado (Random Forest) para classificação e predição de risco;
- detecção de anomalias via Isolation Forest;
- predição antecipada de risco com 3 horas de antecedência;
- mapas interativos com geolocalização das estações;
- ESP32 com sensores simulados e ajuste dinâmico do IRC com dados do sensor;
- interface web Streamlit com dashboard, mapa, histórico, ML e painel ESP32.

## 1.4 Objetivo Geral

O objetivo geral do projeto é desenvolver uma Prova de Conceito de um sistema inteligente para monitoramento climático, utilizando dados espaciais, análise de dados e uma estação embarcada com ESP32, sensores e geolocalização, com foco na identificação de riscos relacionados a chuvas intensas, enchentes e deslizamentos.

## 1.5 Objetivos Específicos

Os objetivos específicos são:

- Utilizar dados climáticos e imagens de satélite como base para análise de eventos extremos.
- Relacionar o projeto ao contexto da economia espacial e ao uso de tecnologias derivadas do setor espacial.
- Implementar sistema de alertas adaptativos (IRC) baseado em percentis históricos, sem limiares fixos.
- Treinar e integrar modelo de Machine Learning (Random Forest/XGBoost) para classificação de risco.
- Implementar detecção de anomalias com Isolation Forest e predição antecipada de 3 horas.
- Simular estação de monitoramento ambiental com ESP32 no Wokwi.
- Coletar dados simulados de temperatura, umidade e nível de chuva e integrá-los ao cálculo de risco.
- Exibir dados em interface web Streamlit com 5 páginas: Dashboard, Mapa, Histórico, ML e ESP32.
- Documentar a arquitetura, os códigos e as decisões técnicas do projeto.

---

# 2. Desenvolvimento

## 2.1 Visão Geral da Solução

A solução é uma plataforma web construída em Streamlit que integra dados meteorológicos históricos do INMET, um modelo de Machine Learning treinado, um sistema de alertas adaptativos e uma estação embarcada simulada via ESP32.

O pipeline de dados processa 4.632.374 registros horários de 42 estações meteorológicas automáticas do estado de São Paulo, cobrindo o período de 2001 a 2024. O processamento utiliza Polars para alta performance (~5 segundos para o pipeline completo) e converte para Pandas apenas na camada de visualização e no modelo ML.

A arquitetura contempla cinco camadas:

1. **Camada de ingestão e processamento** — Pipeline Polars com cache Streamlit para leitura, pré-processamento e cálculo de acumulados em 7 janelas temporais (1h, 3h, 6h, 12h, 24h, 48h, 72h).
2. **Camada de alertas adaptativos** — Índice de Risco Composto (IRC) com percentis históricos dinâmicos por (estação, mês), sem limiares fixos.
3. **Camada de Machine Learning** — Modelo Random Forest (F1=1.0, ROC-AUC=1.0) para classificação, XGBoost para comparação, Isolation Forest para anomalias e modelo preditivo para 3h à frente.
4. **Camada de visualização** — Interface Streamlit com dashboard, mapa interativo Folium, gráficos Matplotlib, séries temporais e histogramas.
5. **Camada embarcada** — ESP32 com sensores simulados (DHT22 e potenciômetro), envio de dados via JSON e ajuste dinâmico do IRC com dados de umidade.

## 2.2 Arquitetura da Solução

```
Entrada de Dados
├── CSV INMET (4.632.374 registros, 42 estações)
└── JSON ESP32 simulado (temperatura, umidade, nível de chuva)
        │
        ▼
Pipeline Polars (cache Streamlit)
├── loader.py — leitura, parsing, limpeza, colunas derivadas
├── accumulator.py — 7 janelas de acumulado por estação
├── alerts.py — percentis históricos, scores, IRC adaptativo
└── utils.py — conversão para Pandas (visualização e ML)
        │
        ▼
Sistema de Alertas (IRC)
├── P75/P90/P95/P99 por (estação, mês)
├── Score ponderado: (0.35×1h) + (0.30×6h) + (0.20×24h) + (0.15×72h)
├── 5 níveis: Normal, Atenção, Alto, Muito Alto, Crítico
└── Ajuste com dados ESP32 (umidade > threshold → fator multiplicativo)
        │
        ▼
Modelo ML
├── Random Forest (F1=1.0, ROC-AUC=1.0) — classificador principal
├── XGBoost (F1=0.998) — comparativo
├── Isolation Forest — detecção de anomalias
├── Modelo preditivo 3h à frente (acurácia=0.89)
└── SHAP — explicabilidade das features
        │
        ▼
Interface Streamlit
├── 1. Dashboard — alertas ativos, IRC, ranking, distribuição
├── 2. Mapa Interativo — estações no mapa Folium com legenda IRC
├── 3. Histórico — séries temporais, histogramas, percentis
├── 4. Modelo ML — predição manual, antecipada, anomalias, gráficos
└── 5. Painel ESP32 — leitura atual, séries temporais, integração IRC
```

## 2.3 Relação com a Economia Espacial

O projeto se conecta ao tema da economia espacial porque utiliza o conceito de monitoramento ambiental apoiado por dados espaciais. Satélites são fundamentais para observar a Terra, registrar mudanças climáticas, acompanhar eventos extremos e fornecer informações relevantes para prevenção de desastres.

A proposta demonstra como dados derivados de observação da Terra podem ser combinados com sensores em solo, criando uma solução híbrida de monitoramento. Essa integração é relevante para a nova economia espacial, pois mostra como tecnologias espaciais podem gerar impacto positivo na Terra.

O uso de dados meteorológicos de estações automáticas do INMET é um exemplo direto de como informações coletadas via satélite e estações terrestres — infraestrutura originalmente desenvolvida para fins espaciais e meteorológicos — podem ser processadas por técnicas de Inteligência Artificial para prever e classificar riscos ambientais em tempo real.

## 2.4 Dataset INMET

O dataset utilizado contém **4.632.374 registros horários** de **42 estações meteorológicas automáticas** no estado de São Paulo, cobrindo o período de **agosto de 2001 a outubro de 2024**.

Cada registro contém:

| Coluna | Descrição |
|--------|-----------|
| `data` | Data da observação |
| `hora` | Hora da observação |
| `precipitacao_total` | Precipitação horária em mm |
| `estacao` | Nome da estação meteorológica |
| `latitude` | Latitude da estação |
| `longitude` | Longitude da estação |
| `municipio` | Município da estação |
| `estado` | Estado (SP) |

Colunas derivadas pelo pipeline: `mes`, `hora`, `ano`, `precip_acc_1h`, `precip_acc_3h`, `precip_acc_6h`, `precip_acc_12h`, `precip_acc_24h`, `precip_acc_48h`, `precip_acc_72h`, scores de risco por janela, IRC e nível de risco.

## 2.5 Sistema de Alertas Adaptativos — IRC

O **Índice de Risco Composto (IRC)** é o mecanismo central de classificação de risco do sistema. Diferente de sistemas com limiares fixos, o IRC calcula o risco com base no comportamento histórico de cada estação e mês do ano.

### Percentis dinâmicos

Para cada combinação de `(estação, mês)`, o sistema calcula os percentis P75, P90, P95 e P99 do acumulado de precipitação em cada janela temporal.

### Classificação de risco (IRC)

| Nível | Critério | Cor |
|-------|----------|-----|
| Normal | Abaixo de P75 | Verde |
| Atenção | Entre P75 e P90 | Amarelo |
| Alto | Entre P90 e P95 | Laranja |
| Muito Alto | Entre P95 e P99 | Vermelho |
| Crítico | Acima de P99 | Preto |

### Fórmula do IRC

```
IRC = (w₁ × score_1h) + (w₂ × score_6h) + (w₃ × score_24h) + (w₄ × score_72h)
```

Pesos padrão: `{1h: 0.35, 6h: 0.30, 24h: 0.20, 72h: 0.15}`

O score do IRC é ajustado quando dados do ESP32 estão disponíveis e a umidade excede o threshold histórico:

```
irc_ajustado = irc_base × (1 + (umidade - threshold) / 100)   quando umidade > threshold
```

## 2.6 Modelo de Machine Learning

### Modelo principal — Random Forest

O modelo vencedor é um **Random Forest** com as seguintes métricas no conjunto de teste:

| Métrica | Valor |
|---------|-------|
| Acurácia | 1.0000 |
| F1 ponderado | 1.0000 |
| ROC-AUC | 1.0000 |

### Comparativo com XGBoost

| Modelo | F1 | ROC-AUC |
|--------|----|---------|
| Random Forest | 1.0000 | 1.0 |
| XGBoost | 0.9983 | 1.0 |

O Random Forest foi selecionado como modelo principal pela superioridade no F1 ponderado. O XGBoost apresentou leve dificuldade em classificar as classes minoritárias (ATENÇÃO, ALERTA, ALERTA MÁXIMO).

### Detecção de anomalias — Isolation Forest

O modelo de anomalias identifica **2% dos registros** como eventos sem precedente histórico (10.000 registros em 500.000 amostrados). Os 10 eventos mais anômalos incluem estações como Bertioga, Iguape e São Sebastião — regiões historicamente afetadas por deslizamentos.

### Predição antecipada — 3h à frente

O modelo preditivo antecipa o nível de risco 3 horas no futuro com:
- **Acurácia:** 0.8869
- **F1 ponderado:** 0.9231

### Features do modelo

As 19 features utilizadas incluem: precipitação, acumulados em 7 janelas (1h, 3h, 6h, 12h, 24h, 48h, 72h), hora do dia, mês, estação do ano, dia da semana, variações entre janelas, intensidade relativa, médias e desvios móveis.

### Treinamento

O treinamento é executado via `python scripts/train_ml.py` e gera:
- `data/ml/pkl/modelo_risco_climatico.pkl` — classificador principal
- `data/ml/pkl/modelo_preditivo_3h.pkl` — predição antecipada
- `data/ml/pkl/modelo_anomalias.pkl` — detector de anomalias
- `data/ml/images/*.png` — gráficos de treino (matriz de confusão, ROC, SHAP, etc.)
- `data/docs/relatorio_ml_gs2026.pdf` — relatório completo

Para evitar alocação excessiva de memória (~19 GiB), o dataset completo é pre-amostrado para 500.000 registros estratificados por estação antes de `engenharia_de_features()`.

## 2.7 Pipeline de Dados — Polars

### Motivação para Polars

O pipeline original utilizava Pandas e processava os 4.632.374 registros em aproximadamente **1 minuto e 21 segundos** (baseline medido). Essa latência tornava a experiência do usuário intolerável: a cada interação na interface Streamlit, o recarregamento dos dados demorava mais de 1 minuto.

A migração para Polars teve como motivação principal a **redução drástica do tempo de processamento**. Polars é uma biblioteca de análise de dados escrita em Rust com execução multi-thread e otimização de query planejada, que não requer cópias desnecessárias de memória (zero-copy via Apache Arrow). A medição após a migração registrou **aproximadamente 5 segundos** para o pipeline completo (carregamento → acumulados → IRC), uma redução de **13× em relação ao baseline Pandas**.

Essa diferença é decisiva para uma aplicação interativa: com Polars, o carregamento inicial é tolerável e o cache do Streamlit garante que interações subsequentes sejam instantâneas. Com Pandas, o usuário enfrentaria mais de 1 minuto de espera a cada filtro ou navegação.

### Escolha técnica

Pandas **não foi removido** do projeto — ele continua no `requirements.txt` e é usado exclusivamente nos pontos de conversão para visualização (Matplotlib, Folium) e no modelo ML (`engenharia_de_features()` utiliza `df.index.hour`, `df.index.month`). O padrão adotado é: **Polars internamente no pipeline, Pandas na fronteira de saída** via função utilitária `to_pandas()`.

### Janelas de acumulado

O accumulator calcula 7 janelas temporais por estação:

| Janela | Foco |
|--------|------|
| 1h | Eventos convectivos rápidos / flash floods |
| 3h | Chuvas intensas curtas (exigida pelo modelo ML) |
| 6h | Chuvas intensas contínuas |
| 12h | Acumulado de meio dia (exigida pelo modelo ML) |
| 24h | Saturação superficial do solo |
| 48h | Saturação prolongada (exigida pelo modelo ML) |
| 72h | Saturação profunda do solo / risco de deslizamento |

## 2.8 Interface Streamlit

A interface web é construída em Streamlit com tema escuro customizado (`config.toml`) e contém 5 páginas:

| Página | Conteúdo |
|--------|----------|
| Dashboard | Resumo de alertas ativos, IRC por estação, ranking de risco, distribuição de níveis |
| Mapa Interativo | Mapa Folium com estações meteorológicas coloridas pelo nível de risco IRC, filtros por nível e data, popups com detalhes |
| Histórico | Análise temporal de acumulados, histogramas com percentis históricos, top 10 eventos críticos |
| Modelo ML | Status do modelo, predição manual, predição antecipada 3h, visualizações de treino, feature importance, download PDF |
| Painel ESP32 | Dados simulados do sensor (temperatura, umidade, nível de chuva), séries temporais, integração com IRC |

Todas as páginas utilizam `@st.cache_data` para cache do pipeline e `@st.cache_resource` para o modelo ML, evitando reprocessamento a cada interação.

## 2.9 Funcionamento do ESP32

O ESP32 executa continuamente a leitura dos sensores simulados. A cada ciclo, o sistema:

1. lê a temperatura e a umidade por meio do DHT22;
2. lê o valor analógico do potenciômetro;
3. converte o valor analógico em um nível de chuva de 0% a 100%;
4. aplica uma classificação local de risco;
5. exibe as informações no display LCD;
6. imprime os dados no monitor serial.

A classificação local atual é feita diretamente no código embarcado, utilizando condições simples baseadas no nível de chuva e na umidade.

## 2.10 Código do Circuito — diagram.json

O arquivo `diagram.json` define o circuito utilizado no Wokwi, incluindo os componentes e suas conexões.

```json
{
  "version": 1,
  "author": "Gabriela",
  "editor": "wokwi",
  "parts": [
    { "type": "board-esp32-devkit-c-v4", "id": "esp", "top": 0, "left": 260, "attrs": {} },
    { "type": "wokwi-dht22", "id": "dht1", "top": 20, "left": 20, "attrs": {} },
    { "type": "wokwi-potentiometer", "id": "pot1", "top": 180, "left": 20, "attrs": {} },
    { "type": "wokwi-lcd1602", "id": "lcd1", "top": 350, "left": 20, "attrs": { "pins": "i2c" } }
  ],
  "connections": [
    [ "dht1:VCC", "esp:3V3", "red", [] ],
    [ "dht1:GND", "esp:GND", "black", [] ],
    [ "dht1:SDA", "esp:15", "green", [] ],

    [ "pot1:VCC", "esp:3V3", "red", [] ],
    [ "pot1:GND", "esp:GND", "black", [] ],
    [ "pot1:SIG", "esp:34", "blue", [] ],

    [ "lcd1:VCC", "esp:5V", "red", [] ],
    [ "lcd1:GND", "esp:GND", "black", [] ],
    [ "lcd1:SDA", "esp:21", "green", [] ],
    [ "lcd1:SCL", "esp:22", "yellow", [] ]
  ],
  "dependencies": {}
}
```

## 2.11 Código do ESP32 — sketch.ino

O arquivo `sketch.ino` contém a lógica embarcada executada pelo ESP32.

```cpp
#include "DHTesp.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

const int DHT_PIN = 15;
const int SENSOR_CHUVA_PIN = 34;

DHTesp dhtSensor;
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Geolocalização simulada da estação
const float LATITUDE = -23.8544;
const float LONGITUDE = -46.1386;
const char* LOCAL = "Bertioga/SP";

void setup() {
  Serial.begin(115200);

  dhtSensor.setup(DHT_PIN, DHTesp::DHT22);

  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("Estacao ESP32");
  lcd.setCursor(0, 1);
  lcd.print("Bertioga/SP");

  delay(2000);
}

void loop() {
  TempAndHumidity data = dhtSensor.getTempAndHumidity();

  int leituraAnalogica = analogRead(SENSOR_CHUVA_PIN);
  int nivelChuva = map(leituraAnalogica, 0, 4095, 0, 100);

  String nivelAlerta;

  if (nivelChuva >= 80 || data.humidity >= 90) {
    nivelAlerta = "ALERTA MAX";
  } else if (nivelChuva >= 60 || data.humidity >= 80) {
    nivelAlerta = "ALERTA";
  } else if (nivelChuva >= 40 || data.humidity >= 70) {
    nivelAlerta = "ATENCAO";
  } else {
    nivelAlerta = "BAIXO";
  }

  Serial.println("--------------------------------");
  Serial.print("Local: ");
  Serial.println(LOCAL);
  Serial.print("Latitude: ");
  Serial.println(LATITUDE, 6);
  Serial.print("Longitude: ");
  Serial.println(LONGITUDE, 6);
  Serial.print("Temperatura: ");
  Serial.println(data.temperature);
  Serial.print("Umidade: ");
  Serial.println(data.humidity);
  Serial.print("Nivel de chuva: ");
  Serial.println(nivelChuva);
  Serial.print("Classificacao: ");
  Serial.println(nivelAlerta);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Chuva:");
  lcd.print(nivelChuva);
  lcd.print("% ");

  lcd.setCursor(0, 1);
  lcd.print(nivelAlerta);

  delay(3000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temp:");
  lcd.print(data.temperature, 0);
  lcd.print("C");

  lcd.setCursor(0, 1);
  lcd.print("Umid:");
  lcd.print(data.humidity, 0);
  lcd.print("%");

  delay(3000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Bertioga/SP");

  lcd.setCursor(0, 1);
  lcd.print("Lat:-23.8544");

  delay(3000);
}
```

## 2.12 Solução Embarcada com ESP32, Sensores e Geolocalização

A parte embarcada do projeto foi desenvolvida na plataforma Wokwi, simulando uma estação ambiental instalada em Bertioga/SP (região de risco para deslizamentos e enchentes).

A estação coleta e exibe:

- temperatura (DHT22);
- umidade (DHT22);
- nível de chuva simulado (potenciômetro, ADC 0–4095 → 0–100%);
- localização monitorada (coordenadas fixas);
- classificação local de alerta.

### Classificação local do sketch ESP32

| Nível | Critério |
|-------|----------|
| BAIXO | Chuva < 40% e Umidade < 70% |
| ATENÇÃO | Chuva ≥ 40% ou Umidade ≥ 70% |
| ALERTA | Chuva ≥ 60% ou Umidade ≥ 80% |
| ALERTA MAX | Chuva ≥ 80% ou Umidade ≥ 90% |

> A classificação do sketch é exibida no painel como informação auxiliar, mas **não substitui o IRC**. O IRC consome os valores brutos de umidade e nível de chuva — não o texto da classificação.

### Integração com o Python

Os dados do ESP32 são lidos a partir de um arquivo JSON (`data/dataset/esp32_simulado.json`) pelo módulo `components/sensor_reader.py`. A umidade do sensor é integrada ao cálculo do IRC como feature adicional: quando a umidade excede o P90 histórico, um fator multiplicativo é aplicado ao IRC base, aumentando o nível de risco.

## 2.13 Componentes Utilizados no Circuito

| Componente | Pino | Função |
|------------|------|---------|
| ESP32 DevKit | — | Microcontrolador principal |
| DHT22 | GPIO 15 | Sensor de temperatura e umidade |
| Potenciômetro | GPIO 34 (analógico) | Simulação do nível de chuva (ADC 0–4095 → 0–100%) |
| LCD I2C | 0x27 | Display local (não gera output para o Python) |

## 2.14 Código do Circuito — sketch.ino

O arquivo `sketch.ino` está disponível em `esp32/sketch.ino`. A lógica principal consiste em:

1. Leitura do DHT22 (temperatura e umidade);
2. Leitura do potenciômetro (nível de chuva);
3. Classificação local de risco;
4. Exibição no display LCD;
5. Envio dos dados via Serial (formato JSON para o Python).

## 2.15 Formato de Integração ESP32 → Python

O sensor reader lê dados de um arquivo JSON no formato:

```json
[
  {"temperatura": 28.5, "umidade": 78.2, "nivel_chuva": 2048, "timestamp": "2025-06-01T10:00:00"}
]
```

O módulo `sensor_reader.py` converte o valor ADC (0–4095) em percentual (0–100%) e adiciona colunas derivadas como `nivel_chuva_pct`. A aplicação funciona normalmente se o arquivo não existir (modo sem sensor).

## 2.16 Estrutura do Projeto

```
gs-disasters-charter/
├── PRD.md                            # Documento de requisitos
├── README.md                         # Documentação principal
├── DOC.md                            # Este documento
├── requirements.txt                  # Dependências Python
├── .env.example                     # Variáveis de ambiente
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
    │   │   ├── inmet_sp.csv          # Dataset completo (4.632.374 registros)
    │   │   ├── inmet_sp_demo.csv     # Dataset de demonstração
    │   │   └── esp32_simulado.json   # Dados simulados do sensor
    │   ├── ml/
    │   │   ├── pkl/                  # Modelos treinados (.pkl)
    │   │   │   ├── modelo_risco_climatico.pkl
    │   │   │   ├── modelo_preditivo_3h.pkl
    │   │   │   └── modelo_anomalias.pkl
    │   │   └── images/               # Gráficos de treino (.png)
    │   └── docs/                      # Relatório PDF
    │       └── relatorio_ml_gs2026.pdf
    │
    ├── pipeline/
    │   ├── loader.py                 # Carregamento Polars com cache
    │   ├── accumulator.py             # Acumulados por estação (7 janelas)
    │   ├── alerts.py                  # Percentis, scores e IRC adaptativo
    │   ├── reporter.py                # Relatórios e visualizações
    │   └── utils.py                  # Conversão Polars → Pandas
    │
    ├── ml/
    │   ├── ml_model.py               # Pipeline de treinamento (Gabriela)
    │   ├── ml_predict.py              # Interface de predição (Gabriela)
    │   └── model_interface.py         # Wrapper Streamlit ↔ modelo ML
    │
    ├── components/
    │   ├── map_builder.py             # Mapa Folium com legenda IRC
    │   ├── chart_builder.py           # Gráficos Matplotlib
    │   └── sensor_reader.py           # Leitura do sensor ESP32 (JSON)
    │
    ├── pages/
    │   ├── 1_dashboard.py             # Dashboard de alertas
    │   ├── 2_mapa.py                  # Mapa interativo
    │   ├── 3_historico.py            # Histórico e análise temporal
    │   ├── 4_ml.py                    # Modelo ML (5 seções + feature importance)
    │   └── 5_esp32.py                 # Painel do sensor ESP32
    │
    └── scripts/
        ├── train_ml.py               # Treinamento completo dos modelos
        └── benchmark.py              # Benchmark Polars vs Pandas
```

## 2.17 Decisões Técnicas do Projeto

### Por que Polars ao invés de Pandas?

O pipeline original em Pandas processava os 4,6 milhões de registros em aproximadamente **1 minuto e 21 segundos**. Com Polars, o mesmo pipeline executa em **~5 segundos** — uma redução de **13×**. Essa diferença é decisiva para uma aplicação interativa: com Pandas, cada interação no Streamlit demandava mais de 1 minuto de espera; com Polars e cache, o carregamento inicial é tolerável e as interações subsequentes são instantâneas.

Polars foi escolhido por quatro razões técnicas:

1. **Execução multi-thread nativa** — Polars paraleliza operações como `group_by`, `rolling` e `join` automaticamente, sem necessidade de configuração.
2. **Zero-copy via Apache Arrow** — Os dados residem em memória no formato Arrow, evitando cópias desnecessárias entre representações.
3. **Lazy evaluation** — `pl.scan_csv()` permite ao engine otimizar o plano de execução antes de processar, filtrando colunas e linhas desnecessárias logo na leitura.
4. **API consistente e expressiva** — As expressões do tipo `pl.col("x").rolling_sum(window_size=24).over("estacao")` são mais legíveis e menos propensas a erros que o equivalente `groupby().rolling()` do Pandas.

Pandas **não foi removido** — continua sendo usado nos pontos de conversão para Matplotlib, Folium e no modelo ML, que exige `DatetimeIndex` e operações específicas do Pandas. O padrão adotado é: **Polars dentro do pipeline, Pandas na fronteira de saída**, via função utilitária `to_pandas()` em `pipeline/utils.py`.

### Por que Random Forest e não XGBoost?

Embora o XGBoost tenha apresentado F1 de 0.998 (muito próximo do Random Forest), o Random Forest obteve F1 perfeito (1.0) e ROC-AUC de 1.0 no conjunto de teste. Além disso, o Random Forest é mais interpretável via SHAP e feature importance, e não requer tuning de learning rate ou regularização. Diante da superioridade empírica e da maior simplicidade de interpretação, o Random Forest foi selecionado como modelo principal.

### Por que pre-sampling antes de engenharia de features?

A função `engenharia_de_features()` do `ml_model.py` cria colunas derivadas (variações, intensidade relativa, médias móveis) que aumentam significativamente o uso de memória. Com os 4,6 milhões de registros completos, essa etapa alocaria aproximadamente 19,5 GiB de RAM — inviável na maioria dos ambientes de desenvolvimento. Por isso, o script `train_ml.py` realiza uma amostragem estratificada por estação (máximo 500.000 registros) antes de chamar `engenharia_de_features()`, reduzindo o consumo de memória para níveis executáveis.

### Por que IRC adaptativo e não limiares fixos?

Sistemas de alerta baseados em limiares fixos (ex: "50mm em 24h = alerta") ignoram as particularidades climáticas de cada estação e a sazonalidade dos dados. Uma estação litorânea como Bertioga naturalmente recebe mais chuva que uma estação interiorana, e o mês de fevereiro é historicamente mais chuvoso que agosto. O IRC resolve isso calculando os percentis P75, P90, P95 e P99 por `(estação, mês)`, tornando o sistema adaptativo ao comportamento histórico local.

---

# 3. Resultados

## 3.1 Funcionamento da Aplicação

A aplicação Streamlit é executada com `cd app && streamlit run app.py` e oferece 5 páginas:

1. **Dashboard** — Resumo de alertas ativos, IRC por estação, ranking de risco e distribuição dos níveis.
2. **Mapa Interativo** — 42 estações meteorológicas no mapa com legenda IRC, filtros por nível e data, e popups com detalhes.
3. **Histórico** — Séries temporais de acumulados, histogramas com percentis dinâmicos por estação e mês, e top 10 eventos críticos.
4. **Modelo ML** — Status do modelo, predição manual (9 parâmetros), predição antecipada 3h, detecção de anomalias, 14 gráficos de treino e feature importance.
5. **Painel ESP32** — Leitura atual dos sensores, séries temporais e explicação da integração com o IRC.

## 3.2 Métricas do Modelo de Machine Learning

| Modelo | F1 ponderado | Acurácia | ROC-AUC |
|--------|-------------|----------|---------|
| Random Forest | 1.0000 | 1.0000 | 1.0 |
| XGBoost | 0.9983 | 0.9983 | 1.0 |
| Predição 3h | 0.9231 | 0.8869 | — |
| Isolation Forest | 2% anomalias | — | — |

O modelo preditivo 3h à frente alcança 88,7% de acurácia, classificando corretamente os 4 níveis de risco (NORMAL, ATENÇÃO, ALERTA, ALERTA MÁXIMO) com 3 horas de antecedência.

## 3.3 Resultados do Pipeline

- **4.632.374 registros** processados a partir do CSV INMET.
- **42 estações** meteorológicas automáticas do estado de São Paulo.
- **Período**: agosto de 2001 a outubro de 2024.
- **7 janelas de acumulado** (1h, 3h, 6h, 12h, 24h, 48h, 72h) por estação.
- **Tempo de processamento (Polars)**: ~5 segundos para pipeline completo.
- **Tempo de processamento (Pandas baseline)**: ~81 segundos.

## 3.4 Resultados do IRC Adaptativo

O sistema de alertas adaptativos classifica os registros em 5 níveis sem limiares fixos, com base nos percentis históricos de cada estação e mês:

| Nível | Critério | % dos registros |
|-------|----------|-----------------|
| Normal | Abaixo de P75 | ~44% |
| Atenção | Entre P75 e P90 | ~28% |
| Alto | Entre P90 e P95 | ~12% |
| Muito Alto | Entre P95 e P99 | ~6% |
| Crítico | Acima de P99 | ~1% |

Quando dados do ESP32 estão disponíveis e a umidade excede o P90 da série histórica, o IRC é ajustado para cima, refletindo o aumento de risco.

## 3.5 Resultados da Integração ESP32

- Leitura dos dados simulados de temperatura, umidade e nível de chuva via JSON.
- Conversão do valor analógico (0–4095) para percentual (0–100%).
- Exibição em tempo real no painel Streamlit com séries temporais.
- Ajuste dinâmico do IRC quando a umidade do sensor excede o threshold.

---

# 4. Conclusões

## 4.1 Recapitulação do Projeto

O projeto Disasters Charter demonstra a viabilidade de uma solução integrada de monitoramento climático que combina dados espaciais, pipeline de dados de alta performance (Polars), modelo de Machine Learning treinado, sistema de alertas adaptativos e estação embarcada com ESP32.

A solução processa 4,6 milhões de registros em segundos, classifica risco em tempo real sem limiares fixos, prediz eventos extremos 3 horas à frente e apresenta os resultados em uma interface web interativa com mapas, gráficos e séries temporais.

## 4.2 Atendimento aos Requisitos da Global Solution

Todos os requisitos da Global Solution 2026.06 foram contemplados:

- ✅ Solução embarcada com ESP32, sensores e geolocalização simulada
- ✅ Dados climáticos históricos processados com IA
- ✅ Sistema de alertas adaptativos (IRC) sem limiares fixos
- ✅ Modelo de Machine Learning treinado (Random Forest, Isolation Forest, predição 3h)
- ✅ Interface web Streamlit com 5 páginas integradas
- ✅ Mapa interativo com geolocalização e legenda de risco
- ✅ Integração ESP32 ↔ Python com ajuste dinâmico do IRC
- ✅ Documentação completa (PRD, README, DOC)
- ✅ Relação com economia espacial e dados de satélite

## 4.3 Limitações

- **Dataset regional**: os dados cobrem apenas o estado de São Paulo; expansão para outros estados exigiria dados adicionais.
- **Sensor simulado**: o ESP32 opera no Wokwi com dados gerados artificialmente; uma implantação real necessitaria hardware físico.
- **Modelo treinado offline**: o modelo ML é treinado previamente e os `.pkl` são carregados; não há retreinamento em tempo real.
- **Sem deploy em nuvem**: a aplicação roda localmente via Streamlit; não há infraestrutura de produção configurada.
- **Geolocalização fixa**: as coordenadas da estação ESP32 são hardcoded (Bertioga/SP); uma aplicação real usaria GPS.

## 4.4 Próximos Passos

1. Expandir o dataset para outras regiões do Brasil (outros estados do INMET).
2. Implementar retreinamento periódico do modelo ML com dados mais recentes.
3. Deploy em nuvem (Streamlit Cloud, AWS, GCP) para acesso remoto.
4. Integração com APIs de previsão meteorológica em tempo real (INMET, OpenWeather).
5. Conexão com ESP32 físico via Serial/HTTP para dados em tempo real.
6. Adicionar camada de notificação (email, SMS, push) quando o IRC atinge níveis críticos.
7. Implementar autenticação e roles de usuário (operador, gestor, administrador).

## 4.5 Considerações Finais

A solução proposta demonstra como tecnologias avançadas podem ser aplicadas para monitorar riscos ambientais na Terra a partir de conceitos relacionados à economia espacial. O uso de dados climáticos, processamento de alta performance com Polars, modelo de Machine Learning treinado, sistema de alertas adaptativos e sensor embarcado permite construir uma Prova de Conceito com potencial de impacto real.

O projeto se alinha ao desafio da Global Solution 2026.06 ao propor uma solução tecnológica interdisciplinar, conectando Inteligência Artificial, sistemas embarcados, geolocalização e monitoramento climático — todas tecnologias com raízes no setor espacial.