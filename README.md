# Global Solution — Disasters Charter 

## Análise e Monitoramento de Eventos Pluviais Extremos em São Paulo: Um Sistema de Alerta Baseado em Dados e Imagens de Satélite  

Integrantes do Grupo:  
Gabriela da Cunha Rocha — RM561041  
Gustavo Segantini Rossignolli — RM560111  
Vitor Lopes Romão — RM559858  
Gabrielle Halasc — RM560147  

Grupo 28 — 1TIAOR
Disciplina: Global Solution 2025.1
Curso: Inteligência Artificial
Instituição: FIAP
Data: 06/06/2025

### 1. Introdução  
#### 1.1. Contextualização  
Nos últimos anos, a intensificação dos eventos climáticos extremos tornou-se uma das maiores preocupações globais. Enchentes, secas prolongadas e deslizamentos de terra, impulsionados pelas mudanças climáticas, têm impactado comunidades com frequência e severidade crescentes. O estado de São Paulo, em particular, possui um histórico de desastres naturais associados a chuvas intensas, especialmente em áreas de serra e regiões litorâneas, resultando em perdas humanas e materiais significativas. Este cenário demanda respostas rápidas, inteligentes e tecnológicas para o monitoramento e mitigação de riscos. 
#### 1.2. Desafio da Global Solution  
Alinhada a essa necessidade, a FIAP propôs para a Global Solution 2025.1 o desafio de desenvolver uma solução digital baseada em dados reais, capaz de prever, monitorar ou mitigar os impactos de eventos naturais extremos. A proposta incentiva a aplicação integrada de conhecimentos em lógica computacional, programação em Python e estruturação de dados, utilizando como base as informações de desastres disponibilizadas por agências globais através do site disasterscharter.org.  
#### 1.3. Problema Específico Abordado  
Nosso grupo escolheu focar no monitoramento de chuvas intensas e na identificação de períodos de alto risco para deslizamentos e enchentes no estado de São Paulo. A escolha foi motivada pela alta relevância do problema para a nossa região e pela disponibilidade de dados abertos de alta qualidade, incluindo registros de estações meteorológicas do Instituto Nacional de Meteorologia (INMET) e imagens de satélite de desastres reais, como o trágico evento ocorrido em São Sebastião em fevereiro de 2023.  
#### 1.4. Objetivos do Projeto  
O objetivo geral deste projeto foi desenvolver um sistema em Python para analisar dados históricos de precipitação, identificar períodos críticos, gerar alerta baseados em limiares configuráveis e correlacionar esses eventos com evidências visuais (imagens de satélite), culminando em uma ferramenta robusta para análise e visualização de eventos extremos.

Os objetivos específicos foram:  

Processar e estruturar um grande volume de dados de precipitação horária de múltiplas estações meteorológicas.
Implementar algoritmos para calcular a precipitação acumulada em diferentes janelas de tempo (1, 3, 6, 12, 24, 48 e 72 horas), um indicador chave para o risco de desastres.
Construir um sistema de alerta com múltiplos níveis de severidade (ATENÇÃO, ALERTA, ALERTA MÁXIMO).
Correlacionar temporalmente os alerta gerados com imagens de satélite de desastres conhecidos.
Desenvolver visualizações de dados eficazes, incluindo gráficos de série temporal e mapas geoespaciais interativos, para facilitar a interpretação dos resultados.
Gerar relatórios textuais automatizados para eventos críticos específicos.

### 2. Desenvolvimento
#### 2.1. Arquitetura da Solução
A solução foi arquitetada como um pipeline de processamento de dados sequencial e modular, implementado em Python. Cada módulo é responsável por uma etapa específica do processo, desde a ingestão dos dados brutos até a geração dos outputs finais (relatórios e visualizações).

Diagrama de Fluxo Conceitual:


Os principais módulos são:
Carregamento de Dados: Lê e pré-processa os dados brutos do arquivo CSV.
Análise de Precipitação: Calcula os acumulados de chuva, enriquecendo os dados.
Sistema de Alerta: Aplica regras de negócio (limiares) para classificar o risco.
Gerenciamento de Imagens: Carrega dinamicamente a lista de imagens de uma pasta e extrai metadados (data).
Geração de Outputs: Cria os relatórios, gráficos e mapas interativos.

#### 2.2. Fontes de Dados
Dados de Precipitação: Utilizamos um conjunto de dados do INMET contendo 4.632.374 registros de medições horárias de estações meteorológicas automáticas em São Paulo. As colunas essenciais para o projeto foram data, hora, precipitacao_total, estacao, municipio, latitude e longitude.
Imagens de Satélite: Foram utilizadas 15 imagens de alta resolução do portal disasterscharter.org, focadas no desastre de São Sebastião em fevereiro de 2023, além de outros eventos no estado. Essas imagens serviram como “verdade em campo” para validar a eficácia do nosso sistema de alerta.
#### 2.3. Ferramentas e Justificativas
A solução foi inteiramente desenvolvida em Python 3, devido à sua versatilidade e ao seu ecossistema maduro para análise de dados. As seguintes bibliotecas foram cruciais:
Pandas: Essencial para a manipulação, limpeza e análise dos mais de 4,6 milhões de registros de chuva, permitindo cálculos eficientes de séries temporais e agregações.
Matplotlib: Utilizada para a criação dos gráficos de série temporal, que permitem uma análise visual detalhada da evolução da chuva e dos acumulados ao longo do tempo.
Folium: Escolhida para a geração dos mapas geoespaciais interativos, por sua facilidade de uso e pela capacidade de criar visualizações ricas (marcadores coloridos, popups com HTML, legendas) que podem ser salvas em arquivos .html autônomos.
Pillow (PIL): Empregada para o processamento de imagens, especificamente para criar as miniaturas (thumbnails) exibidas nos popups dos mapas, otimizando o tamanho e a performance.
Bibliotecas Padrão (OS, RE, Datetime, Unicodedata, Base64): Usadas para tarefas fundamentais como manipulação de caminhos de arquivos, extração de datas de nomes de arquivos com expressões regulares, normalização de texto e codificação de imagens para embutir no HTML.

#### 2.4. Implementação Detalhada (Códigos e Lógica)
O código foi estruturado em funções modulares para garantir clareza, reusabilidade e facilidade de manutenção. Apresentamos aqui a lógica dos componentes mais importantes.

Lógica de Geração de Alerta:
O coração do sistema reside na função gerar_alertas_precipitacao. Após o cálculo dos acumulados de chuva para cada estação, esta função itera sobre cada registro horário e o compara com uma lista de limiares pré-definidos (MEUS_LIMIARES_DE_ALERTA).

Lógica de Correlação de Imagem Relevante:
~~~python
for limiar in limiares:
           coluna_gatilho = limiar['coluna_acumulado']
           valor_limite = limiar['valor_mm']
           nivel = limiar['nivel_alerta']
           msg_template = limiar['mensagem']
           if coluna_gatilho in linha.index: # Verificar se a coluna de acumulado existe na linha
               valor_observado = linha[coluna_gatilho]
               if pd.notna(valor_observado) and valor_observado > valor_limite:
                   alerta = {
                       'timestamp': timestamp,
                       'estacao': nome_estacao,
                       'nivel_alerta': nivel,
                       'descricao_alerta': msg_template,
                       'coluna_trigger': coluna_gatilho,
                       'valor_observado_mm': valor_observado,
                       'limiar_mm': valor_limite
                   }
                   alerta.update(dados_extras_estacao)
                   alertas_gerados.append(alerta)
           else:
               print(f"Aviso: Coluna de gatilho '{coluna_gatilho}' não encontrada para a estação {nome_estacao} em {timestamp}.")
~~~~

Na função plotar_mapa_alertas_dia_com_thumbnails_relevantes, implementamos uma lógica para exibir nos popups somente as imagens geograficamente relevantes para a estação em questão. Para isso, criamos um mapeamento regional (MAPEAMENTO_REGIONAL_RELEVANCIA) e uma lista de palavras-chave (PALAVRAS_CHAVE_LOCAIS_IMAGEM). 

A relevância é determinada por:

Inferência do Local da Imagem: O nome do arquivo da imagem é analisado em busca de palavras-chave que indiquem um município.
Verificação de Relevância: Uma imagem é considerada relevante para uma estação se: 
a. O município da imagem for o mesmo da estação. 
b. O município da imagem estiver mapeado como “relevante” para o município da estação.

Isso garante que o popup de uma estação em Campinas, por exemplo, não mostre uma imagem do desastre em São Sebastião.
Fluxo de Execução Principal:
O script é orquestrado por uma função main que executa todas as etapas em sequência, de forma dinâmica e automatizada:

Carrega dinamicamente os nomes dos arquivos da pasta de imagens.
Processa o CSV de chuvas e gera os alertas para todo o período.
Identifica as datas únicas para as quais existem imagens.
Inicia um loop por cada data de evento única, gerando automaticamente: 
a. Relatórios textuais detalhados para cada imagem daquela data. 
b. O mapa de alerta geoespacial para aquela data, com miniaturas relevantes. 
c. Gráficos de série temporal para as estações que apresentaram os alertas mais significativos naquele dia. (O código-fonte completo do projeto está disponível no repositório do projeto)

### 3. Resultados e Discussão
A execução do sistema sobre o conjunto de dados completo do INMET e as 15 imagens de desastres fornecidas gerou resultados significativos que validam a eficácia da nossa abordagem.

#### 3.1. Visão Geral dos Alertas
O sistema processou os 4,6 milhões de registros e identificou um total de 110.142 alerta ao longo de todo o período, distribuídos entre 42 estações meteorológicas únicas. Isso demonstra a capacidade da solução de analisar grandes volumes de dados e extrair eventos de interesse.

#### 3.2. Análise de Evento Crítico 
Desastre de São Sebastião/Bertioga (Fevereiro de 2023)
O foco da nossa análise de resultados foi o evento extremo que atingiu o litoral norte de São Paulo em fevereiro de 2023.

Data do Evento: 19/02/2023
Neste dia, o sistema registrou 144 alerta, dos quais 25 foram de Nível Máximo. A estação de Bertioga, geograficamente próxima e climaticamente similar a São Sebastião, foi o epicentro dos alertas, registrando um alarmante acumulado de chuva em 24 horas de 292,6 mm, quase o triplo do limiar de risco alto (100 mm).


Legenda da Figura 2: O gráfico ilustra a evolução da precipitação horária (barras azuis) e acumulada em 24h (linha vermelha) para a estação de Bertioga. É visível o rápido aumento do acumulado, ultrapassando os limiares de ATENÇÃO (amarelo), ALERTA (laranja) e ALERTA MÁXIMO (vermelho) durante os dias 18 e 19, coincidindo com a data da imagem de satélite do desastre.

Data do Evento: 20/02/2023
A situação crítica persistiu. Neste dia, foram 65 alerta, com a estação de Bertioga ainda reportando um acumulado em 24h de 156,8 mm e um acumulado em 72h de impressionantes 295,4 mm, indicando um solo completamente saturado e altíssimo risco de deslizamentos. O mapa de alerta gerado para esta data mostra claramente a concentração e a severidade dos alertas na região litorânea.

Legenda da Figura 3: Mapa geoespacial dos alertas gerados em 20/02/2023. O marcador vermelho sobre Bertioga indica “ALERTA MÁXIMO” O popup interativo mostra os detalhes dos alertas e a miniatura da imagem de satélite correspondente, provando a correlação direta entre os dados de chuva e a evidência visual do desastre.

A forte correlação entre os alertas de nível máximo gerados pelo sistema para a estação de Bertioga e o desastre documentado pelas imagens de São Sebastião valida a eficácia dos limiares e da metodologia empregada.

### 4. Conclusões
#### 4.1. Recapitulação dos Resultados
Este projeto resultou em um sistema funcional e automatizado capaz de processar um grande volume de dados climáticos, identificar eventos de chuva extrema com base em limiares técnicos, e correlacioná-los de forma temporal e espacial com imagens de desastres reais. A análise do caso de São Sebastião/Bertioga demonstrou com sucesso a capacidade do sistema de identificar um evento real de altíssimo impacto.

#### 4.2. Atendimento aos Objetivos da Global Solution
Consideramos que os objetivos da Global Solution foram plenamente atendidos. Desenvolvemos uma solução digital funcional, baseada em dados reais, aplicando integradamente conceito de lógica de programação (loops, condicionais), estruturação de dados (DataFrames, dicionários, listas), modularização de código (funções) e boas práticas de desenvolvimento em Python, culminando em uma ferramenta de análise com impacto potencial real.

#### 4.3. Limitações e Trabalhos Futuros
A solução atual, embora robusta, possui limitações, a análise depende da densidade de estações do INMET, que pode não cobrir todas as áreas de risco. A correlação com imagens é primariamente temporal e a inferência de localidade é baseada em heurísticas sobre nomes de arquivos.

Como trabalhos futuros, o sistema poderia ser expandido para:

Integrar dados de previsão meteorológica para gerar alertas preditivos.
Utilizar dados de relevo, tipo de solo e níveis de rios para um cálculo de risco mais sofisticado. Implementar modelos de Machine Learning para prever a probabilidade de desastres com base em múltiplos fatores. Evoluir para uma aplicação web em tempo real com um painel de monitoramento (dashboard).

#### 4.4. Considerações Finais
O desenvolvimento deste projeto foi uma experiência de aprendizado imensurável, permitindo-nos aplicar conceitos teóricos em um problema prático, complexo e socialmente relevante. Acreditamos que soluções baseadas em dados como a que desenvolvemos são fundamentais para construir um futuro mais resiliente aos desafios climáticos.

