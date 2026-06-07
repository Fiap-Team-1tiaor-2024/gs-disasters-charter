# Simulação ESP32 com Sensores e Geolocalização

Esta pasta contém a simulação da solução embarcada desenvolvida para a Global Solution 2026.2.

A simulação foi criada na plataforma Wokwi e utiliza:

- ESP32 DevKit
- Sensor DHT22 para temperatura e umidade
- Potenciômetro para simular o nível de chuva
- Display LCD 16x2 para exibir as leituras
- Geolocalização simulada da estação em Bertioga/SP

## Objetivo

A solução embarcada representa uma estação ambiental instalada em uma região de risco. O ESP32 coleta dados dos sensores, calcula uma classificação local de risco e exibe as informações no display LCD.

## Dados exibidos

- Local monitorado
- Latitude e longitude simuladas
- Temperatura
- Umidade
- Nível de chuva simulado
- Classificação do alerta

## Relação com o projeto principal

Essa camada IoT complementa a análise climática desenvolvida em Python no notebook principal. Enquanto o Python processa dados históricos e gera alertas, o ESP32 simula a coleta de dados ambientais em tempo real.