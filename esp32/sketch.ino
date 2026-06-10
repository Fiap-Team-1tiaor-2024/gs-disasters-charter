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