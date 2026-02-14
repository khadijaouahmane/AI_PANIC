#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <HTTPClient.h>

#include <Adafruit_MPU6050.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_Sensor.h>

#include "MAX30105.h"
#include "heartRate.h"

// Librairie Edge Impulse générée (Arduino Library)
#include "doaa-project-1_inferencing.h"

// ============================================================================
// 1) CONFIGURATION MATERIELLE (I2C + PINS)
// ============================================================================

// Bus I2C unique pour OLED + MPU6050 + MAX30105
const int SDA_I2C = 21;
const int SCL_I2C = 22;

// Adresse OLED
const uint8_t OLED_ADDR = 0x3C;

// Bouton (INPUT_PULLUP : HIGH au repos, LOW quand on appuie)
const int BUTTON_PIN = 32;

// Vibreur (moteur)
const int MOTOR_PIN  = 13;

// ============================================================================
// 2) WIFI / MQTT / FLASK
// ============================================================================

// WiFi
static const char* WIFI_SSID = "S21 de Jihane";
static const char* WIFI_PASS = "gdpi5594";

// MQTT (broker Node-RED)
static const char* MQTT_HOST = "10.104.31.93";
static const int   MQTT_PORT = 1883;

// Topics MQTT
static const char* TOPIC_PANIC_ALERT  = "panic/alert";
static const char* TOPIC_PANIC_CANCEL = "panic/cancel";
static const char* TOPIC_STATUS       = "womenSafety/esp32/status";
static const char* TOPIC_SENSORS      = "sensors/data"; // debug

// Flask (serveur IA audio / traitement)
static const char* FLASK_URL_BASE  = "http://10.104.31.5:6000";
static const char* EP_TRIGGER_AUTO = "/trigger_auto";

// ============================================================================
// 3) SEUILS BPM (NORMAL / TEST)
// ============================================================================

static bool MODE_TEST = true;

struct Thresholds {
  int bpmMin;
  int bpmMax;
};

Thresholds cfgNormal {50, 130};
Thresholds cfgTest   {65,  85};
Thresholds config;

// ============================================================================
// 4) CONFIGURATION EDGE IMPULSE (LABEL + SEUIL)
// ============================================================================

// Mettre ici le label exact défini dans Edge Impulse
static const char* EI_DANGER_LABEL = "danger";

// Score minimal pour considérer "danger"
static const float EI_DANGER_MIN_SCORE = 0.70f;

// ============================================================================
// 5) TIMING (debounce / double click / auto)
// ============================================================================

static const uint32_t DEBOUNCE_MS = 50;
static const uint32_t DOUBLECLICK_WINDOW_MS = 2000; // 2s (double clic)
static const uint32_t LOCKOUT_MS = 350;             // anti rebond logique
static const uint32_t UI_EVERY_MS = 200;            // rafraîchissement OLED
static const uint32_t SENSOR_PUBLISH_MS = 500;      // debug MQTT
static const uint32_t AUTO_TRIGGER_COOLDOWN_MS = 8000; // anti spam auto

// ============================================================================
// 6) OBJETS (capteurs + réseau)
// ============================================================================

TwoWire I2C_Bus0 = TwoWire(0);

Adafruit_SSD1306 display(128, 64, &I2C_Bus0, -1);
Adafruit_MPU6050 mpu;
MAX30105 max30105;

WiFiClient espClient;
PubSubClient mqtt(espClient);

// ============================================================================
// 7) DONNEES CAPTEURS (BPM + IMU)
// ============================================================================

struct SensorData {
  float bpm = 0;
  float bpmEma = 0;
  bool finger = false;
  uint32_t irAvg = 0;

  float ax = 0, ay = 0, az = 0;

  uint32_t lastBeatMs = 0;
  uint32_t lastAutoTriggerMs = 0;
} data;

// ============================================================================
// 8) GESTION BOUTON (debounce + double click)
// ============================================================================

static int stableBtn = HIGH;
static int lastRead  = HIGH;
static uint32_t lastChangeMs = 0;

static bool waitingSecondClick = false;
static uint32_t firstClickMs = 0;
static uint32_t lockoutUntilMs = 0;

// ============================================================================
// 9) EDGE IMPULSE BUFFER (fenêtre IMU)
// ============================================================================

// Buffer d’entrée du modèle EI
static float ei_features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static size_t ei_feat_ix = 0;
static uint32_t lastEiSampleMs = 0;

// Période d’échantillonnage EI (définie dans EI_CLASSIFIER_FREQUENCY)
static inline uint32_t eiSamplePeriodMs() {
  return (uint32_t)(1000.0f / EI_CLASSIFIER_FREQUENCY);
}

// Etat auto : collecte IMU ou idle
enum class AutoState { IDLE, COLLECTING };
AutoState autoState = AutoState::IDLE;

// ============================================================================
// 10) OUTILS (vibration / JSON)
// ============================================================================

// Petit buzz du moteur (feedback utilisateur)
void motorBuzz(uint16_t ms = 30) {
  digitalWrite(MOTOR_PIN, HIGH);
  delay(ms);
  digitalWrite(MOTOR_PIN, LOW);
}

// JSON capteurs (debug)
String jsonSensors() {
  String p="{";
  p += "\"bpm\":" + String(data.bpm, 1) + ",";
  p += "\"finger\":" + String(data.finger ? "true":"false") + ",";
  p += "\"irAvg\":" + String((unsigned long)data.irAvg) + ",";
  p += "\"ax\":" + String(data.ax, 3) + ",";
  p += "\"ay\":" + String(data.ay, 3) + ",";
  p += "\"az\":" + String(data.az, 3);
  p += "}";
  return p;
}

// JSON alerte (manual ou auto)
String jsonAlert(const char* reason) {
  String p="{";
  p += "\"panic\":true,";
  p += "\"reason\":\"" + String(reason) + "\",";
  p += "\"alertId\":\"" + String(millis()) + "-" + String(random(1000,9999)) + "\",";
  p += "\"bpm\":" + String(data.bpm, 1) + ",";
  p += "\"irAvg\":" + String((unsigned long)data.irAvg) + ",";
  p += "\"ax\":" + String(data.ax, 3) + ",";
  p += "\"ay\":" + String(data.ay, 3) + ",";
  p += "\"az\":" + String(data.az, 3) + ",";
  p += "\"time_ms\":" + String((unsigned long)millis());
  p += "}";
  return p;
}

// ============================================================================
// 11) MQTT
// ============================================================================

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  char msg[128];
  unsigned int n = (length < sizeof(msg)-1) ? length : sizeof(msg)-1;
  memcpy(msg, payload, n);
  msg[n] = '\0';
  Serial.printf("[MQTT] RX %s => %s\n", topic, msg);
}

// Connexion MQTT stable
bool mqttEnsureConnected() {
  if (WiFi.status() != WL_CONNECTED) return false;
  if (mqtt.connected()) return true;

  String cid = "esp32-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  Serial.printf("[MQTT] Connecting %s:%d ...\n", MQTT_HOST, MQTT_PORT);

  bool ok = mqtt.connect(cid.c_str());
  if (!ok) {
    Serial.printf("[MQTT] FAIL state=%d\n", mqtt.state());
    return false;
  }

  Serial.println("[MQTT] Connected");
  mqtt.publish(TOPIC_STATUS, "online", true);
  return true;
}

// Publication MQTT sécurisée
bool mqttPublishChecked(const char* topic, const String& payload, bool retained=false) {
  if (!mqttEnsureConnected()) {
    Serial.printf("[MQTT] cannot publish %s\n", topic);
    return false;
  }
  bool ok = mqtt.publish(topic, payload.c_str(), retained);
  Serial.printf("[MQTT] TX %s => %s\n", topic, payload.c_str());
  return ok;
}

// ============================================================================
// 12) FLASK (POST automatique)
// ============================================================================

// Envoi du résultat EI + mesures vers Flask
bool sendToFlaskAuto(const char* label, float score, const char* reason) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(FLASK_URL_BASE) + String(EP_TRIGGER_AUTO);

  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);

  String body="{";
  body += "\"source\":\"esp32\",";
  body += "\"reason\":\"" + String(reason) + "\",";
  body += "\"label\":\"" + String(label) + "\",";
  body += "\"score\":" + String(score, 5) + ",";
  body += "\"bpm\":" + String(data.bpm, 1) + ",";
  body += "\"irAvg\":" + String((unsigned long)data.irAvg) + ",";
  body += "\"ax\":" + String(data.ax, 3) + ",";
  body += "\"ay\":" + String(data.ay, 3) + ",";
  body += "\"az\":" + String(data.az, 3) + ",";
  body += "\"time_ms\":" + String((unsigned long)millis());
  body += "}";

  int code = http.POST(body);
  String resp = http.getString();
  http.end();

  Serial.printf("[FLASK] POST %s => code=%d resp=%s\n", url.c_str(), code, resp.c_str());
  return (code >= 200 && code < 300);
}

// ============================================================================
// 13) MAX30105 (BPM + finger detect)
// ============================================================================

uint8_t ledAmp = 0x10;

// Ajustement automatique du gain selon IR average
void autoGainFromIRavg(uint32_t avg) {
  if (avg < 20000 && ledAmp < 0x3F) ledAmp += 1;
  else if (avg > 90000 && ledAmp > 0x05) ledAmp -= 1;

  max30105.setPulseAmplitudeIR(ledAmp);
  max30105.setPulseAmplitudeRed(ledAmp);
}

// Lecture MAX30105 + calcul BPM
void updateMax30105() {
  static uint32_t acc = 0;
  static int n = 0;
  static uint32_t lastAvgMs = 0;

  max30105.check();

  while (max30105.available()) {
    uint32_t ir = max30105.getIR();
    acc += ir;
    n++;

    if (checkForBeat(ir)) {
      uint32_t now = millis();
      if (data.lastBeatMs != 0) {
        uint32_t dt = now - data.lastBeatMs;

        if (dt >= 333 && dt <= 1500) {
          float bpmRaw = 60000.0f / (float)dt;

          if (data.bpmEma <= 0) data.bpmEma = bpmRaw;
          else data.bpmEma = 0.80f * data.bpmEma + 0.20f * bpmRaw;

          data.bpm = data.bpmEma;
        }
      }
      data.lastBeatMs = now;
    }

    max30105.nextSample();
  }

  // Finger detect via moyenne IR
  if (millis() - lastAvgMs >= 500) {
    lastAvgMs = millis();

    uint32_t avg = (n > 0) ? (acc / (uint32_t)n) : 0;
    acc = 0; n = 0;

    data.irAvg = avg;
    autoGainFromIRavg(avg);

    const uint32_t FINGER_ON  = 12000;
    const uint32_t FINGER_OFF = 8000;

    if (!data.finger && avg > FINGER_ON) {
      data.finger = true;
      data.bpm = 0; data.bpmEma = 0; data.lastBeatMs = 0;
      Serial.printf("[SENSOR] Finger ON avg=%lu led=0x%02X\n", (unsigned long)avg, ledAmp);
    } else if (data.finger && avg < FINGER_OFF) {
      data.finger = false;
      data.bpm = 0; data.bpmEma = 0; data.lastBeatMs = 0;
      Serial.printf("[SENSOR] Finger OFF avg=%lu led=0x%02X\n", (unsigned long)avg, ledAmp);
    }
  }
}

// ============================================================================
// 14) MPU6050 (accélération -> ax/ay/az en g)
// ============================================================================

void updateMotion() {
  sensors_event_t a, g, t;
  if (mpu.getEvent(&a, &g, &t)) {
    data.ax = a.acceleration.x / 9.81f;
    data.ay = a.acceleration.y / 9.81f;
    data.az = a.acceleration.z / 9.81f;
  }
}

// ============================================================================
// 15) DETECTION BPM ANORMAL + DEMARRAGE EI
// ============================================================================

// Détection BPM anormal (seuils)
bool isAbnormalBpm() {
  if (!data.finger) return false;
  if (data.bpm < 5) return false;
  return (data.bpm < config.bpmMin || data.bpm > config.bpmMax);
}

// Démarre la collecte d’une fenêtre IMU pour EI
void startAutoEiCollection() {
  autoState = AutoState::COLLECTING;
  ei_feat_ix = 0;
  lastEiSampleMs = 0;
  Serial.println("[AUTO] Start EI window collection...");
}

// Collecte IMU + inférence EI + envoi Flask
void processAutoEi() {
  if (autoState != AutoState::COLLECTING) return;

  uint32_t now = millis();
  if (now - lastEiSampleMs < eiSamplePeriodMs()) return;
  lastEiSampleMs = now;

  // Remplir buffer EI: ax, ay, az
  if (ei_feat_ix + 2 < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    ei_features[ei_feat_ix++] = data.ax;
    ei_features[ei_feat_ix++] = data.ay;
    ei_features[ei_feat_ix++] = data.az;
  }

  // Attendre fenêtre complète
  if (ei_feat_ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) return;

  // Lancer classification
  signal_t signal;
  int err = numpy::signal_from_buffer(ei_features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
    Serial.println("[EI] signal_from_buffer error");
    autoState = AutoState::IDLE;
    return;
  }

  ei_impulse_result_t result = {0};
  err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.printf("[EI] run_classifier error=%d\n", err);
    autoState = AutoState::IDLE;
    return;
  }

  // Choisir la classe avec la meilleure probabilité
  size_t best_i = 0;
  float best_v = result.classification[0].value;
  for (size_t i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > best_v) {
      best_v = result.classification[i].value;
      best_i = i;
    }
  }

  const char* best_label = result.classification[best_i].label;
  Serial.printf("[EI] pred=%s score=%.4f\n", best_label, best_v);

  // Envoyer vers Flask (pour déclencher traitement micro / IA audio)
  sendToFlaskAuto(best_label, best_v, "auto_abnormal_bpm");

  // Optionnel : si danger => MQTT panic/alert
  bool isDanger = (String(best_label) == EI_DANGER_LABEL) && (best_v >= EI_DANGER_MIN_SCORE);
  if (isDanger) {
    motorBuzz(60);
    mqttPublishChecked(TOPIC_PANIC_ALERT, jsonAlert("auto_abnormal_bpm+ei_danger"));
  }

  autoState = AutoState::IDLE;
}

// Déclenchement auto: BPM anormal => lancer EI (avec cooldown)
void autoTriggerIfNeeded() {
  if (autoState != AutoState::IDLE) return;
  if (!isAbnormalBpm()) return;

  uint32_t now = millis();
  if (now - data.lastAutoTriggerMs < AUTO_TRIGGER_COOLDOWN_MS) return;
  data.lastAutoTriggerMs = now;

  Serial.printf("[AUTO] Abnormal BPM %.1f => EI + Flask\n", data.bpm);
  startAutoEiCollection();
}

// ============================================================================
// 16) BOUTON (1 clic -> attente 2s -> panic, 2 clics -> cancel)
// ============================================================================

void handleButtonManual() {
  int raw = digitalRead(BUTTON_PIN);

  // Détection changement raw (bouncing)
  if (raw != lastRead) {
    lastRead = raw;
    lastChangeMs = millis();
  }

  // Validation état stable après debounce
  bool pressEvent = false;
  if ((millis() - lastChangeMs) > DEBOUNCE_MS && raw != stableBtn) {
    stableBtn = raw;
    if (stableBtn == LOW) pressEvent = true; // appui détecté
  }

  uint32_t now = millis();

  // Si on attend un 2e clic et 2s écoulées => PANIC
  if (waitingSecondClick && (now - firstClickMs >= DOUBLECLICK_WINDOW_MS)) {
    waitingSecondClick = false;
    motorBuzz(60);
    mqttPublishChecked(TOPIC_PANIC_ALERT, jsonAlert("manual_button"));
  }

  // Gestion de l'appui
  if (pressEvent) {
    if (now < lockoutUntilMs) {
      Serial.println("[CLICK] ignored (lockout)");
      return;
    }

    // Premier clic => on démarre la fenêtre 2s
    if (!waitingSecondClick) {
      waitingSecondClick = true;
      firstClickMs = now;
      motorBuzz(20);
      Serial.println("[CLICK] first click -> waiting 2s");
    }
    // Deuxième clic dans 2s => CANCEL
    else {
      waitingSecondClick = false;
      motorBuzz(20);
      motorBuzz(20);

      mqttPublishChecked(TOPIC_PANIC_CANCEL, "{\"action\":\"cancel\"}");
      Serial.println("[CLICK] second click -> CANCEL");

      lockoutUntilMs = now + LOCKOUT_MS;
    }
  }
}

// ============================================================================
// 17) OLED (affichage simple)
// ============================================================================

void oledStatus() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(0,0);
  display.print("MODE:");
  display.print(MODE_TEST ? "TEST" : "NORMAL");

  display.setCursor(0,12);
  display.print("WiFi:");
  display.print(WiFi.status()==WL_CONNECTED ? "OK" : "NO");

  display.setCursor(0,24);
  display.print("MQTT:");
  display.print(mqtt.connected() ? "OK" : "NO");

  display.setCursor(0,36);
  display.print("BPM:");
  display.print((int)data.bpm);
  display.print(" IR:");
  display.print((unsigned long)data.irAvg);

  display.setCursor(0,48);
  display.print("BTN2s:");
  display.print(waitingSecondClick ? "YES" : "NO");
  display.print(" AUTO:");
  display.print(autoState == AutoState::COLLECTING ? "EI" : "-");

  display.display();
}

// ============================================================================
// 18) WIFI CONNECT (reconnexion automatique)
// ============================================================================

void wifiEnsureConnected() {
  if (WiFi.status() == WL_CONNECTED) return;

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("[WIFI] Connecting");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 12000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  Serial.printf("[WIFI] Status=%d IP=%s\n", WiFi.status(), WiFi.localIP().toString().c_str());
}

// ============================================================================
// 19) SETUP / LOOP
// ============================================================================

void setup() {
  Serial.begin(115200);
  randomSeed(esp_random());

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  config = MODE_TEST ? cfgTest : cfgNormal;

  // Initialisation bus I2C
  I2C_Bus0.begin(SDA_I2C, SCL_I2C, 400000);

  // OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("[OLED] NOT FOUND");
    while(true) delay(100);
  }
  Serial.println("[OLED] OK");

  // MPU6050
  if (!mpu.begin(0x68, &I2C_Bus0)) {
    Serial.println("[MPU] NOT FOUND");
  } else {
    Serial.println("[MPU] OK");
  }

  // MAX30105
  if (!max30105.begin(I2C_Bus0, I2C_SPEED_FAST)) {
    Serial.println("[MAX30105] NOT FOUND");
    while(true) delay(100);
  }
  Serial.println("[MAX30105] OK");

  // Configuration MAX30105
  max30105.setup(0x10, 4, 2, 400, 411, 4096);
  max30105.setPulseAmplitudeIR(ledAmp);
  max30105.setPulseAmplitudeRed(ledAmp);

  // Connexion réseau
  wifiEnsureConnected();

  // MQTT
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqttEnsureConnected();

  // Initialisation bouton (debounce)
  stableBtn = digitalRead(BUTTON_PIN);
  lastRead  = stableBtn;
  lastChangeMs = millis();

  Serial.println("[SYSTEM] Ready (STABLE listening)");
  oledStatus();
}

void loop() {
  // Lecture capteurs en continu
  updateMax30105();
  updateMotion();

  // Déclenchement manuel (bouton)
  handleButtonManual();

  // Déclenchement auto (BPM anormal => EI => Flask)
  autoTriggerIfNeeded();
  processAutoEi();

  // Maintien WiFi/MQTT
  wifiEnsureConnected();
  mqttEnsureConnected();
  mqtt.loop();

  // Publication debug capteurs (optionnel)
  static uint32_t lastPub = 0;
  if (millis() - lastPub > SENSOR_PUBLISH_MS) {
    lastPub = millis();
    if (mqtt.connected()) mqtt.publish(TOPIC_SENSORS, jsonSensors().c_str());
  }

  // Affichage OLED
  static uint32_t lastUi = 0;
  if (millis() - lastUi > UI_EVERY_MS) {
    lastUi = millis();
    oledStatus();
  }

  delay(5);
}