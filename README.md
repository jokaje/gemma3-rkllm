# Gemma3 RKLLM - Multimodal AI auf Orange Pi 5 Plus

Eine vollständige, einfach zu verwendende Lösung für die Ausführung von Gemma3 multimodalen Modellen auf dem Orange Pi 5 Plus mit Rockchip NPU-Unterstützung.

## Überblick

Dieses Projekt bietet eine Flask-basierte Server-Lösung, die es ermöglicht, Gemma3-Modelle im RKLLM-Format auf der NPU des Orange Pi 5 Plus auszuführen. Die Lösung unterstützt sowohl Text- als auch multimodale Eingaben (Text + Bilder) und ist kompatibel mit der Ollama-API.

### Hauptfunktionen

- **Multimodale KI**: Unterstützung für Text- und Bildeingaben mit Gemma3
- **NPU-Optimierung**: Automatische Frequenzskalierung und Performance-Optimierung für RK3588
- **Einfache Installation**: Ein-Klick-Setup mit automatischer Konfiguration
- **API-Kompatibilität**: Ollama-kompatible REST-API für einfache Integration
- **Streaming-Unterstützung**: Echtzeit-Textgenerierung mit Server-Sent Events
- **Robuste Architektur**: Modularer Aufbau mit umfassendem Logging und Fehlerbehandlung

## Systemanforderungen

### Hardware
- **Orange Pi 5 Plus** mit RK3588-Prozessor
- **Mindestens 8GB RAM** (16GB empfohlen)
- **Mindestens 5GB freier Speicherplatz**
- **MicroSD-Karte oder eMMC** (Class 10 oder besser)

### Software
- **Armbian** oder **Ubuntu 22.04 LTS** für ARM64
- **Python 3.11** oder höher
- **Sudo-Berechtigung** für Systemkonfiguration

## Schnellstart

### 1. Projekt herunterladen

```bash
# Projekt in gewünschtes Verzeichnis kopieren
cd /home/your-username
# [Projekt-Dateien hier einfügen]
cd gemma3-rkllm
```

### 2. Automatische Installation

```bash
# Setup-Skript ausführen
./setup.sh
```

Das Setup-Skript führt automatisch folgende Schritte aus:
- Überprüfung der Systemkompatibilität
- Installation aller Systemabhängigkeiten
- Einrichtung der Python-Umgebung
- Konfiguration der NPU-Optimierung
- Erstellung der Systemdienste

### 3. RKLLM-Bibliothek installieren

```bash
# RKLLM-Bibliothek von Rockchip herunterladen
wget https://github.com/airockchip/rknn-llm/releases/latest/download/librkllmrt.so
# In lib-Verzeichnis verschieben
mv librkllmrt.so lib/
```

### 4. Modell installieren

```bash
# Modellverzeichnis erstellen
mkdir -p models/gemma3-4b

# Ihre .rkllm-Datei in das Modellverzeichnis kopieren
cp /path/to/your/gemma3-4b.rkllm models/gemma3-4b/

# Modelfile erstellen
cat > models/gemma3-4b/Modelfile << EOF
FROM="gemma3-4b.rkllm"
HUGGINGFACE_PATH="google/gemma-2-2b-it"
SYSTEM="You are a helpful AI assistant with vision capabilities."
TEMPERATURE=0.7
TOKENIZER="google/gemma-2-2b-it"
EOF
```

### 5. Server starten

```bash
# Server starten
./start.sh

# Oder mit benutzerdefinierten Parametern
./start.sh --host 0.0.0.0 --port 8080 --debug
```

### 6. Testen

```bash
# Einfacher Texttest
./client.py --prompt "Hallo, wie geht es dir?"

# Multimodaler Test mit Bild
./client.py --prompt "Beschreibe dieses Bild" --image /path/to/image.jpg

# Streaming-Test
./client.py --prompt "Erzähle mir eine Geschichte" --stream
```

## Projektstruktur

```
gemma3-rkllm/
├── server.py                 # Haupt-Flask-Server
├── setup.sh                  # Automatisches Setup-Skript
├── start.sh                  # Server-Start-Skript
├── stop.sh                   # Server-Stop-Skript
├── client.py                 # Test-Client
├── requirements.txt          # Python-Abhängigkeiten
├── config/
│   └── default.ini          # Hauptkonfiguration
├── src/                     # Quellcode-Module
│   ├── __init__.py
│   ├── gemma3_model.py      # Hauptmodellklasse
│   ├── rkllm_runtime.py     # RKLLM C++ Integration
│   ├── npu_optimizer.py     # NPU-Optimierung
│   ├── image_processor.py   # Bildverarbeitung
│   ├── config_manager.py    # Konfigurationsverwaltung
│   ├── api_handlers.py      # API-Handler
│   ├── logger.py            # Logging-System
│   └── utils.py             # Hilfsfunktionen
├── lib/                     # Bibliotheken
│   └── librkllmrt.so        # RKLLM Runtime (manuell installieren)
├── models/                  # Modellverzeichnis
│   └── [model-name]/
│       ├── model.rkllm      # RKLLM-Modelldatei
│       └── Modelfile        # Modellkonfiguration
├── logs/                    # Log-Dateien
├── docs/                    # Dokumentation
└── tests/                   # Tests
```

## Konfiguration

### Hauptkonfiguration (config/default.ini)

Die Konfiguration erfolgt über die Datei `config/default.ini`. Wichtige Parameter:

```ini
[server]
host = 0.0.0.0
port = 8080
debug = false

[model]
default_model = gemma3-4b
max_context_length = 128000
default_temperature = 0.7

[multimodal]
max_image_size = 2048
supported_formats = jpg,jpeg,png,webp,bmp

[npu]
platform = rk3588
frequency_mode = performance
enable_optimization = true
```

### Umgebungsvariablen

Konfiguration kann auch über Umgebungsvariablen erfolgen:

```bash
export GEMMA3_HOST=0.0.0.0
export GEMMA3_PORT=8080
export GEMMA3_DEBUG=false
export GEMMA3_MODELS_DIR=./models
export GEMMA3_MAX_CONTEXT=128000
export GEMMA3_TEMPERATURE=0.7
export GEMMA3_LOG_LEVEL=INFO
export GEMMA3_NPU_PLATFORM=rk3588
export GEMMA3_MAX_IMAGE_SIZE=2048
```

## API-Dokumentation

### Ollama-kompatible Endpunkte

#### Chat-Endpunkt (Multimodal)

```bash
POST /api/chat
Content-Type: application/json

{
  "model": "gemma3-4b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Was siehst du in diesem Bild?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
          }
        }
      ]
    }
  ],
  "stream": false
}
```

#### Generierungs-Endpunkt

```bash
POST /api/generate
Content-Type: application/json

{
  "model": "gemma3-4b",
  "prompt": "Erkläre mir Quantencomputing",
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Native Endpunkte

#### Modell laden

```bash
POST /load_model
Content-Type: application/json

{
  "model": "gemma3-4b"
}
```

#### Multimodale Generierung

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "Beschreibe dieses Bild detailliert",
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

#### Bild hochladen

```bash
POST /upload_image
Content-Type: multipart/form-data

# Form-Daten mit 'image' Feld
```

### Systemendpunkte

#### Gesundheitscheck

```bash
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Verfügbare Modelle

```bash
GET /models

Response:
{
  "models": [
    {
      "name": "gemma3-4b",
      "path": "./models/gemma3-4b",
      "multimodal": true,
      "size": "unknown"
    }
  ]
}
```

#### Konfiguration anzeigen

```bash
GET /config

Response:
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false
  },
  "model": {
    "max_context_length": 128000,
    "default_temperature": 0.7
  },
  "multimodal": {
    "max_image_size": 2048,
    "supported_formats": ["jpeg", "png", "webp", "bmp"]
  }
}
```

## NPU-Optimierung

Das System optimiert automatisch die NPU-Performance:

### Frequenzmodi

- **performance**: 1000 MHz (maximale Leistung)
- **balanced**: 600 MHz (ausgewogene Leistung/Verbrauch)
- **powersave**: 300 MHz (minimaler Verbrauch)

### Automatische Optimierungen

- NPU-Frequenzskalierung
- CPU-Governor-Einstellung
- Speicher-Optimierung
- Swap-Deaktivierung
- Scheduler-Optimierung

### Manuelle NPU-Konfiguration

```bash
# NPU-Frequenz prüfen
cat /sys/class/devfreq/fdab0000.npu/cur_freq

# NPU auf Performance-Modus setzen
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor

# CPU-Governor auf Performance setzen
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Multimodale Funktionen

### Unterstützte Bildformate

- JPEG/JPG
- PNG
- WebP
- BMP
- TIFF

### Bildvorverarbeitung

Das System führt automatisch folgende Vorverarbeitungsschritte durch:

1. **Format-Validierung**: Überprüfung des Bildformats
2. **Größenanpassung**: Intelligente Skalierung auf 384x384 Pixel (SigLIP-kompatibel)
3. **Normalisierung**: Anwendung der SigLIP-Normalisierung
4. **Verbesserung**: Automatischer Kontrast und Schärfung
5. **Feature-Extraktion**: Vorbereitung für das Vision-Modell

### Bildverarbeitung-Pipeline

```python
# Beispiel für die Bildverarbeitung
from src.image_processor import ImageProcessor

processor = ImageProcessor(config)

# Bild aus Datei verarbeiten
result = processor.process_image_file("image.jpg")

# Bild aus Base64 verarbeiten
result = processor.process_base64_image(base64_string)

# Batch-Verarbeitung
results = processor.batch_process_images([
    "image1.jpg",
    "image2.png",
    base64_string
])
```

## Systemdienst

### Service installieren

```bash
# Service aktivieren
sudo systemctl enable gemma3-rkllm

# Service starten
sudo systemctl start gemma3-rkllm

# Status prüfen
sudo systemctl status gemma3-rkllm

# Logs anzeigen
sudo journalctl -u gemma3-rkllm -f
```

### Service-Konfiguration

Die Service-Datei befindet sich unter `/etc/systemd/system/gemma3-rkllm.service`:

```ini
[Unit]
Description=Gemma3 RKLLM Server
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/gemma3-rkllm
Environment=PATH=/path/to/gemma3-rkllm/venv/bin
ExecStart=/path/to/gemma3-rkllm/venv/bin/python server.py --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Logging und Monitoring

### Log-Konfiguration

Logs werden in `logs/gemma3-rkllm.log` gespeichert mit automatischer Rotation:

- **Maximale Größe**: 10MB pro Datei
- **Backup-Anzahl**: 5 Dateien
- **Format**: Zeitstempel, Level, Datei:Zeile, Nachricht

### Log-Level

- **DEBUG**: Detaillierte Debugging-Informationen
- **INFO**: Allgemeine Informationen
- **WARNING**: Warnungen
- **ERROR**: Fehler
- **CRITICAL**: Kritische Fehler

### Performance-Monitoring

Das System protokolliert automatisch:

- Inferenz-Zeiten
- Token-Generierungsraten
- Speicherverbrauch
- NPU-Auslastung
- API-Request-Metriken

### Log-Beispiele

```bash
# Live-Logs anzeigen
tail -f logs/gemma3-rkllm.log

# Nur Fehler anzeigen
grep ERROR logs/gemma3-rkllm.log

# Performance-Metriken filtern
grep "Inference completed" logs/gemma3-rkllm.log
```

## Fehlerbehebung

### Häufige Probleme

#### 1. RKLLM-Bibliothek nicht gefunden

```
Error: RKLLM library not found
```

**Lösung**:
```bash
# Bibliothek von Rockchip herunterladen
wget https://github.com/airockchip/rknn-llm/releases/latest/download/librkllmrt.so
mv librkllmrt.so lib/
```

#### 2. NPU nicht erkannt

```
Warning: NPU devfreq not found
```

**Lösung**:
```bash
# NPU-Module prüfen
lsmod | grep npu

# Device-Tree prüfen
cat /proc/device-tree/compatible
```

#### 3. Speicher-Fehler

```
Error: Out of memory
```

**Lösung**:
```bash
# Swap deaktivieren
sudo swapoff -a

# Speicherverbrauch prüfen
free -h

# Modell-Kontext reduzieren
# In config/default.ini: max_context_length = 64000
```

#### 4. Port bereits belegt

```
Error: Address already in use
```

**Lösung**:
```bash
# Prozess finden und beenden
sudo lsof -i :8080
sudo kill -9 <PID>

# Oder anderen Port verwenden
./start.sh --port 8081
```

### Debug-Modus

```bash
# Server im Debug-Modus starten
./start.sh --debug

# Oder Umgebungsvariable setzen
export GEMMA3_DEBUG=true
./start.sh
```

### Systemdiagnose

```bash
# System-Informationen sammeln
python -c "
import sys
sys.path.insert(0, 'src')
from src.utils import get_system_info
from src.npu_optimizer import NPUOptimizer
from src.config_manager import ConfigManager

config = ConfigManager().load_config()
npu = NPUOptimizer(config)

print('System Info:', get_system_info())
print('NPU Status:', npu.get_optimization_status())
"
```

## Performance-Optimierung

### Hardware-Optimierungen

1. **NPU-Frequenz maximieren**:
```bash
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
```

2. **CPU-Kerne für Performance konfigurieren**:
```bash
# A76-Kerne (4-7) auf Performance setzen
for cpu in {4..7}; do
    echo performance | sudo tee /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor
done
```

3. **Speicher-Optimierung**:
```bash
# Swap deaktivieren
sudo swapoff -a

# VM-Parameter optimieren
echo 1 | sudo tee /proc/sys/vm/swappiness
echo 15 | sudo tee /proc/sys/vm/dirty_ratio
```

### Software-Optimierungen

1. **Kontext-Länge anpassen**:
```ini
# In config/default.ini
[model]
max_context_length = 64000  # Reduziert für bessere Performance
```

2. **Batch-Größe optimieren**:
```ini
[server]
max_workers = 2  # Reduziert für weniger Speicherverbrauch
```

3. **Bildverarbeitung optimieren**:
```ini
[multimodal]
max_image_size = 1024  # Kleinere Bilder für bessere Performance
image_quality = 75     # Reduzierte Qualität für weniger Speicher
```

### Benchmark-Tests

```bash
# Performance-Test durchführen
python -c "
import time
import requests

url = 'http://localhost:8080/api/generate'
prompt = 'Erkläre mir Quantencomputing in 100 Wörtern.'

start_time = time.time()
response = requests.post(url, json={'model': 'gemma3-4b', 'prompt': prompt})
end_time = time.time()

result = response.json()
print(f'Response time: {end_time - start_time:.2f}s')
print(f'Tokens per second: {result.get(\"tokens_per_second\", 0):.2f}')
"
```

## Entwicklung und Erweiterung

### Entwicklungsumgebung einrichten

```bash
# Entwicklungsabhängigkeiten installieren
pip install -r requirements.txt
pip install pytest black flake8

# Code-Formatierung
black src/ server.py

# Linting
flake8 src/ server.py

# Tests ausführen
pytest tests/
```

### Neue Modelle hinzufügen

1. **Modellverzeichnis erstellen**:
```bash
mkdir -p models/new-model
```

2. **Modelfile konfigurieren**:
```bash
cat > models/new-model/Modelfile << EOF
FROM="new-model.rkllm"
HUGGINGFACE_PATH="organization/model-name"
SYSTEM="Custom system prompt"
TEMPERATURE=0.8
TOKENIZER="organization/model-name"
EOF
```

3. **Modell-spezifische Anpassungen** in `src/gemma3_model.py`

### API erweitern

Neue Endpunkte können in `server.py` hinzugefügt werden:

```python
@app.route('/custom_endpoint', methods=['POST'])
def custom_endpoint():
    # Implementierung hier
    pass
```

### Monitoring erweitern

Neue Metriken können in `src/logger.py` hinzugefügt werden:

```python
def log_custom_metric(self, metric_name: str, value: float):
    self.logger.info(f"Custom metric - {metric_name}: {value}")
```

## Sicherheit

### Grundlegende Sicherheitsmaßnahmen

1. **Firewall konfigurieren**:
```bash
sudo ufw allow 8080/tcp
sudo ufw enable
```

2. **Reverse Proxy verwenden** (Nginx):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **API-Key-Authentifizierung aktivieren**:
```ini
# In config/default.ini
[security]
api_key_required = true
```

### Produktionsumgebung

Für den Produktionseinsatz sollten folgende Maßnahmen ergriffen werden:

1. **HTTPS verwenden**
2. **Rate Limiting aktivieren**
3. **Input-Validierung verstärken**
4. **Logging-Level auf INFO setzen**
5. **Regelmäßige Updates durchführen**

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe LICENSE-Datei für Details.

## Support und Community

### Dokumentation

- **GitHub Repository**: [Link zum Repository]
- **API-Dokumentation**: http://localhost:8080/health
- **Beispiele**: Siehe `examples/` Verzeichnis

### Probleme melden

Bei Problemen oder Fragen:

1. **GitHub Issues**: Für Bugs und Feature-Requests
2. **Diskussionen**: Für allgemeine Fragen
3. **Wiki**: Für zusätzliche Dokumentation

### Beitragen

Beiträge sind willkommen! Bitte:

1. Fork des Repositories erstellen
2. Feature-Branch erstellen
3. Änderungen committen
4. Pull Request erstellen

## Changelog

### Version 1.0.0 (Initial Release)

- Vollständige Gemma3 RKLLM-Integration
- Multimodale Unterstützung (Text + Bilder)
- NPU-Optimierung für Orange Pi 5 Plus
- Ollama-kompatible API
- Automatisches Setup-Skript
- Umfassende Dokumentation
- Systemdienst-Integration
- Performance-Monitoring
- Robuste Fehlerbehandlung

## Danksagungen

- **Rockchip** für die RKLLM-Bibliothek
- **Google** für das Gemma3-Modell
- **Orange Pi** für die Hardware-Plattform
- **Open Source Community** für die verwendeten Bibliotheken

---

**Autor**: Manus AI  
**Version**: 1.0.0  
**Datum**: 2024  
**Lizenz**: MIT

