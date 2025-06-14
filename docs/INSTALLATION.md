# Installation Guide - Gemma3 RKLLM

Detaillierte Installationsanleitung für Gemma3 RKLLM auf Orange Pi 5 Plus.

## Voraussetzungen prüfen

### Hardware-Anforderungen

Bevor Sie mit der Installation beginnen, stellen Sie sicher, dass Ihr System die folgenden Anforderungen erfüllt:

**Orange Pi 5 Plus Spezifikationen:**
- **Prozessor**: Rockchip RK3588 (4x Cortex-A76 + 4x Cortex-A55)
- **NPU**: 6 TOPS AI-Beschleuniger
- **RAM**: Mindestens 8GB (16GB empfohlen)
- **Speicher**: Mindestens 5GB freier Speicherplatz
- **Betriebssystem**: Armbian oder Ubuntu 22.04 LTS für ARM64

### Software-Voraussetzungen

```bash
# System-Informationen prüfen
uname -a
cat /proc/cpuinfo | grep -i rk3588
free -h
df -h
```

**Erwartete Ausgabe:**
```
Linux orangepi5plus 5.10.110-rockchip-rk3588 #1 SMP aarch64 GNU/Linux
```

## Schritt-für-Schritt Installation

### Schritt 1: System vorbereiten

```bash
# System aktualisieren
sudo apt update && sudo apt upgrade -y

# Grundlegende Tools installieren
sudo apt install -y git wget curl unzip build-essential
```

### Schritt 2: Projekt herunterladen

```bash
# In Home-Verzeichnis wechseln
cd ~

# Projekt-Verzeichnis erstellen (falls nicht vorhanden)
# [Hier würden Sie normalerweise das Projekt klonen oder herunterladen]
# Für diese Anleitung nehmen wir an, dass die Dateien bereits vorhanden sind
cd gemma3-rkllm
```

### Schritt 3: Automatische Installation ausführen

```bash
# Setup-Skript ausführbar machen
chmod +x setup.sh

# Installation starten
./setup.sh
```

Das Setup-Skript führt automatisch folgende Aktionen durch:

1. **Plattform-Erkennung**: Überprüfung der RK3588-Kompatibilität
2. **Abhängigkeiten**: Installation aller erforderlichen Pakete
3. **Python-Umgebung**: Erstellung einer virtuellen Umgebung
4. **NPU-Konfiguration**: Optimierung der NPU-Einstellungen
5. **Systemdienste**: Erstellung von Systemd-Services

### Schritt 4: RKLLM-Bibliothek installieren

Die RKLLM-Bibliothek muss separat von Rockchip bezogen werden:

```bash
# Bibliothek herunterladen (Beispiel-URL)
wget https://github.com/airockchip/rknn-llm/releases/latest/download/librkllmrt.so

# In lib-Verzeichnis verschieben
mv librkllmrt.so lib/

# Berechtigung setzen
chmod 755 lib/librkllmrt.so
```

**Hinweis**: Die genaue URL kann sich ändern. Besuchen Sie das offizielle Rockchip-Repository für die neueste Version.

### Schritt 5: Modell vorbereiten

#### Modellverzeichnis erstellen

```bash
# Beispiel für Gemma3-4B Modell
mkdir -p models/gemma3-4b
```

#### Modelfile konfigurieren

```bash
cat > models/gemma3-4b/Modelfile << EOF
FROM="gemma3-4b.rkllm"
HUGGINGFACE_PATH="google/gemma-2-2b-it"
SYSTEM="You are a helpful AI assistant with vision capabilities. You can understand and analyze images as well as engage in text conversations."
TEMPERATURE=0.7
TOKENIZER="google/gemma-2-2b-it"
EOF
```

#### RKLLM-Modell installieren

```bash
# Ihre .rkllm-Datei in das Modellverzeichnis kopieren
# Beispiel:
cp /path/to/your/gemma3-4b.rkllm models/gemma3-4b/
```

**Hinweis**: Sie müssen Ihr Gemma3-Modell zunächst in das RKLLM-Format konvertieren. Anleitungen dazu finden Sie in der Rockchip-Dokumentation.

## Installation verifizieren

### Schritt 1: Grundfunktionen testen

```bash
# Python-Umgebung aktivieren
source venv/bin/activate

# Grundlegende Imports testen
python -c "
import sys
sys.path.insert(0, 'src')
from src import ConfigManager, ImageProcessor, NPUOptimizer
print('✓ Alle Module erfolgreich importiert')
"
```

### Schritt 2: Konfiguration prüfen

```bash
# Konfiguration laden und validieren
python -c "
import sys
sys.path.insert(0, 'src')
from src.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config('config/default.ini')

if config_manager.validate_config():
    print('✓ Konfiguration ist gültig')
else:
    print('✗ Konfigurationsfehler')
"
```

### Schritt 3: NPU-Status prüfen

```bash
# NPU-Verfügbarkeit prüfen
python -c "
import sys
sys.path.insert(0, 'src')
from src.npu_optimizer import NPUOptimizer
from src.config_manager import ConfigManager

config = ConfigManager().load_config()
npu = NPUOptimizer(config)
status = npu.get_optimization_status()

print(f'NPU Platform: {status.get(\"platform\", \"unknown\")}')
print(f'Frequency Mode: {status.get(\"frequency_mode\", \"unknown\")}')
print(f'Current NPU Frequency: {status.get(\"current_npu_frequency\", \"unknown\")}')
"
```

## Server starten und testen

### Server starten

```bash
# Server im Vordergrund starten (für Tests)
./start.sh

# Oder im Hintergrund
nohup ./start.sh > server.log 2>&1 &
```

### Grundlegende Tests

#### 1. Gesundheitscheck

```bash
curl http://localhost:8080/health
```

**Erwartete Antwort:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_loaded": false,
  "version": "1.0.0"
}
```

#### 2. Modell laden

```bash
curl -X POST http://localhost:8080/load_model \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3-4b"}'
```

#### 3. Textgenerierung testen

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3-4b",
    "prompt": "Hallo, wie geht es dir?",
    "stream": false
  }'
```

#### 4. Client-Skript verwenden

```bash
# Einfacher Test
./client.py --prompt "Erkläre mir Quantencomputing"

# Mit Streaming
./client.py --prompt "Erzähle mir eine Geschichte" --stream
```

## Systemdienst konfigurieren

### Service aktivieren

```bash
# Service für automatischen Start aktivieren
sudo systemctl enable gemma3-rkllm

# Service starten
sudo systemctl start gemma3-rkllm

# Status prüfen
sudo systemctl status gemma3-rkllm
```

### Service-Logs überwachen

```bash
# Live-Logs anzeigen
sudo journalctl -u gemma3-rkllm -f

# Letzte 100 Zeilen
sudo journalctl -u gemma3-rkllm -n 100
```

## Häufige Installationsprobleme

### Problem 1: Python-Version zu alt

**Symptom:**
```
Error: Python 3.11 or higher required
```

**Lösung:**
```bash
# Python 3.11 aus Deadsnakes PPA installieren (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Problem 2: Unzureichende Berechtigungen

**Symptom:**
```
Permission denied: /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

**Lösung:**
```bash
# Benutzer zur sudo-Gruppe hinzufügen
sudo usermod -aG sudo $USER

# Oder spezifische Berechtigung für NPU-Konfiguration
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/devices/system/cpu/*/cpufreq/scaling_governor" | sudo tee /etc/sudoers.d/npu-config
```

### Problem 3: Speicher-Fehler während Installation

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Lösung:**
```bash
# Swap-Datei erstellen (temporär)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Nach Installation wieder deaktivieren
sudo swapoff /swapfile
sudo rm /swapfile
```

### Problem 4: NPU nicht erkannt

**Symptom:**
```
Warning: NPU devfreq not found
```

**Lösung:**
```bash
# Kernel-Module prüfen
lsmod | grep npu

# Device-Tree prüfen
ls -la /sys/class/devfreq/

# Falls NPU-Module fehlen, Kernel neu kompilieren oder aktualisieren
```

## Performance-Optimierung nach Installation

### NPU-Frequenz optimieren

```bash
# Aktuelle NPU-Frequenz prüfen
cat /sys/class/devfreq/fdab0000.npu/cur_freq

# Verfügbare Frequenzen anzeigen
cat /sys/class/devfreq/fdab0000.npu/available_frequencies

# Performance-Modus aktivieren
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
```

### CPU-Konfiguration optimieren

```bash
# CPU-Governor für alle Kerne auf Performance setzen
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done

# CPU-Frequenzen prüfen
cat /proc/cpuinfo | grep "cpu MHz"
```

### Speicher-Optimierung

```bash
# Swap permanent deaktivieren
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# VM-Parameter optimieren
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Nächste Schritte

Nach erfolgreicher Installation können Sie:

1. **Weitere Modelle hinzufügen**: Siehe Abschnitt "Modell-Management"
2. **API-Integration**: Verwenden Sie die Ollama-kompatible API
3. **Multimodale Tests**: Testen Sie Bild-Text-Kombinationen
4. **Performance-Tuning**: Optimieren Sie die Konfiguration für Ihre Anwendung
5. **Produktionsumgebung**: Konfigurieren Sie Reverse Proxy und Sicherheit

## Support

Bei Problemen während der Installation:

1. **Logs prüfen**: `tail -f logs/gemma3-rkllm.log`
2. **System-Status**: `./start.sh --debug`
3. **Community**: GitHub Issues oder Diskussionen
4. **Dokumentation**: Siehe README.md für detaillierte Informationen

---

**Hinweis**: Diese Anleitung wird regelmäßig aktualisiert. Prüfen Sie die neueste Version im Repository.

