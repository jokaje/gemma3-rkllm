# API Documentation - Gemma3 RKLLM

Vollständige API-Dokumentation für den Gemma3 RKLLM Server.

## Übersicht

Der Gemma3 RKLLM Server bietet eine RESTful API mit Unterstützung für:

- **Text-Generierung**: Klassische LLM-Funktionalität
- **Multimodale Eingabe**: Text + Bild-Kombinationen
- **Streaming**: Echtzeit-Token-Generierung
- **Ollama-Kompatibilität**: Drop-in-Ersatz für Ollama
- **Modell-Management**: Dynamisches Laden/Entladen von Modellen

## Base URL

```
http://localhost:8080
```

## Authentifizierung

Standardmäßig ist keine Authentifizierung erforderlich. Für Produktionsumgebungen kann API-Key-Authentifizierung aktiviert werden:

```bash
# In config/default.ini
[security]
api_key_required = true
```

Bei aktivierter Authentifizierung:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8080/api/generate
```

## Content-Type

Alle POST-Requests verwenden:
```
Content-Type: application/json
```

Ausnahme: Datei-Uploads verwenden:
```
Content-Type: multipart/form-data
```

## Ollama-kompatible Endpunkte

### POST /api/chat

Ollama-kompatible Chat-API mit multimodaler Unterstützung.

#### Request

```json
{
  "model": "gemma3-4b",
  "messages": [
    {
      "role": "system",
      "content": "Du bist ein hilfreicher KI-Assistent mit Bildverständnis."
    },
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
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 1000
  }
}
```

#### Response (Non-Streaming)

```json
{
  "model": "gemma3-4b",
  "created_at": "2024-01-15T10:30:00.000Z",
  "message": {
    "role": "assistant",
    "content": "Ich sehe ein Bild von einem Orange Pi 5 Plus Single-Board-Computer. Das Board zeigt verschiedene Anschlüsse wie USB, HDMI, Ethernet und GPIO-Pins. Es ist ein kompakter Computer, der für KI-Anwendungen und Edge-Computing geeignet ist."
  },
  "done": true,
  "total_duration": 2500000000,
  "load_duration": 0,
  "prompt_eval_count": 45,
  "prompt_eval_duration": 500000000,
  "eval_count": 67,
  "eval_duration": 2000000000
}
```

#### Response (Streaming)

Bei `"stream": true` werden Server-Sent Events gesendet:

```
data: {"model":"gemma3-4b","created_at":"2024-01-15T10:30:00.000Z","message":{"role":"assistant","content":"Ich"},"done":false}

data: {"model":"gemma3-4b","created_at":"2024-01-15T10:30:00.000Z","message":{"role":"assistant","content":" sehe"},"done":false}

data: {"model":"gemma3-4b","created_at":"2024-01-15T10:30:00.000Z","message":{"role":"assistant","content":" ein"},"done":false}

...

data: {"model":"gemma3-4b","created_at":"2024-01-15T10:30:00.000Z","message":{"role":"assistant","content":""},"done":true,"total_duration":2500000000}
```

### POST /api/generate

Ollama-kompatible Generierungs-API.

#### Request

```json
{
  "model": "gemma3-4b",
  "prompt": "Erkläre mir die Funktionsweise einer NPU in einfachen Worten.",
  "stream": false,
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  ],
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 500,
    "stop": ["\n\n", "###"]
  }
}
```

#### Response

```json
{
  "model": "gemma3-4b",
  "created_at": "2024-01-15T10:30:00.000Z",
  "response": "Eine NPU (Neural Processing Unit) ist ein spezieller Chip, der für künstliche Intelligenz optimiert ist. Im Gegensatz zu normalen Prozessoren (CPU) oder Grafikkarten (GPU) ist die NPU speziell dafür entwickelt, neuronale Netzwerke sehr effizient auszuführen...",
  "done": true,
  "context": [1, 2, 3, 4, 5],
  "total_duration": 1800000000,
  "load_duration": 0,
  "prompt_eval_count": 23,
  "prompt_eval_duration": 300000000,
  "eval_count": 89,
  "eval_duration": 1500000000
}
```

## Native Endpunkte

### GET /health

Gesundheitscheck des Servers.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### GET /models

Liste aller verfügbaren Modelle.

#### Response

```json
{
  "models": [
    {
      "name": "gemma3-4b",
      "path": "./models/gemma3-4b",
      "multimodal": true,
      "size": "4.2GB"
    },
    {
      "name": "gemma3-9b", 
      "path": "./models/gemma3-9b",
      "multimodal": true,
      "size": "9.1GB"
    }
  ]
}
```

### POST /load_model

Lädt ein Modell in den Speicher.

#### Request

```json
{
  "model": "gemma3-4b"
}
```

#### Response

```json
{
  "message": "Model gemma3-4b loaded successfully",
  "model": "gemma3-4b",
  "multimodal": true
}
```

### POST /unload_model

Entlädt das aktuelle Modell aus dem Speicher.

#### Response

```json
{
  "message": "Model unloaded successfully"
}
```

### POST /generate

Native Generierungs-API mit erweiterten Funktionen.

#### Request

```json
{
  "prompt": "Beschreibe die Vorteile von Edge-AI-Computing",
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "top_p": 0.9,
  "top_k": 40,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["###", "\n\n"],
  "stream": false
}
```

#### Response

```json
{
  "text": "Edge-AI-Computing bietet mehrere entscheidende Vorteile: Erstens ermöglicht es Echtzeitverarbeitung direkt am Entstehungsort der Daten, was Latenzzeiten drastisch reduziert...",
  "prompt_tokens": 15,
  "completion_tokens": 234,
  "total_tokens": 249,
  "generation_time": 2.34,
  "tokens_per_second": 100.0,
  "model": "gemma3-4b",
  "multimodal": true,
  "finish_reason": "stop"
}
```

### POST /upload_image

Lädt ein Bild für multimodale Verarbeitung hoch.

#### Request (multipart/form-data)

```bash
curl -X POST http://localhost:8080/upload_image \
  -F "image=@/path/to/image.jpg"
```

#### Response

```json
{
  "message": "Image processed successfully",
  "image_id": "img_1642234567_abc123",
  "size": [1920, 1080],
  "format": "JPEG"
}
```

### GET /config

Zeigt die aktuelle Serverkonfiguration.

#### Response

```json
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

## Parameter-Referenz

### Generierungs-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `temperature` | float | 0.7 | Kreativität der Antworten (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus Sampling (0.0-1.0) |
| `top_k` | integer | 40 | Top-K Sampling |
| `max_tokens` | integer | 2048 | Maximale Anzahl generierter Tokens |
| `frequency_penalty` | float | 0.0 | Bestrafung für häufige Tokens (-2.0 bis 2.0) |
| `presence_penalty` | float | 0.0 | Bestrafung für bereits verwendete Tokens (-2.0 bis 2.0) |
| `stop` | array | [] | Stop-Sequenzen |
| `stream` | boolean | false | Streaming-Modus aktivieren |

### Multimodale Parameter

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `images` | array | Array von Base64-kodierten Bildern |
| `image_url` | object | Objekt mit URL-Feld für Bilder |
| `max_image_size` | integer | Maximale Bildgröße in Pixeln |

## Fehler-Codes

### HTTP Status Codes

| Code | Bedeutung |
|------|-----------|
| 200 | Erfolg |
| 400 | Ungültige Anfrage |
| 404 | Endpunkt nicht gefunden |
| 500 | Interner Serverfehler |
| 503 | Service nicht verfügbar |

### Fehler-Antworten

```json
{
  "error": {
    "code": "model_not_loaded",
    "message": "No model is currently loaded",
    "timestamp": 1642234567.123
  }
}
```

### Häufige Fehler

#### Modell nicht geladen

```json
{
  "error": "No model loaded"
}
```

**Lösung**: Laden Sie zuerst ein Modell mit `/load_model`

#### Ungültiges Bildformat

```json
{
  "error": "Unsupported image format"
}
```

**Lösung**: Verwenden Sie JPEG, PNG, WebP oder BMP

#### Kontext zu lang

```json
{
  "error": "Context length exceeds maximum"
}
```

**Lösung**: Reduzieren Sie die Eingabelänge oder `max_tokens`

## Code-Beispiele

### Python

```python
import requests
import base64
import json

# Server-URL
BASE_URL = "http://localhost:8080"

# Modell laden
def load_model(model_name):
    response = requests.post(f"{BASE_URL}/load_model", 
                           json={"model": model_name})
    return response.json()

# Text-Generierung
def generate_text(prompt, temperature=0.7):
    response = requests.post(f"{BASE_URL}/api/generate",
                           json={
                               "model": "gemma3-4b",
                               "prompt": prompt,
                               "temperature": temperature
                           })
    return response.json()["response"]

# Multimodale Generierung
def generate_with_image(prompt, image_path):
    # Bild laden und kodieren
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = requests.post(f"{BASE_URL}/generate",
                           json={
                               "prompt": prompt,
                               "images": [f"data:image/jpeg;base64,{image_data}"]
                           })
    return response.json()["text"]

# Streaming-Generierung
def generate_stream(prompt):
    response = requests.post(f"{BASE_URL}/api/generate",
                           json={
                               "model": "gemma3-4b", 
                               "prompt": prompt,
                               "stream": True
                           },
                           stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                print(data.get("response", ""), end="", flush=True)

# Beispiel-Verwendung
if __name__ == "__main__":
    # Modell laden
    load_model("gemma3-4b")
    
    # Einfache Textgenerierung
    result = generate_text("Erkläre mir maschinelles Lernen")
    print(result)
    
    # Multimodale Generierung
    result = generate_with_image("Was siehst du in diesem Bild?", "image.jpg")
    print(result)
    
    # Streaming
    generate_stream("Erzähle mir eine Geschichte über KI")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

const BASE_URL = 'http://localhost:8080';

// Modell laden
async function loadModel(modelName) {
    const response = await axios.post(`${BASE_URL}/load_model`, {
        model: modelName
    });
    return response.data;
}

// Text-Generierung
async function generateText(prompt, temperature = 0.7) {
    const response = await axios.post(`${BASE_URL}/api/generate`, {
        model: 'gemma3-4b',
        prompt: prompt,
        temperature: temperature
    });
    return response.data.response;
}

// Multimodale Generierung
async function generateWithImage(prompt, imagePath) {
    const imageBuffer = fs.readFileSync(imagePath);
    const imageBase64 = imageBuffer.toString('base64');
    
    const response = await axios.post(`${BASE_URL}/generate`, {
        prompt: prompt,
        images: [`data:image/jpeg;base64,${imageBase64}`]
    });
    return response.data.text;
}

// Chat-API verwenden
async function chat(messages) {
    const response = await axios.post(`${BASE_URL}/api/chat`, {
        model: 'gemma3-4b',
        messages: messages
    });
    return response.data.message.content;
}

// Beispiel-Verwendung
async function main() {
    try {
        // Modell laden
        await loadModel('gemma3-4b');
        
        // Einfache Textgenerierung
        const result1 = await generateText('Was ist künstliche Intelligenz?');
        console.log(result1);
        
        // Chat-Konversation
        const messages = [
            { role: 'system', content: 'Du bist ein hilfreicher Assistent.' },
            { role: 'user', content: 'Hallo! Wie geht es dir?' }
        ];
        const result2 = await chat(messages);
        console.log(result2);
        
        // Multimodale Generierung
        const result3 = await generateWithImage('Beschreibe dieses Bild', 'image.jpg');
        console.log(result3);
        
    } catch (error) {
        console.error('Fehler:', error.response?.data || error.message);
    }
}

main();
```

### cURL-Beispiele

```bash
# Modell laden
curl -X POST http://localhost:8080/load_model \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3-4b"}'

# Einfache Textgenerierung
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3-4b",
    "prompt": "Erkläre mir Quantencomputing",
    "temperature": 0.7
  }'

# Streaming-Generierung
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3-4b",
    "prompt": "Erzähle mir eine Geschichte",
    "stream": true
  }'

# Chat mit multimodaler Eingabe
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
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
    ]
  }'

# Bild hochladen
curl -X POST http://localhost:8080/upload_image \
  -F "image=@/path/to/image.jpg"

# Gesundheitscheck
curl http://localhost:8080/health

# Verfügbare Modelle
curl http://localhost:8080/models

# Konfiguration anzeigen
curl http://localhost:8080/config
```

## Rate Limiting

Der Server unterstützt Rate Limiting zur Vermeidung von Überlastung:

```ini
# In config/default.ini
[api]
rate_limit = 100  # Requests pro Minute
```

Bei Überschreitung des Limits:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

## WebSocket-Unterstützung

Für Echtzeit-Anwendungen kann WebSocket verwendet werden:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'generate',
        prompt: 'Erzähle mir eine Geschichte',
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data.text);
};
```

## Monitoring und Metriken

### Performance-Metriken

```bash
# Aktuelle Performance abrufen
curl http://localhost:8080/metrics
```

```json
{
  "requests_total": 1234,
  "requests_per_minute": 45,
  "average_response_time": 2.34,
  "tokens_generated_total": 56789,
  "average_tokens_per_second": 98.5,
  "model_load_time": 12.3,
  "memory_usage_mb": 4567,
  "npu_utilization": 0.85
}
```

### Health Checks

```bash
# Detaillierter Gesundheitscheck
curl http://localhost:8080/health/detailed
```

```json
{
  "status": "healthy",
  "components": {
    "model": "loaded",
    "npu": "optimized", 
    "memory": "sufficient",
    "disk": "sufficient"
  },
  "metrics": {
    "uptime": 3600,
    "requests_handled": 1234,
    "errors": 5
  }
}
```

---

Diese API-Dokumentation wird regelmäßig aktualisiert. Für die neueste Version besuchen Sie das GitHub-Repository.

