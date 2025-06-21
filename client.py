#!/usr/bin/env python3
"""
Conversational client for Gemma3 RKLLM server
Manages chat history to provide context and memory.
"""

import requests
import json
import sys
import argparse
import base64
from pathlib import Path

# Globale Variable für die Konversationshistorie
conversation_history = []

def send_chat_request(url, model="gemma3-4b-2", stream=True):
    """
    Sendet eine Chat-Anfrage mit der gesamten Konversationshistorie.
    """
    global conversation_history
    
    data = {
        "model": model,
        "messages": conversation_history,
        "stream": stream,
        "options": {
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(f"{url}/api/chat", json=data, stream=stream)
        response.raise_for_status()
        
        full_response = ""
        if stream:
            for line in response.iter_lines():
                if line:
                    try:
                        # Jede Zeile ist ein separates JSON-Objekt
                        chunk = json.loads(line.decode('utf-8'))
                        
                        # Non-Streaming-Fehler abfangen
                        if "error" in chunk:
                            print(f"\n[Server-Fehler: {chunk['error']}]")
                            return

                        # Streaming-Logik
                        if not chunk.get("done"):
                            content = chunk.get("message", {}).get("content", "")
                            print(content, end="", flush=True)
                            full_response += content
                    except json.JSONDecodeError:
                        print(f"\n[Warnung: Konnte eine Zeile nicht dekodieren: {line}]")
                        continue
            print() # Für einen Zeilenumbruch am Ende der gestreamten Antwort
        else:
            result = response.json()
            full_response = result.get("message", {}).get("content", "")
            print(full_response)

        # Füge die Antwort des Assistenten zur Historie hinzu, um das "Gedächtnis" zu wahren
        if full_response:
            conversation_history.append({"role": "assistant", "content": full_response})

    except requests.exceptions.RequestException as e:
        print(f"\n[Fehler bei der Anfrage: {e}]")
        # Bei einem Fehler die letzte Benutzer-Nachricht entfernen, um Wiederholungen zu vermeiden
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

def main():
    parser = argparse.ArgumentParser(description="Gemma3 RKLLM Conversational Client")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--model", default="gemma3-4b-2", help="Model name as defined on the server")
    
    args = parser.parse_args()
    
    print("Starte Konversation. Gib 'exit' oder 'quit' ein, um den Client zu beenden.")
    
    while True:
        try:
            prompt = input("Du: ")
            if prompt.lower() in ["exit", "quit"]:
                print("Beende Client...")
                break
            
            # Füge die neue Benutzernachricht zur Historie hinzu
            conversation_history.append({"role": "user", "content": prompt})
            
            # Sende die gesamte Historie an den Server
            print("AI: ", end="", flush=True)
            send_chat_request(args.url, args.model)
            
        except KeyboardInterrupt:
            print("\nBeende Client...")
            break
        except requests.exceptions.ConnectionError:
            print(f"\n[Fehler: Verbindung zum Server unter {args.url} fehlgeschlagen. Läuft der Server?]")
            sys.exit(1)
        except Exception as e:
            print(f"\n[Ein unerwarteter Fehler ist aufgetreten: {e}]")
            break

if __name__ == "__main__":
    main()
