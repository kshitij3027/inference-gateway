#!/bin/bash
set -e

# Start Ollama server in the background
ollama serve &

# Wait for Ollama to be ready (up to 30 seconds)
echo "Waiting for Ollama server to start..."
for i in $(seq 1 30); do
    if ollama list >/dev/null 2>&1; then
        echo "Ollama server is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama server failed to start after 30 seconds."
        exit 1
    fi
    echo "Attempt $i/30: Ollama not ready yet..."
    sleep 1
done

# Pull TinyLlama model
echo "Pulling TinyLlama model..."
ollama pull tinyllama
echo "TinyLlama model pulled successfully."

# Keep the container running (wait for background ollama serve)
wait
