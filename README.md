# Ollama Llama3.2:1b Quick Setup Guide for Ubuntu

## Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
sudo systemctl start ollama
sudo systemctl enable ollama
```

## Configure for Remote Access (Optional)

```bash
# Edit systemd service for external connections
sudo systemctl edit ollama
```

Add this content:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

```bash
# Restart service
sudo systemctl restart ollama

# Allow firewall access
sudo ufw allow 11434
```

## Install Llama3.2:1b Model

```bash
# Download Llama3.2:1b model (~1.3GB)
ollama pull llama3.2:1b

# Test the model
ollama run llama3.2:1b "Hello, how are you?"
```

## Python Usage with LangExtract

```bash
# Install LangExtract
pip install langextract[ollama]
```

```python
import langextract as lx

# Your extraction code
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="llama3.2:1b",  # Automatically selects Ollama provider
    model_url="http://159.223.114.164:11434"
)
```

## Verify Installation

```bash
# Check status
ollama list
sudo systemctl status ollama

# Test API
curl http://localhost:11434/api/tags
```

**Done!** Llama3.2:1b is ready to use with LangExtract.
# langextract_ollama
