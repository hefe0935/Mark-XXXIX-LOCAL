# MARK XXXIX-OR

# MARK XXXIX-OR Local Fork
A personal desktop assistant forked from the original MARK XXXIX-OR project. This version keeps the Jarvis-style voice interface, desktop actions, file tools, memory, web helpers, and planning workflow, while preferring a local Ollama model for normal LLM work.

The goal of this fork is simple: keep the assistant practical, easier to run, and less dependent on hosted model quotas.

## What Changed In This Fork

- Local-first LLM calls through Ollama, using `qwen3:8b` by default.
- OpenRouter is now only a fallback for modules that use `or_client.py`.
- The setup screen can enable Ollama and choose one of the local models already installed.

Gemini Live is still available when Ollama is disabled and a Gemini key is saved. When Ollama is enabled, startup does not require a Gemini key and local responses are spoken with the operating system voice.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11
- Ollama running locally
- A local Ollama model, recommended: `qwen3:8b`
- Gemini API key only when using Gemini Live mode
- Optional OpenRouter key for fallback remote model calls
- Microphone access for voice mode

## Local Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install
```

Install and start the local model:

```powershell
ollama pull qwen3:8b
ollama serve
```

Run the assistant:

```powershell
python main.py
```

## Local AI Settings

The local client reads these environment variables:

```powershell
$env:USE_OLLAMA = "1"
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "qwen3:8b"
```

Those are already the defaults. The first startup screen also saves the Ollama choice and selected local model into `config/api_keys.json`. Environment variables still override the saved values when set.

To use OpenRouter as fallback, add this to `config/api_keys.json`:

```json
{
  "gemini_api_key": "your-gemini-key",
  "openrouter_api_key": "your-openrouter-key",
  "os_system": "windows",
  "use_ollama": true,
  "ollama_host": "http://127.0.0.1:11434",
  "ollama_model": "qwen3:8b"
}
```

For local mode, `gemini_api_key` and `openrouter_api_key` can be left blank. Gemini is only needed when `use_ollama` is false and the app should use Gemini Live.

## Notes

Some OS-specific packages can be picky. If a dependency fails during install, check the package name and Python version first. `pyaudio`, `pycaw`, `comtypes`, and `pywinauto` are the usual Windows-sensitive ones.

## License

This fork follows the original project's personal and non-commercial use terms under Creative Commons BY-NC 4.0.
