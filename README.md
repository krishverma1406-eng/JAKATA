# JAKATA

JAKATA is a modular Python desktop AI assistant focused on practical personal productivity, voice interaction, memory, and local tool use.

## What It Includes

- Streaming chat with live LLM routing
- Voice output with Edge TTS
- Push-to-talk speech input and wake-word support
- Memory retrieval with lexical and vector recall
- Tool calling for:
  - reminders
  - clipboard
  - notes
  - calculator and unit conversion
  - system info
  - weather
  - Gmail
  - Google Calendar
  - screenshots
  - app launching
  - file management
  - web search
  - desktop control

## Project Layout

- [jarvis/main.py](./jarvis/main.py): CLI entry point
- [jarvis/core](./jarvis/core): agent loop, planner, memory, tool registry, provider routing
- [jarvis/tools](./jarvis/tools): tool modules exposed to the assistant
- [jarvis/services](./jarvis/services): helper services for TTS, STT, OAuth, reminders, OS control, and wake word
- [jarvis/data_ai](./jarvis/data_ai): prompt files that describe behavior, capabilities, and tool guidance
- [jarvis/config](./jarvis/config): settings and local environment configuration

## Quick Start

1. Install Python 3.11+.
2. Install dependencies from your own environment or requirements list.
3. Copy [jarvis/config/.env.example](./jarvis/config/.env.example) to `jarvis/config/.env`.
4. Fill in the provider and API keys you want to use.
5. Run:

```powershell
cd jarvis
python .\main.py
```

## Notes

- This public repo intentionally excludes local secrets, OAuth tokens, user memory data, screenshots, and runtime logs.
- Gmail and Calendar require Google OAuth setup.
- Weather requires an OpenWeatherMap API key.
- Wake-word and voice features depend on local audio setup.
