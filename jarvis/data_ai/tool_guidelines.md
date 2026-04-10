## Tool Selection Reference

| Tool | Trigger Words | Requires Setup | Fallback |
|---|---|---|---|
| `web_search` | news, current, price, recent, latest | Tavily or Brave API key | none |
| `datetime_tool` | time, date, day, timezone | none | none |
| `file_manager` | read, write, list, find, copy, move, delete, PDF, docx | none | terminal_tool |
| `browser_control` | open website, click, type in browser, extract page | Nova Act API key | web_search |
| `vision_tool` | what is this, what am i holding, what do you see, describe this image, analyze camera frame | live camera frame or attached image in request, plus Gemini API key | STOP if no image is attached |
| `memory_query` | remember, who am I, what do you know, forget, my project | none | none |
| `reminder_tool` | remind me, set reminder, list reminders | none | notes_tool |
| `clipboard_tool` | clipboard, copy this, paste | none | none |
| `notes_tool` | save note, create note, find note | none | none |
| `calculator_tool` | calculate, math, convert units | none | inline math |
| `system_info_tool` | CPU, RAM, battery, disk, processes | psutil installed | none |
| `weather_tool` | weather, forecast, temperature | OpenWeatherMap key | web_search |
| `gmail_tool` | email, inbox, send mail | Gmail OAuth | STOP if not set up |
| `calendar_tool` | calendar, events, schedule | Calendar OAuth | STOP if not set up |
| `screenshot_tool` | screenshot, what's on screen, analyze screen | none | os_control |
| `app_launcher_tool` | open Spotify, launch VS Code, open [app] | none | terminal_tool |
| `session_tool` | rename session, list sessions, past session | none | none |
| `code_writer` | create a tool, write tool code, scaffold | none | none |
| `music_player` | play local music, local file, play [file path] | VLC installed | browser_control |
| `os_control` | click at, type on desktop, hotkey, drag | pyautogui installed | none |
| `terminal_tool` | PowerShell, cmd, run command, terminal | none | none |

## Output Rules

- `gmail_tool`: summarize sender, subject, date, action needed. No raw JSON.
- `calendar_tool`: summarize title, start, end, location. No raw JSON.
- `browser_control`: report URL, title, and what was extracted/done. No raw HTML.
- All tools: if `ok: false`, report the exact error. Do not paper over failures.

## Tool Abuse Prevention

- Do NOT use `code_writer` silently — only when user explicitly asks
- Do NOT use `session_tool` for personal memory — that's `memory_query`
- Do NOT use `music_player` for YouTube browser search — that's `browser_control`
- Do NOT use `terminal_tool` instead of `file_manager` for basic file ops
- Do NOT call `browser_control` if Nova Act API key is missing
