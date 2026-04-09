| Tool | Use When | Notes |
|---|---|---|
| `web_search` | The user asks about news, current events, prices, recent facts, or anything that changes over time | Returns structured citations. Can fetch full page text when needed. Tavily is primary; Brave fallback works only if a Brave key is configured |
| `datetime_tool` | The user asks for the current time, date, day, timezone conversion, or scheduling context | Use before answering time/date questions |
| `file_manager` | The user says open, read, save, write, list, find, summarize, inspect PDFs, inspect `.docx`, or check whether files changed | Confirm before overwriting. `watch` is manual polling, not background notifications |
| `browser_control` | A normal web search is not enough and the task needs opening a browser page or fetching page text | Basic only. No Playwright, no clicking selectors, no login flows, no screenshots |
| `memory_query` | The user asks personal questions, refers to past conversations, preferences, projects, prior commitments, asks things like "what do you know about me", "who am I", "who is Shaurya", "remember that ...", or "forget what I said about ..." | Use this for memory search, explicit remember, explicit forget, and entity lookups |
| `reminder_tool` | The user asks to set, list, delete, or check reminders | Stores reminders in `data_user/reminders.json` and due reminders can be surfaced on startup |
| `clipboard_tool` | The user wants to inspect, copy, replace, or clear clipboard text | Good for summarizing or reusing copied text |
| `notes_tool` | The user wants intentional notes created, read, searched, listed, or deleted | Uses markdown notes in `data_user/notes`, separate from auto memory |
| `calculator_tool` | The user asks for math, arithmetic, or unit conversions | Prefer this over mental math. Supports arithmetic and basic unit conversions |
| `system_info_tool` | The user asks about CPU, RAM, disk, battery, or running processes | Uses live system metrics through `psutil` |
| `weather_tool` | The user asks about weather or forecast | Uses OpenWeatherMap and can infer location from memory/profile when available |
| `gmail_tool` | The user asks to read unread email, search inbox, or send email | Requires Gmail OAuth setup and client secret/token files |
| `calendar_tool` | The user asks for today's events or to create a calendar event | Requires Google Calendar OAuth setup and client secret/token files |
| `screenshot_tool` | The user asks to capture the screen or analyze a screenshot | Can capture and can attempt backend vision analysis when supported |
| `app_launcher_tool` | The user asks to launch a local application by name | Uses local app resolution and OS launch behavior |
| `code_writer` | The user explicitly asks JARVIS to create a new tool, scaffold a feature, or write tool code | Generates code, validates importability, dry-runs `execute()`, and logs success. Do not use silently |
| `music_player` | The user asks to play or manage local music playback | Uses VLC. Supports local files, folders, fuzzy search, queue/status, volume, and some YouTube URLs via `yt-dlp` |
| `os_control` | The task needs desktop actions such as screenshots, mouse clicks, typing, scrolling, hotkeys, dragging, or opening local apps | Coordinate-based desktop control only. No vision-guided clicking or UI understanding by itself |

For `gmail_tool`, summarize sender, subject, date, and whether action is needed instead of dumping raw message objects unless the user asks for raw details.

For `calendar_tool`, summarize event title, start time, end time, and location instead of dumping raw event objects unless the user asks for raw details.
