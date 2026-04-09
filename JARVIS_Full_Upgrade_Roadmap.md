# JARVIS — Full Upgrade Roadmap

---

## Phase 1 — Fix the Foundation (Do This First)

These aren't features, they're cracks in the current floor. Everything else builds on top.

**1.1 — Fix `_select_tool_definitions` empty list bug**
When no keywords match, agent passes zero tools to LLM. Change fallback from `return []` to `return all_tool_definitions` so the LLM always has tools available and decides itself.

**1.2 — Delete dead `simple_provider` / `complex_provider` properties in settings**
They're hardcoded strings that ignore your configurable order tuples. Dead code that will confuse you in 2 weeks.

**1.3 — Fix `mark_step_completed` in planner**
Pass step index instead of matching by tool name so multi-step plans with repeated tools don't desync.

**1.4 — Expand `_should_query_memory` keyword list**
Too many real memory queries get skipped. Either expand keywords aggressively or just always run the vector lookup — it's fast enough.

**1.5 — Suppress embedding load logs**
`BertModel LOAD REPORT` printing mid-conversation is jarring. Redirect stderr during that import so the CLI stays clean.

**1.6 — Upgrade `_task_kind` detection**
`"help me build a Python script"` doesn't trigger code model. Broaden the keyword list so Gemini 2.5 Pro is actually used for coding tasks.

---

## Phase 2 — Voice Layer (Makes it Feel Like JARVIS)

This is the biggest UX jump you can make. Text → Voice changes everything.

**2.1 — TTS integration (`services/tts.py`)**
Use `edge-tts` (free, Microsoft neural voices, no API key needed). Pick a deep confident voice like `en-US-GuyNeural` or `en-IN-PrabhatNeural`. Wire it so every JARVIS response gets spoken after printing. Add a `--voice` flag in `main.py` to toggle it.

**2.2 — STT integration (`services/stt.py`)**
Use OpenAI Whisper locally (the `tiny` or `base` model runs fine on 8GB RAM). Add a voice input mode — press Enter to start recording, press Enter again to stop, auto-transcribes and sends to agent. No wake word yet, just push-to-talk first.

**2.3 — Wake word detection**
After STT works, add Porcupine wake word (free tier, "Jarvis" is a built-in keyword). JARVIS runs in background, listening. You say "JARVIS" → it beeps → listens for your command → speaks response. This is the Iron Man moment.

**2.4 — Audio feedback sounds**
Add subtle activation sounds — a short beep when wake word detected, a different tone when JARVIS starts responding. Use `playsound` or `pygame`. Small detail, massive feel difference.

---

## Phase 3 — Tool Upgrades (Turn Placeholders into Real Power)

**3.1 — `browser_control.py` → Full Playwright**
Current version just opens URLs. Upgrade to full Playwright automation:
- Click elements by selector or text
- Fill forms
- Extract page content as clean text
- Take screenshots
- Handle login flows
- Scroll and wait for dynamic content

This unlocks booking sites, Google searches without API, form submissions, anything.

**3.2 — `music_player.py` → Real Playback**
Wire to VLC via `python-vlc`. Support:
- Play local files by name/folder
- Play YouTube URLs (via `yt-dlp` + VLC)
- Pause, stop, next, volume control
- Queue management

**3.3 — `code_writer.py` → Full Self-Coding Pipeline**
Right now it's a scaffold. Upgrade to:
- LLM generates full tool code using Gemini 2.5 Pro
- Auto-validates: does it have `TOOL_DEFINITION`? Does `execute()` exist? Does it import correctly?
- Runs a dry test call with dummy params
- If passes → saves to `/tools`, logs to `self_update_log.md`, hot-reloads registry
- If fails → sends error back to LLM for a fix attempt (up to 3 retries)

**3.4 — `os_control.py` → Real Computer Use**
Current version is 3 lines. Upgrade using `pyautogui` + `Pillow`:
- Take screenshots and send to vision model for context
- Click by coordinates or by describing what to click (vision-guided)
- Type text anywhere
- Open apps by name
- Scroll, drag, hotkeys

**3.5 — `file_manager.py` → Enhanced**
Add:
- Recursive search with fuzzy matching (find files even with partial names)
- Read PDFs (via `pypdf`)
- Read `.docx` files (via `python-docx`)
- Summarize long files before returning (avoid token overflow)
- File watching — notify JARVIS when a file changes

**3.6 — `web_search.py` → Citation Overhaul**
Current search returns results but citation structure is loose. Upgrade:
- Return structured JSON: `{title, url, snippet, published_date}`
- Add a secondary Brave Search API fallback (free tier) if Tavily quota hits
- Add a `fetch_full_page` option that uses `browser_control` to get full article text

---

## Phase 4 — New Tools to Build

**4.1 — `reminder_tool.py`**
Schedule future reminders. Store in a `data_user/reminders.json` file. On startup JARVIS checks for due reminders and speaks/prints them. Use APScheduler for in-process scheduling.

**4.2 — `clipboard_tool.py`**
Read and write system clipboard using `pyperclip`. Simple but insanely useful — "JARVIS, summarize what's in my clipboard", "JARVIS, copy this to clipboard."

**4.3 — `notes_tool.py`**
Create, read, search, and delete markdown notes in `data_user/notes/`. Different from memory — these are intentional user-created notes, not auto-extracted facts.

**4.4 — `calculator_tool.py`**
Evaluate math expressions safely using `ast.literal_eval` or `sympy`. Prevents LLM from doing mental math wrong. Also handles unit conversions.

**4.5 — `system_info_tool.py`**
CPU %, RAM usage, disk space, running processes, battery level via `psutil`. "JARVIS, how's my system doing?"

**4.6 — `weather_tool.py`**
OpenWeatherMap API (free tier is generous). Current weather + 5-day forecast for your location. Detects location from `data_user/profile.md` automatically.

**4.7 — `gmail_tool.py`**
Read unread emails, search inbox, send emails via Gmail API. Needs OAuth setup once, then works forever.

**4.8 — `calendar_tool.py`**
Google Calendar integration. Read today's events, create events, set reminders. Pairs with `reminder_tool`.

**4.9 — `screenshot_tool.py`**
Take screenshot, optionally send to Gemini Vision for analysis. "JARVIS, what's on my screen?" → Gemini describes it. Foundation for visual computer use.

**4.10 — `app_launcher_tool.py`**
Open apps by name on Windows/Linux. Map common names to executable paths. "JARVIS, open VS Code" → launches it.

---

## Phase 5 — Memory System Upgrades

**5.1 — Memory confidence scoring**
When extracting memories, have the LLM score each item 1-10 for confidence and importance. Only embed items above threshold 7. Prevents low-quality junk from polluting the vector DB.

**5.2 — Memory decay / cleanup**
Old chunks that haven't been retrieved in 30 days get demoted. Facts that contradict newer facts get overwritten not appended. Keeps memory fresh and accurate.

**5.3 — Explicit memory commands**
User can say "JARVIS, remember that I hate meetings on Mondays" → goes directly to `data_user/profile.md` and vector DB, bypassing the extraction pipeline. "JARVIS, forget what I said about X" → deletes matching vectors.

**5.4 — Memory summary on startup**
On first message of the day, JARVIS silently reads `profile.md` + `projects.md` and generates a 2-sentence internal context summary. Injected into every prompt that day. Like a human colleague catching up before a meeting.

**5.5 — Entity tracking**
Extract named entities (people, places, projects, tools) from conversations and maintain a separate `data_user/entities.json` with a knowledge graph. "Who is Shaurya?" → JARVIS can answer from stored context.

---

## Phase 6 — Brain / LLM Upgrades

**6.1 — Streaming responses**
Right now full response arrives at once. Add streaming so text prints word-by-word as it generates. For Gemini use `generateContent` with `stream: true`. For Groq it's SSE. Makes JARVIS feel alive during long responses.

**6.2 — Vision input**
Accept image input from user. "JARVIS, analyze this screenshot" → base64 encode image, send to `gemini-2.5-flash` with vision. Opens up: describe images, read text from photos, analyze diagrams.

**6.3 — Temperature routing**
- Tools/code/factual: `0.1`
- Planning: `0.3`
- Casual conversation: `0.7`
- Creative tasks: `0.9`

Currently hardcoded at `0.2` for everything.

**6.4 — Add OpenRouter as third provider**
OpenRouter gives you access to 100+ models (Claude, GPT, Mistral, etc.) through one API. Add it as Provider 3 in the fallback chain. When Groq fails and Gemini is rate-limited, OpenRouter saves you.

**6.5 — Context window management**
Currently history is just sliced to last 16 messages. Add smart truncation — summarize older turns into one compressed context block instead of hard cutting. Prevents losing important context from earlier in long sessions.

---

## Phase 7 — Interface Upgrades

**7.1 — Rich CLI with colors**
Use `rich` library. JARVIS responses in green, tool calls in yellow, errors in red, memory context in dim gray. Makes the terminal feel like a real interface.

**7.2 — Web UI (optional but cool)**
Simple FastAPI + HTML frontend. Text input box, response stream display, tool call visualization sidebar. You've already built FastAPI UIs before (WITNESS, JARVIS backend) so this is natural.

**7.3 — Multi-turn conversation modes**
- Normal chat mode (current)
- Focus mode — JARVIS only responds to task-relevant messages, filters small talk
- Briefing mode — at start of session, JARVIS summarizes pending tasks, reminders, and today's calendar
- Debug mode — prints plan, tool calls, provider used, latency for every turn

**7.4 — Session naming**
Each session gets a name (auto-generated or user-set). Sessions are stored as named JSONL files. You can later say "JARVIS, what did we discuss in the session about HireEnv?"

---

## Phase 8 — Self-Improvement Loop (Advanced)

**8.1 — Tool usage analytics**
Track which tools are called most, which fail most, average latency per tool. Store in `data_ai/tool_stats.json`. JARVIS can report: "web_search failed 3 times today."

**8.2 — Failure memory**
When a tool call fails, store what failed, why, and what the user was trying to do. Next time a similar request comes, JARVIS knows to try a different approach first.

**8.3 — Auto-prompt refinement**
Every week, JARVIS reviews the last 7 days of `behavior_rules.md` violations (cases where it broke a rule) and suggests edits to `system_prompt.md`. You approve or reject. Behavior improves over time without you manually editing files.

**8.4 — Capability gap detection**
When JARVIS fails a task because it has no tool for it, log the gap to `data_ai/capability_gaps.md`. Periodically review this file and use `code_writer` to fill the gaps.

---

## Priority Order Summary

| Phase | Focus | Impact | Effort |
|---|---|---|---|
| 1 | Bug fixes | 🔴 Critical | Low |
| 2 | Voice layer | 🟠 Massive UX | Medium |
| 3 | Tool upgrades | 🟠 High | Medium |
| 4 | New tools | 🟡 High | Low per tool |
| 5 | Memory upgrades | 🟡 Medium | Medium |
| 6 | Brain upgrades | 🟡 Medium | Medium |
| 7 | Interface | 🟢 Nice to have | Low-High |
| 8 | Self-improvement | 🟢 Long term | High |

---

**Next step**: Phase 1 bugs take 20 minutes total. Do those today, then start Phase 2.1 — `edge-tts` TTS wiring. That single change is the biggest feel upgrade in the least amount of time.
