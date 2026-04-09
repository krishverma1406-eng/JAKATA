# JARVIS — Personal AI Architecture

Big vision, solid foundation. Here's the full breakdown — no code, just the architecture, reasoning, and answers to all your questions.

---

## Core Philosophy

The whole system is built on **three principles**:
1. **Tools are plugins** — drop a `.py` file in `/tools`, the AI discovers and uses it automatically. No core changes ever.
2. **Agent loop is the brain** — Plan → Pick Tools → Execute → Observe → Respond. Everything flows through this.
3. **Memory is persistent** — every conversation, every learned fact, every result gets stored. JARVIS gets smarter over time.

---

## Folder Structure

```
jarvis/
│
├── main.py                    # Entry point — starts everything
│
├── core/
│   ├── brain.py               # LLM caller — all API calls live here
│   ├── agent.py               # The ReAct loop (Plan → Act → Observe → Respond)
│   ├── planner.py             # Breaks complex tasks into subtasks
│   ├── tool_registry.py       # Auto-scans /tools, builds tool list for LLM
│   └── memory.py              # Handles both vector memory and JSONL logs
│
├── tools/                     # ← DROP NEW TOOLS HERE, nothing else changes
│   ├── datetime_tool.py
│   ├── web_search.py
│   ├── music_player.py
│   ├── file_manager.py
│   ├── browser_control.py
│   ├── code_writer.py         # The self-coding tool
│   └── ... any new tool
│
├── memory/
│   ├── vector_db/             # ChromaDB persistent storage folder
│   └── logs/
│       ├── 2025-04-01.jsonl
│       ├── 2025-04-02.jsonl
│       └── ...                # One file per day, auto-created
│
├── services/
│   ├── tts.py                 # Text-to-speech (edge-tts, ElevenLabs, etc.)
│   ├── stt.py                 # Speech-to-text (Whisper)
│   └── os_control.py         # Keyboard, mouse, screen control (pyautogui)
│
├── config/
│   ├── settings.py            # Model name, API keys loader, feature flags
│   └── .env                   # API keys — never commit this
│
└── README.md
```

---

## How Each Part Works

### `tool_registry.py` — The Plugin Engine

This is the magic. On startup, it:
1. Scans every `.py` file in `/tools`
2. Each tool file must export two things: a `TOOL_DEFINITION` dict (name, description, parameters in JSON Schema) and an `execute(params)` function
3. It builds a master list of all tools and passes them to the LLM as function-calling definitions

**Result**: You drop `spotify_tool.py` in `/tools` tonight. Tomorrow morning JARVIS knows it exists and can use it. Zero changes elsewhere.

---

### `agent.py` — The ReAct Loop

This is the brain's execution engine. Every user message goes through this loop:

```
User says something
      ↓
Planner decides: simple task or multi-step?
      ↓
If multi-step → break into subtasks
      ↓
For each step:
   LLM picks a tool (or responds directly)
   Tool executes, returns result
   Result fed back to LLM as observation
   LLM decides: done? or need another tool?
      ↓
Final answer assembled and returned to user
      ↓
Entire exchange saved to JSONL + vector DB
```

This is basically how OpenClaw works — agents loop until the task is fully complete, not just one tool call per message.

---

### `memory.py` — Two Types of Memory

**Short-term (Context Window)**: Last N messages passed to LLM each time. Keeps conversation coherent.

**Long-term Vector Memory (ChromaDB)**:
- Every conversation chunk gets embedded and stored
- When you say "remember I have a meeting on Friday" → stored as vector
- Next time you ask "what do I have this week?" → semantic search finds it
- Also stores results from tool calls (web search results, file contents, etc.)
- ChromaDB is local, free, persistent — perfect for low budget

**Episodic Logs (JSONL)**:
- Every message pair (user + JARVIS) saved to `memory/logs/YYYY-MM-DD.jsonl`
- JSONL = one JSON object per line, append-only, dead simple
- Format per line: `{"timestamp": "...", "role": "user/assistant", "content": "...", "tools_used": [...]}`
- You can grep your entire life history later

---

### The Tool Standard

Every tool file follows the same contract:

```
TOOL_DEFINITION = {
  "name": "web_search",
  "description": "Search the internet for current information",
  "parameters": {
    "query": string,
    "max_results": integer (optional)
  }
}

def execute(params):
    # do the thing
    return result
```

**Minimal tools to start with:**
- `datetime_tool.py` — current date, time, day of week
- `web_search.py` — Tavily or SerpAPI (cheapest)
- `music_player.py` — VLC via python-vlc, plays local music or YouTube
- `file_manager.py` — read/write/list files
- `browser_control.py` — Playwright for full browser automation
- `memory_query.py` — tool to explicitly search past memories
- `code_writer.py` — writes a new `.py` file to `/tools` (self-coding, covered below)

---

## Answering Your Specific Questions

### ❓ Self-Coding — Can JARVIS write its own tools?

**Yes, and here's exactly how:**

You give JARVIS a tool called `code_writer.py`. When you say *"JARVIS, make a tool that sends WhatsApp messages"*, the agent:
1. Plans: "I need to write a new tool"
2. Calls `code_writer` with the tool description + your intent
3. `code_writer` calls the LLM (GPT-5/whatever you're using) with a prompt that includes the tool template standard
4. LLM generates the tool code
5. `code_writer` saves it to `/tools/whatsapp_tool.py`
6. `tool_registry.py` hot-reloads (or JARVIS restarts and picks it up next time)

**You don't need the "best model" for this.** GPT-4o writes clean Python tool files reliably. GPT-5 will be even better but it's not a requirement. What matters is the prompt template you give it (include the TOOL_DEFINITION schema and the execute() contract).

---

### ❓ GPT-5 — Is it good for this?

**Yes, GPT-5 is excellent here.** It has:
- Strong function/tool calling (critical for your agent loop)
- Good computer use benchmarks (OpenAI has been pushing this)
- Large context window (you can pass more memory + tools)
- Reasoning ability for complex multi-step planning

For **low budget** though:
- Use GPT-5 as the "planner brain" only (one call per user message to plan)
- Use a cheaper model (GPT-4o-mini or Groq Llama) for simple tool execution steps
- This is called **tiered routing** — big model for thinking, small model for doing
- Groq is free-tier friendly and insanely fast — use it for tool result summarization

---

### ❓ External Service Connections

Each external service gets a tool file. The `.env` holds the API keys. Examples:
- Google Calendar → `gcal_tool.py` (Google API)
- Spotify → `spotify_tool.py` (spotipy library)
- Gmail → `gmail_tool.py`
- Home automation → `smarthome_tool.py`
- GitHub → `github_tool.py`

You add the key to `.env`, drop the tool file, done.

---

### ❓ Multi-Tool Calls + Proper Planning

The planner uses a **Task Decomposition** approach:

When you say *"Book me a cab to Connaught Place at 6pm and remind me 30 minutes before"*:

```
Planner output:
  Step 1 → web_search("Ola/Uber API or open booking link")
  Step 2 → browser_control(open booking page, fill details)
  Step 3 → datetime_tool(calculate 5:30pm)
  Step 4 → reminder_tool(set alert at 5:30pm)
```

The agent executes these in order (or in parallel where possible), feeding each result into the next step. The LLM sees the full chain of observations before giving you the final answer.

---

## What to Build First (Priority Order)

| Phase | What to build | Why |
|---|---|---|
| **1** | `tool_registry.py` + `brain.py` + `datetime_tool.py` + `web_search.py` | Proves the plugin system works |
| **2** | `memory.py` (JSONL only first, add ChromaDB after) | Persistence is core |
| **3** | `agent.py` full ReAct loop | Multi-tool chaining |
| **4** | `planner.py` | Complex task decomposition |
| **5** | `browser_control.py` + `os_control.py` | Real computer use |
| **6** | `code_writer.py` | Self-evolving capability |
| **7** | Voice (STT + TTS) | JARVIS feel |

---

## Stack Summary (Low Budget Friendly)

| Component | Pick | Cost |
|---|---|---|
| LLM Brain | GPT-5 mini / Groq Llama | Low / Free tier |
| Vector DB | ChromaDB (local) | Free |
| Embeddings | `sentence-transformers` (local) or OpenAI small | Free / cheap |
| Web Search | Tavily API | Free tier generous |
| Browser | Playwright | Free |
| TTS | edge-tts | Free |
| STT | Whisper (local) | Free |
| Music | python-vlc | Free |
