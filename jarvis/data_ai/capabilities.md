## Can Do

- Search the web for current information and return structured citations when a search provider is configured
- Tell the current date and time
- Read, write, list, fuzzy-find, summarize, and manually watch files and folders
- Read PDF files and `.docx` documents through the file tool
- Set, list, delete, and surface due reminders from local reminder storage
- Read from and write to the system clipboard
- Create, read, search, list, and delete intentional markdown notes
- Evaluate math expressions and basic unit conversions safely
- Report live CPU, RAM, disk, battery, and running process information
- Get current weather and short forecast data when OpenWeatherMap is configured
- Read unread Gmail, search inbox, and send email after Gmail OAuth setup
- Read today's calendar events and create new events after Google Calendar OAuth setup
- Capture screenshots and attempt vision analysis when the configured backend/model supports image input
- Launch local apps by name or executable path
- Name sessions, list recent sessions, and search past named sessions
- Open browser pages and fetch page content in a basic way
- Remember past conversation chunks and search memory
- Create new tool modules with validation, dry-run checks, and self-update logging
- Control local VLC music playback for local files, folders, queue actions, volume, and some YouTube URLs
- Take desktop screenshots and perform direct desktop actions like clicking, typing, hotkeys, scrolling, dragging, and opening local apps
- Log conversations and memory extractions locally

## Cannot Do Yet

- Control Spotify directly
- Book cabs
- Send WhatsApp messages
- Perform advanced browser automation flows without a fuller Playwright integration
- Click website elements by selector or text through the current browser tool
- Reliably understand screen contents or click things by visual description alone unless the current backend/model actually supports the screenshot analysis request
- Guarantee that `music_player` will work if VLC desktop runtime is missing
- Push file change notifications automatically in the background; file watching is a manual polling action
- Guarantee live internet answers if required API keys or OAuth credentials are not configured
- Guarantee Gmail or Calendar access until the first OAuth authorization has been completed on this machine
