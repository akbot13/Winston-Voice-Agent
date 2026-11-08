# Winston Voice Agent

A voice agent. Say the wake word, speak, and Claude Code does the work and answers out loud.

```
mic → wake word → speech to text → Claude Code (in your workspace) → voice → speaker
     openWakeWord   faster-whisper    your Claude plan, your agents      edge-tts
```

## Install from a fresh copy

Voice samples, the trained wake model and other models are not in this repository, so a fresh copy needs these steps (Windows):

1. `py -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt`
2. `cd tools`, then `npm install`, then log in to Claude Code once
3. Create `config.local.json` with your own settings, such as `user_name`, `workspace`, `claude_exe`, `vocabulary` and `voice_notes`. It overrides `config.json` and is never uploaded.
4. Optional: build your own "Hey Winston" model: run `tools\record_test.py`, `tools\record_wake.py`, `tools\record_bare.py`, then `tools\train_wake.py`, and set `"wake_mode": "personal"` in `config.local.json`.
   Until then the agent uses the stock "Hey Jarvis" wake word.

## Start it

Double-click **Winston.bat** (or the "Winston" shortcut on the desktop). The first start loads the speech
models, which takes up to a minute. When you hear "Winston is ready", say the wake word.

The wake word is **"Hey Winston"**: a small personal model trained on your own voice. It works even while Winston is talking (see the wake-word section below).

## What to say

| Say | What happens |
|---|---|
| *Hey Winston* … *what's the status of my website?* | Claude answers in a sentence or two |
| *Hey Winston* … *ask the research agent for three new ideas* | runs that agent and sums up its answer |
| *Hey Winston* … *open Chrome to my website* | asks "May I …? Yes or no." first, then does it |
| *Hey Winston* while Winston is talking | stops mid-sentence (and stops the task) and listens |
| *Hey Winston, …* in one breath | beeps and keeps listening, so the whole request gets through |
| anything without the word *Winston* | ignored, unless Winston has asked you yes or no |
| wake word while Claude is working, then *stop* | cancels the task |
| *switch to the billing session* | brings the Claude app forward and opens that session |
| *what did it say?* | reads you Claude's latest reply in the open session |
| *tell the website session to run the tests* | types it into that session, presses Enter and sums up Claude's reply |
| *which Claude windows do I have?* | lists your Claude app sessions |
| *switch to Chrome* | brings any window to the front |
| **Ctrl+Alt+W** or click the orb | cuts Winston off (and the task) and listens |
| *new conversation* / *start over* | forgets the conversation so far |
| *goodbye Winston* / *shut down* | quits |

Winston remembers the conversation until you start a new one.

## Safety

- Reading, searching, web lookups and the department agents run without asking.
- Edits to files inside your workspace are allowed, except in `.git`, `.env` and `.claude`.
- **Everything else asks you out loud first**: commands, deletions, files outside the workspace, and other tools.
  Say *yes* or *no*. No answer, or an unclear one asked twice, counts as **no**.
- Nothing listens on the network. The approval channel is `127.0.0.1` only and needs a per-run secret token.

Change the rules in `config.json` (`allowed_tools`, `auto_allow_edits_in_workspace`).

## ElevenLabs voice (optional)

1. Copy `.env.example` to `.env` and paste your key after `ELEVENLABS_API_KEY=`.
2. Restart Winston.

The voice is set in `config.json` → `elevenlabs.voice_id` (default: Daniel, British). If ElevenLabs fails or
runs out of credits, Winston switches to the free voice for that sentence. Set `elevenlabs.enabled` to
`false` to turn it off.

## Settings (`config.json`)

- `voice`: any edge-tts voice, e.g. `en-GB-RyanNeural` (default) or `en-GB-ThomasNeural`. The Windows voice is used automatically when offline.
- `whisper_model`: `small.en` (default). Use `base.en` for faster, less accurate recognition, or `medium.en` for slower, more accurate.
- `personal_wake_threshold` (0.9) and `personal_wake_frames` (3): lower them if Winston misses you; raise them if it wakes by mistake.
- `wake_threshold`: raise it (e.g. `0.6`) if Winston wakes by mistake; lower it (`0.4`) if it misses you.
- `silence_seconds`: how long a pause ends your sentence.

## Custom "Hey Winston" wake word

The model lives in `models\hey_winston.pkl`. To retrain it, for example after a new mic or if it often misses you:

1. `.venv\Scripts\python.exe tools\record_wake.py` (about 3 minutes of saying "Hey Winston" plus some normal speech), then `tools\record_bare.py` (about 2 minutes of saying just "Winston")
2. `.venv\Scripts\python.exe tools\train_wake.py` (prints held-out catch rate and false wakes, then saves the model)

Restart Winston to load it. Set `"wake_mode": "openwakeword"` to fall back to the stock "Hey Jarvis" model.

## Files

- `winston/main.py`: main loop (wake word, listening, approvals)
- `winston/audio.py`: mic, wake word, speech to text, voice
- `winston/brain.py`: runs Claude Code headless and streams replies
- `winston/approval_server.py`: MCP permission tool Claude calls before risky actions
- `tools/`: the Claude Code CLI Winston uses (log in once with `tools\node_modules\@anthropic-ai\claude-code\bin\claude.exe auth login`)
