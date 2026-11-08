"""Runs Claude Code headless and streams its replies back as spoken sentences."""
import json
import os
import queue
import subprocess
import sys
import threading

VOICE_PROMPT = """You are Winston, {user}'s voice remote for the Claude desktop app. {user} talks to you \
by voice; everything you write is read aloud.

YOUR MAIN JOB: pass what {user} says to the Claude session that is open in the Claude app, then tell \
{user} what Claude answered.
- By default, treat what {user} says as a prompt for the open Claude session: call send_to_claude_session \
with it. Fix obvious speech-recognition slips and drop filler ("can you", "for me"), but keep the meaning \
and wording; don't add your own ideas. "Write a prompt for X" or "ask Claude X" means send X.
- send_to_claude_session waits and returns Claude's reply. Keep that reply in mind, but don't describe it \
unless {user} asks you to explain (see Reporting back). Never read code or long text aloud.
- "Add X" means a new prompt: when {user} says "add" while working with Claude, call \
send_to_claude_session with X as the text and wait_for_reply=false, so X goes into the chat box as another \
prompt. Don't merge it into an earlier prompt, and don't send the word "add" itself. Outside Claude, "add" \
has its normal meaning.
- Only handle things yourself when {user} is clearly talking to you about the app: "switch to / open / \
go to <session>" (open_claude_session), "what did it say" (read_claude_session), "which sessions do I \
have" (list_claude_sessions), "send it to <session>" (send_to_claude_session with session=), "switch to \
<app>" (focus_window), or simple chat like greetings or the time.
- Never answer a prompt yourself instead of sending it, and never write files or build pages for it.
- If the Claude app isn't open, say so in one sentence.

Reporting back:
- When something {user} asked for is finished, say only "Done." or "Understood." Nothing else: don't say \
what you did or what Claude replied.
- Only when {user} asks you to explain ("explain this", "explain what you did", "what did you do") say what \
was done, in one to three short sentences. Use read_claude_session if you need Claude's reply again.
- If {user} asks a question or asks for information ("what did it say", "which sessions do I have"), answer \
it briefly; that isn't a task to report on.
- Never say "Done." for something that didn't happen. If it failed, say so in a few words. If Claude's reply \
is waiting on an answer or a decision from {user}, say only "Claude needs your answer."

Speaking:
- Short, plain spoken English. No markdown, lists, emoji or code.
- Never say "Winston" or "Jarvis": hearing the wake word is how {user} interrupts you.
- Your input comes from speech recognition, so expect misheard words. "problem" is often "prompt". Pick the closest real name rather than asking.
- Risky actions such as commands or deletions are confirmed with {user} by voice automatically; don't \
ask permission yourself.
- Any agents {user} has set up can be run with the Agent tool \
when {user} names one."""


class Claude:
    """One long-lived Claude Code process, fed requests as stream-json over stdin.

    Keeping it running saves the 1-3 s start-up (and MCP server launch) on every request.
    It is started ahead of time by warm_up(), and restarted with --resume after a cancel.
    """

    def __init__(self, cfg, mcp_config_path):
        self.cfg = cfg
        self.mcp_config = mcp_config_path
        self.session_id = None
        self.proc = None
        self.out = None
        self.lock = threading.Lock()

    def _start(self):
        cfg = self.cfg
        cmd = [cfg["claude_exe"], "-p",
               "--input-format", "stream-json",
               "--output-format", "stream-json", "--verbose",
               "--append-system-prompt", (VOICE_PROMPT.format(user=cfg["user_name"]) + "\n" + cfg.get("voice_notes", "")).strip(),
               "--mcp-config", self.mcp_config,
               "--permission-prompt-tool", "mcp__winston__approve",
               "--allowedTools", ",".join(cfg["allowed_tools"])]
        if cfg.get("model"):
            cmd += ["--model", cfg["model"]]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        # load Winston's tools up front instead of searching for them on every call (saves a round trip)
        # MCP_TOOL_TIMEOUT: send_to_claude_session waits for the other session's reply (up to 4 min)
        env = dict(os.environ, ENABLE_TOOL_SEARCH="false", MCP_TOOL_TIMEOUT="300000")
        self.proc = subprocess.Popen(cmd, cwd=cfg["workspace"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     encoding="utf-8", errors="replace", env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        self.out = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self.out), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

    @staticmethod
    def _read(proc, out):
        for line in proc.stdout:
            try:
                out.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        out.put(None)  # process ended

    @staticmethod
    def _drain_stderr(proc):
        for line in proc.stderr:
            if line.strip():
                print(f"[claude] {line.rstrip()[:300]}", file=sys.stderr)

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def warm_up(self):
        """Start the process now so the first request doesn't pay for it."""
        with self.lock:
            if not self._alive():
                self._start()

    def _kill(self):
        if self._alive():
            # kill the whole tree: Claude spawns tool processes and the MCP server
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        self.proc = None

    def reset(self):
        self._kill()
        self.session_id = None
        threading.Thread(target=self.warm_up, daemon=True).start()

    def cancel(self):
        self._kill()  # the next ask() restarts with --resume, keeping the conversation

    def ask(self, text, on_say):
        """Send one request. on_say(text) is called for each block of text Claude writes."""
        self.warm_up()
        proc, out = self.proc, self.out
        msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except OSError:  # died since warm-up: start again and retry once
            self._kill()
            self.warm_up()
            proc, out = self.proc, self.out
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        spoke = False
        while True:
            msg = out.get()
            if msg is None:  # process ended (cancelled or crashed) before finishing the turn
                return
            if msg.get("session_id"):
                self.session_id = msg["session_id"]
            kind = msg.get("type")
            if kind == "assistant":
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block["text"].strip():
                        on_say(block["text"])
                        spoke = True
                    elif block.get("type") == "tool_use":
                        print(f"  [tool] {block.get('name')}", file=sys.stderr)
            elif kind == "result":
                if msg.get("is_error"):
                    on_say(f"Sorry, Claude reported a problem: {msg.get('result') or msg.get('subtype')}")
                elif not spoke and msg.get("result"):
                    on_say(msg["result"])
                return


def write_mcp_config(base_dir, port, token):
    """Point Claude at our approval server, telling it how to reach this Winston process."""
    run_dir = os.path.join(base_dir, ".runtime")
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "mcp.json")
    config = {"mcpServers": {"winston": {
        "command": sys.executable,
        "args": [os.path.join(base_dir, "winston", "approval_server.py")],
        "env": {"WINSTON_PORT": str(port), "WINSTON_TOKEN": token},
    }}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path
