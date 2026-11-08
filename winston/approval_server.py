"""MCP server Claude Code calls whenever a tool needs permission.

Safe edits inside the workspace are allowed here; everything else is forwarded to the
running Winston process, which asks the user out loud and waits for a spoken yes or no.
"""
import json
import os
import sys
import urllib.request

from mcp.server.mcpserver import MCPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from winston.settings import load  # noqa: E402

CONFIG = load(sys.path[0])
WORKSPACE = os.path.normcase(os.path.abspath(CONFIG["workspace"]))
PORT = os.environ.get("WINSTON_PORT")
TOKEN = os.environ.get("WINSTON_TOKEN", "")

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
PROTECTED = (".git", ".env", ".claude")

mcp = MCPServer("winston")


def in_workspace(path):
    if not path:
        return False
    full = os.path.normcase(os.path.abspath(path))
    if not full.startswith(WORKSPACE + os.sep):
        return False
    parts = full[len(WORKSPACE) + 1:].split(os.sep)
    return not any(p.startswith(PROTECTED) for p in parts)


def describe(tool, args):
    if tool == "mcp__winston__send_to_claude_session":
        where = f" in the {args['session']} session" if args.get("session") else ""
        return f"type this into Claude{where}: {args.get('text', '')[:200]}"
    if tool in ("Bash", "PowerShell"):
        text = args.get("description") or " ".join(str(args.get("command", "")).split()[:12])
        return text[:1].lower() + text[1:]  # it follows "May I ..."
    if tool in EDIT_TOOLS:
        return f"change the file {os.path.basename(str(args.get('file_path', '')))} outside the workspace"
    return f"use the {tool.replace('mcp__', '').replace('__', ' ')} tool"


def ask_by_voice(question):
    if not PORT:
        return False
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/approve",
                                 data=json.dumps({"question": question}).encode(),
                                 headers={"Content-Type": "application/json", "X-Winston-Token": TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r).get("allow") is True
    except Exception:
        return False


@mcp.tool()
def approve(tool_name: str, input: dict, tool_use_id: str = "") -> dict:
    """Decide whether Claude may run a tool call."""
    # Returned as a dict so the structured result and the text result carry the same JSON;
    # a str return would be wrapped as {"result": "..."}, which Claude Code rejects.
    if CONFIG.get("auto_allow_edits_in_workspace") and tool_name in EDIT_TOOLS \
            and in_workspace(input.get("file_path") or input.get("notebook_path")):
        allowed = True
    else:
        allowed = ask_by_voice(describe(tool_name, input))
    if allowed:
        return {"behavior": "allow", "updatedInput": input}
    return {"behavior": "deny",
            "message": "The user declined this by voice (or did not answer). Do not retry it; "
                       "tell the user briefly what you could not do."}


# --- Claude desktop app and window control -----------------------------------
# Listing, switching and reading are allowed without asking (config allowed_tools);
# sending a message goes through approve() above, so the user confirms it by voice.

@mcp.tool()
def list_claude_sessions() -> str:
    """List the sessions (chats/windows) in the user's Claude desktop app and say which one is open."""
    from winston import claude_app
    names, current = claude_app.sessions()
    return f"Open: {current}. All sessions: " + "; ".join(names)


@mcp.tool()
def open_claude_session(name: str) -> str:
    """Bring the Claude desktop app to the front and switch to a session. name can be approximate
    (speech recognition mangles names); the closest session title is used."""
    from winston import claude_app
    return claude_app.open_session(name)


@mcp.tool()
def read_claude_session() -> str:
    """Read Claude's latest reply in the session currently open in the Claude desktop app."""
    from winston import claude_app
    return claude_app.read_session()


@mcp.tool()
def send_to_claude_session(text: str, session: str = "", wait_for_reply: bool = True) -> str:
    """Type a prompt into a Claude desktop app session and press Enter. If session is given,
    switch to it first (approximate name is fine). By default waits until Claude has finished
    and returns its reply, so you can tell the user what it said."""
    from winston import claude_app
    return claude_app.send_to_session(text, session or None, wait=wait_for_reply)


@mcp.tool()
def wait_for_claude_reply(timeout_seconds: int = 240) -> str:
    """Wait for the open Claude session to finish its current reply, then return it."""
    from winston import claude_app
    return claude_app.wait_for_reply(timeout_seconds)


@mcp.tool()
def list_windows() -> str:
    """List the titles of the open windows on this PC."""
    from winston import claude_app
    return "; ".join(claude_app.windows())


@mcp.tool()
def focus_window(title: str) -> str:
    """Bring a window to the front by (approximate) title, e.g. 'Chrome', 'File Explorer'."""
    from winston import claude_app
    return claude_app.focus_window(title)


if __name__ == "__main__":
    mcp.run()
