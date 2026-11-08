"""Drive the Claude desktop app through Windows UI Automation: list, switch, read and type into sessions.

The app is Electron, so its sidebar, message list and message box are exposed as accessible
elements. Chromium only builds that tree once an automation client asks, so the first call
may need a short retry.
"""
import difflib
import re
import time

from pywinauto import Desktop
from pywinauto.keyboard import send_keys


def _claude_window():
    for w in Desktop(backend="uia").windows():
        if w.window_text() == "Claude" and w.element_info.class_name == "Chrome_WidgetWin_1":
            return w
    raise RuntimeError("The Claude app isn't open.")


def _buttons(win):
    for attempt in range(3):
        buttons = win.descendants(control_type="Button")
        if len(buttons) > 10:  # accessibility tree is populated
            return buttons
        time.sleep(0.7)
    return buttons


def sessions():
    """Names of the sessions in the sidebar, and which one is open."""
    win = _claude_window()
    names, current = [], None
    for b in _buttons(win):
        text = b.window_text()
        if text.startswith("More options for "):
            name = text[len("More options for "):]
            if name not in names:
                names.append(name)
        elif text.endswith(", rename session"):
            current = text[: -len(", rename session")]
    if current and current not in names:
        names.insert(0, current)
    return names, current


def _best_match(wanted, names):
    """Speech recognition mangles names ("fleet OS H changes"), so match loosely."""
    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).replace("  ", " ").strip()
    w = norm(wanted)
    scored = []
    for n in names:
        nn = norm(n)
        score = difflib.SequenceMatcher(None, w, nn).ratio()
        if nn and (nn in w or w in nn):
            score += 0.5
        score += 0.1 * len(set(w.split()) & set(nn.split()))
        scored.append((score, n))
    scored.sort(reverse=True)
    return scored[0] if scored else (0, None)


def open_session(name):
    """Bring the Claude app forward and switch to the session that best matches name."""
    win = _claude_window()
    names, current = sessions()
    score, best = _best_match(name, names)
    if not best or score < 0.45:
        return f"No session matches '{name}'. Sessions: " + "; ".join(names)
    if win.is_minimized():
        win.restore()
    win.set_focus()
    if best != current:
        for b in _buttons(win):
            if b.window_text() == best:
                b.invoke()
                break
        time.sleep(0.8)
    return f"Opened the session '{best}'."


def read_session(max_chars=1500):
    """The latest message in the open session, as plain text."""
    win = _claude_window()
    docs = win.descendants(control_type="Document")
    title = docs[0].window_text().removesuffix(" - Claude Code") if docs else "Claude"
    messages = []
    for g in win.descendants(control_type="Group"):
        m = re.fullmatch(r"Message (\d+)", g.window_text())
        if m:
            messages.append((int(m.group(1)), g))
    if not messages:
        return f"{title}: no messages visible."
    # newest first; prefer Claude's reply over the user's own message
    for _, group in sorted(messages, key=lambda x: -x[0]):
        parts = [t.window_text() for t in group.descendants(control_type="Text") if t.window_text().strip()]
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if text and not text.startswith("You said"):
            return f"{title}, Claude's latest reply: {text[-max_chars:]}"
    return f"{title}, latest message: {text[-max_chars:]}"


def _escape_keys(text):
    # pywinauto treats these as key codes; braces make them literal
    return re.sub(r"([{}+^%~()\[\]])", r"{\1}", text).replace("\n", "+{ENTER}")


def _streaming(win):
    return any(g.window_text() == "Currently streaming message"
               for g in win.descendants(control_type="Group"))


def wait_for_reply(timeout=240):
    """Wait until Claude stops writing in the open session, then return its reply."""
    win = _claude_window()
    time.sleep(2)  # give the app a moment to start streaming
    end, quiet = time.time() + timeout, 0
    while time.time() < end:
        quiet = 0 if _streaming(win) else quiet + 1
        if quiet >= 2:  # not streaming on two checks in a row: the turn is over
            return read_session()
        time.sleep(1.5)
    return "Claude is still working after " + str(timeout) + " seconds. " + read_session()


def send_to_session(text, session=None, wait=True):
    """Type a message into a session's message box, send it, and (by default) wait for the reply."""
    note = open_session(session) + " " if session else ""
    if note.startswith("No session"):
        return note
    win = _claude_window()
    win.set_focus()
    boxes = [e for e in win.descendants(control_type="Edit") if "ProseMirror" in e.element_info.class_name]
    if not boxes:
        return note + "I couldn't find the message box."
    box = boxes[-1]
    box.set_focus()
    box.click_input()
    send_keys(_escape_keys(text), with_spaces=True, with_newlines=False, pause=0)
    time.sleep(0.2)
    send_keys("{ENTER}")
    _, current = sessions()
    note += f"Sent to '{current}'."
    return note + " " + wait_for_reply() if wait else note


def windows():
    """Titles of the open windows."""
    return [w.window_text() for w in Desktop(backend="uia").windows()
            if w.window_text() and w.window_text() not in ("Taskbar", "Program Manager", "pythonw")]


def focus_window(title):
    """Bring the window whose title best matches to the front."""
    wins = [w for w in Desktop(backend="uia").windows() if w.window_text()]
    score, best = _best_match(title, [w.window_text() for w in wins])
    if not best or score < 0.45:
        return f"No window matches '{title}'. Open windows: " + "; ".join(windows())
    w = next(w for w in wins if w.window_text() == best)
    if w.is_minimized():
        w.restore()
    w.set_focus()
    return f"Switched to '{best}'."
