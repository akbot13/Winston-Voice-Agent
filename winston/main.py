"""Winston: say "Hey Winston" (or click the orb), speak, and Claude Code answers out loud."""
import json
import os
import queue
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from winston.audio import (EchoCanceller, Mic, NameWake, PersonalWake, Speaker,  # noqa: E402
                           Transcriber, WakeWord, record_utterance, speakable)
from winston.brain import Claude, write_mcp_config  # noqa: E402
from winston.settings import load as load_settings  # noqa: E402

YES = re.compile(r"\b(yes|yeah|yep|yup|sure|okay|ok|go ahead|do it|approve|approved|confirm|fine|please do)\b")
NO = re.compile(r"\b(no|nope|don't|do not|stop|cancel|deny|negative|wait)\b")
STOP = re.compile(r"^\s*(stop|cancel|never ?mind|abort|quiet|shut up)\W*$")
RESET = re.compile(r"\b(new conversation|start over|fresh start|forget (that|this|everything))\b")
SHUTDOWN = re.compile(r"\b(shut down|go to sleep|goodbye|good night)( winston)?\W*$|^\s*exit\W*$")
STOCK_PHRASES = ["Yes?", "Okay.", "Cancelled.", "Still working.", "Say stop to cancel.",
                 "I didn't catch that.", "Alright, I won't.", "Sorry, was that a yes or a no?",
                 "I'll take that as a no.", "Fresh start.", "What can I do?", "Yes or no.",
                 "Done.", "Understood.", "Claude needs your answer."]
THANKS = re.compile(r"^\s*(thanks|thank you|that's all|that is all|nothing|no thanks)\b")


class NullUI:
    """Used with --no-orb: the console is the only display."""
    def state(self, s): pass
    def caption(self, who, text): pass
    def level(self, x): pass


class ApprovalHandler(BaseHTTPRequestHandler):
    """Approval requests from the MCP server; answered by the main thread over the mic."""
    requests_q = None
    token = None

    def do_POST(self):
        if self.path != "/approve" or self.headers.get("X-Winston-Token") != self.token:
            self.send_response(403)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        answer = queue.Queue()
        self.requests_q.put((body.get("question", "do something"), answer))
        allow = answer.get()
        out = json.dumps({"allow": allow}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class Winston:
    def __init__(self, ui=None):
        self.ui = ui or NullUI()
        self.ui.state("loading")
        self.cfg = load_settings(BASE)
        print("Loading speech models (first start can take a minute)...")
        try:
            echo = EchoCanceller() if self.cfg.get("echo_cancellation", True) else None
        except Exception as e:
            print(f"[echo] not available ({e}); interrupt with the orb or Ctrl+Alt+W")
            echo = None
        self.speaker = Speaker(self.cfg, BASE, echo)
        self.stt = Transcriber(self.cfg)
        if self.cfg.get("wake_mode") == "name":
            self.wake = NameWake(self.cfg, refine=self.stt.transcribe)
        elif self.cfg.get("wake_mode") == "personal":
            self.wake = PersonalWake(self.cfg, BASE)
        else:
            self.wake = WakeWord(self.cfg, BASE)
        self.mic = Mic(echo)

        self.approvals = queue.Queue()
        ApprovalHandler.requests_q = self.approvals
        ApprovalHandler.token = secrets.token_hex(16)
        # 127.0.0.1 only: nothing on the network can reach this
        server = ThreadingHTTPServer(("127.0.0.1", 0), ApprovalHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        mcp_path = write_mcp_config(BASE, server.server_address[1], ApprovalHandler.token)
        self.claude = Claude(self.cfg, mcp_path)
        threading.Thread(target=self.claude.warm_up, daemon=True).start()
        self.speaker.prepare(STOCK_PHRASES)
        self.worker = None
        self.turn = 0  # bumped per request and on interrupt, so a cancelled turn doesn't trigger follow-up
        self.wake_request = threading.Event()  # set by clicking the orb
        self.follow_up = False                 # listen for a reply once Winston finishes
        self.shown = None

    # --- helpers -------------------------------------------------------------
    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def show(self, state):
        if state != self.shown:
            self.shown = state
            self.ui.state(state)

    def say(self, text):
        self.ui.caption(self.cfg["name"], speakable(text))
        self.speaker.say(text)

    def chime(self, text):
        self.say(text)
        self.speaker.wait()

    def listen(self, max_wait=None, flush=True):
        """Record one utterance and return its text ('' if nothing usable was said).
        flush=False keeps audio already captured, e.g. words said right after the wake word."""
        self.speaker.wait()
        if flush:
            time.sleep(0.15)   # let the speaker's tail die away before we listen
            self.mic.flush()
        self.show("listening")
        audio = record_utterance(self.mic, self.cfg, max_wait=max_wait, on_level=self.ui.level)
        self.show("thinking")
        text = self.stt.transcribe(audio) if audio is not None else ""
        if text:
            print(f"\n  {self.cfg['user_name']}: {text}")
            self.ui.caption(self.cfg["user_name"], text)
        return text

    def handle_approval(self, question, answer):
        self.speaker.wait()
        for attempt in range(2):
            self.show("asking")
            prompt = f"May I {question}? Yes or no." if attempt == 0 else "Sorry, was that a yes or a no?"
            self.chime(prompt)
            reply = self.listen().lower()
            if YES.search(reply) and not NO.search(reply):
                self.say("Okay.")
                answer.put(True)
                return
            if NO.search(reply):
                self.say("Alright, I won't.")
                answer.put(False)
                return
        self.say("I'll take that as a no.")
        answer.put(False)

    # Whenever the user says "Winston", stop whatever you are saying or doing and start listening.
    # This applies whether the user says "Winston" alone or with other words after it.
    def run_request(self, text):
        self.turn += 1
        my_turn = self.turn

        def work():
            try:
                self.claude.ask(text, lambda t: self.say(t) if my_turn == self.turn else None)
            except Exception as e:
                self.say(f"Something went wrong talking to Claude: {e}")
            if my_turn == self.turn:
                self.follow_up = self.cfg["follow_up_seconds"] > 0
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def interrupt(self):
        """The user cut in: stop talking, and stop the task if Claude is still on it."""
        self.speaker.stop()
        self.follow_up = False
        if self.busy():
            self.turn += 1
            self.claude.cancel()

    # --- commands --------------------------------------------------------------
    def new_conversation(self):
        self.claude.reset()
        self.say("Fresh start. What can I do?")

    def quit(self):
        self.claude.cancel()
        self.speaker.stop()
        self.chime("Goodbye.")
        os._exit(0)

    def handle(self, text):
        """Act on something the user said (after the wake word, or as a follow-up)."""
        low = text.lower()
        if SHUTDOWN.search(low):
            self.quit()
        elif RESET.search(low):
            self.new_conversation()
        elif STOP.search(low) or THANKS.search(low):
            self.say("Okay.")
        else:
            self.run_request(text)

    def on_wake(self, said_after=""):
        self.speaker.stop()
        if self.busy():
            if said_after:
                if STOP.search(said_after.lower()):
                    self.interrupt()
                    self.say("Cancelled.")
                return
            self.speaker.beep()
            text = self.listen(flush=False)
            if not text:
                return
            self.interrupt()  # a new request replaces the one still running
            if STOP.search(text.lower()):
                self.say("Cancelled.")
            else:
                self.handle(text)
            return
        if said_after and len(said_after.split()) >= 2:
            print(f"\n  {self.cfg['user_name']}: {said_after}")
            self.ui.caption(self.cfg["user_name"], said_after)
            self.handle(said_after)
            return
        if isinstance(self.wake, NameWake):
            self.chime("Yes?")
            text = self.listen()
        else:
            # beep instead of "Yes?" and keep recording, so "Hey Winston, open the website
            # session" said in one breath isn't cut off
            self.speaker.beep()
            text = self.listen(flush=False)
        if text:  # silence after the beep is most likely a false trigger: say nothing
            self.handle(text)

    # --- main loop -------------------------------------------------------------
    def heard_wake(self, chunk):
        """None if not woken; otherwise the words said after the name ('' if none)."""
        if isinstance(self.wake, NameWake):
            return self.wake.feed(chunk, self.mic.noise)
        return "" if self.wake.heard(chunk) else None

    def loop(self):
        # no name in the greeting: hearing the name while speaking is the interrupt signal
        self.say("Ready when you are.")
        print(f"\nListening for '{self.wake.label}'. Ctrl+C to quit.\n")
        while True:
            try:
                question, answer = self.approvals.get_nowait()
                self.handle_approval(question, answer)
                continue
            except queue.Empty:
                pass

            speaking = self.speaker.speaking()
            self.show("speaking" if speaking else "thinking" if self.busy() else "idle")

            # a reply has finished: give the user a few seconds to answer without the wake word
            if self.follow_up and not speaking and not self.busy():
                self.follow_up = False
                text = self.listen(max_wait=self.cfg["follow_up_seconds"])
                if isinstance(self.wake, PersonalWake):
                    self.wake.reset()
                if text:
                    self.handle(text)
                continue

            chunk = self.mic.read()
            if self.wake_request.is_set():  # orb click or Ctrl+Alt+W: cut in and listen
                self.wake_request.clear()
                if speaking:
                    self.interrupt()
                self.on_wake()
                self.mic.flush()
                if isinstance(self.wake, PersonalWake):
                    self.wake.reset()
                continue
            if speaking:
                # "Winston" over Winston's own voice cuts it off.
                if isinstance(self.wake, PersonalWake):
                    # no said_recently() check here: a reply that merely mentioned "Winston"
                    # used to block the interrupt for the next 20 seconds
                    if self.wake.heard(chunk, speaking=True):
                        self.interrupt()
                        self.on_wake()
                        self.mic.flush()
                        self.wake.reset()
                    continue
                # "Stop" etc. If Winston itself just said that word, what we hear is probably
                # its own voice, so ignore it.
                if isinstance(self.wake, NameWake):
                    word = self.wake.heard_over_speech(chunk)
                else:
                    word = self.wake.label.split()[-1].lower() if self.wake.heard(chunk) else None
                if word and not self.speaker.said_recently(word):
                    self.interrupt()
                    if word in self.cfg["wake_words"] or not isinstance(self.wake, NameWake):
                        self.on_wake()
                    else:
                        self.say("Okay.")
                    self.mic.flush()
                continue
            self.mic.track_noise(chunk)
            said_after = self.heard_wake(chunk)
            if said_after is not None:
                self.on_wake(said_after)
                self.mic.flush()
                if isinstance(self.wake, PersonalWake):
                    self.wake.reset()


def start_hotkey(event):
    """Ctrl+Alt+W anywhere in Windows = click the orb (interrupt and listen)."""
    import ctypes
    from ctypes import wintypes

    def watch():
        user32 = ctypes.windll.user32
        MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, WM_HOTKEY = 0x1, 0x2, 0x4000, 0x0312
        if not user32.RegisterHotKey(None, 1, MOD_ALT | MOD_CONTROL | MOD_NOREPEAT, ord("W")):
            print("[hotkey] Ctrl+Alt+W is taken by another app; use the orb instead")
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                event.set()
    threading.Thread(target=watch, daemon=True).start()


def run_console():
    try:
        w = Winston()
        start_hotkey(w.wake_request)
        w.loop()
    except KeyboardInterrupt:
        print("\nBye.")


def run_with_orb():
    from PySide6.QtWidgets import QApplication
    from winston.orb import Bridge, Orb

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    bridge = Bridge()
    holder = {}

    class UI:
        def state(self, s): bridge.state.emit(s)
        def caption(self, who, text): bridge.caption.emit(who, text)
        def level(self, x): bridge.level.emit(x)

    def click():
        if holder.get("w"):
            holder["w"].wake_request.set()

    def new_chat():
        if holder.get("w"):
            holder["w"].new_conversation()

    def quit_all():
        if holder.get("w"):
            threading.Thread(target=holder["w"].quit, daemon=True).start()
        else:
            os._exit(0)

    orb = Orb(bridge, os.path.join(BASE, ".runtime", "orb.json"), click, new_chat, quit_all)
    orb.show()

    def start():
        try:
            holder["w"] = Winston(UI())
            start_hotkey(holder["w"].wake_request)
            holder["w"].loop()
        except Exception as e:
            import traceback
            traceback.print_exc()
            bridge.state.emit("error")
            bridge.caption.emit("Winston", f"I couldn't start: {e}")

    threading.Thread(target=start, daemon=True).start()
    sys.exit(app.exec())


def main():
    if sys.stdout is None:  # started with pythonw (no console): log to a file instead
        os.makedirs(os.path.join(BASE, ".runtime"), exist_ok=True)
        log = open(os.path.join(BASE, ".runtime", "winston.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
        print(f"\n===== Winston started {time.ctime()} =====")
    if "--no-orb" in sys.argv:
        run_console()
    else:
        run_with_orb()


if __name__ == "__main__":
    main()
