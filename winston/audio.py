"""Microphone, wake word, speech-to-text and text-to-speech for Winston."""
import collections
import wave
import asyncio
import json
import os
import queue
import re
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd

RATE = 16000
CHUNK = 1280  # 80 ms, the frame size openWakeWord expects
# A laptop mic can be quiet: normal speech measures ~90-600 RMS over a ~15 noise floor, so a
# fixed 350 bar cut off soft first words and ended sentences early. Measured on his recordings.
SPEECH_FLOOR = 60.0
LEAD_IN = 12  # chunks (~1 s) kept from before speech is detected


def _add_cuda_dlls():
    """faster-whisper needs cuBLAS/cuDNN; they ship as pip wheels under site-packages/nvidia."""
    import site
    for sp in site.getsitepackages():
        root = os.path.join(sp, "nvidia")
        if not os.path.isdir(root):
            continue
        for pkg in os.listdir(root):
            bin_dir = os.path.join(root, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


class EchoCanceller:
    """WebRTC echo cancellation (via LiveKit): removes Winston's own voice from the mic.

    The speaker feeds everything it plays into render(); the mic passes every chunk through
    capture(). Without this, Whisper hears Winston's voice over the user's and misses "Hey Winston".
    """
    FRAME = RATE // 100  # WebRTC works in 10 ms frames

    def __init__(self, delay_ms=60):
        from livekit import rtc
        self.rtc = rtc
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
        self.apm.set_stream_delay_ms(delay_ms)
        self.pending = np.zeros(0, dtype=np.int16)
        self.lock = threading.Lock()

    def render(self, block, rate):
        """Audio that is being played right now, at its own sample rate."""
        if rate != RATE:
            n = int(len(block) * RATE / rate)
            block = np.interp(np.linspace(0, len(block) - 1, n), np.arange(len(block)), block)
        with self.lock:
            self.pending = np.concatenate([self.pending, block.astype(np.int16)])
            while len(self.pending) >= self.FRAME:
                frame, self.pending = self.pending[:self.FRAME], self.pending[self.FRAME:]
                self.apm.process_reverse_stream(self.rtc.AudioFrame(frame.tobytes(), RATE, 1, self.FRAME))

    def capture(self, chunk):
        out = []
        for i in range(0, len(chunk) - self.FRAME + 1, self.FRAME):
            frame = self.rtc.AudioFrame(chunk[i:i + self.FRAME].tobytes(), RATE, 1, self.FRAME)
            self.apm.process_stream(frame)  # cleaned in place
            out.append(np.frombuffer(frame.data, dtype=np.int16).copy())
        return np.concatenate(out) if out else chunk


class Mic:
    """Always-open input stream; chunks of int16 audio land in a queue."""

    def __init__(self, echo=None):
        self.q = queue.Queue()
        self.echo = echo
        self.noise = 200.0  # running estimate of the room's background level (RMS)
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                     blocksize=CHUNK, callback=self._cb)
        self.stream.start()

    def _cb(self, data, frames, t, status):
        chunk = data[:, 0].copy()
        if self.echo:
            try:
                chunk = self.echo.capture(chunk)
            except Exception as e:
                print(f"[echo] disabled: {e}")
                self.echo = None
        self.q.put(chunk)

    def read(self, timeout=None):
        return self.q.get(timeout=timeout)

    def flush(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def track_noise(self, chunk):
        rms = _rms(chunk)
        # follow the floor down fast, up slowly, so speech doesn't inflate it
        self.noise = rms if rms < self.noise else self.noise * 0.995 + rms * 0.005


def _rms(chunk):
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) + 1e-6)


class WakeWord:
    def __init__(self, cfg, base_dir):
        from openwakeword.model import Model
        from openwakeword.utils import download_models
        custom = os.path.join(base_dir, cfg["custom_wake_model"])
        if os.path.exists(custom):
            model = custom
            self.label = cfg["name"]
        else:
            download_models(model_names=[cfg["wake_model"]])
            model = cfg["wake_model"]
            self.label = cfg["wake_model"].replace("_", " ")
        self.model = Model(wakeword_models=[model], inference_framework="onnx")
        self.threshold = cfg["wake_threshold"]

    def heard(self, chunk):
        scores = self.model.predict(chunk)
        if max(scores.values()) >= self.threshold:
            self.model.reset()
            return True
        return False


class PersonalWake:
    """"Hey Winston" detector trained on the user's own voice (tools/train_wake.py).

    openWakeWord's speech-embedding front end turns each 80 ms chunk into features; a small
    classifier trained on the user's recordings scores the last ~1.3 s. Firing needs `need` hits in a
    row, then a short cooldown so one "Hey Winston" can't fire twice.
    """

    def __init__(self, cfg, base_dir, model=None):
        import pickle
        from openwakeword.utils import AudioFeatures
        m = model or pickle.load(open(os.path.join(base_dir, cfg["personal_wake_model"]), "rb"))
        self.scaler, self.clf = m["scaler"], m["clf"]
        self.feats = AudioFeatures(inference_framework="onnx")
        self.threshold = cfg.get("personal_wake_threshold", 0.9)
        self.need = cfg.get("personal_wake_frames", 2)
        self.label = f"Hey {cfg['name']}"
        self.hits = self.cooldown = 0
        # Winston's echo lowers the scores while it talks, so the bar is lower then; a false hit
        # there only makes it stop talking.
        self.speaking_threshold = cfg.get("personal_wake_speaking_threshold", self.threshold)
        self.speaking_need = cfg.get("personal_wake_speaking_frames", self.need)
        # A wake word is ~0.6 s of sound. Clicks, taps and bumps are far shorter, so they can't pass.
        self.min_voiced = cfg.get("personal_wake_min_voiced", 0)
        self.voiced_rms = cfg.get("personal_wake_voiced_rms", 100.0)
        self.levels = collections.deque(maxlen=16)
        # Each wake's audio is kept on this PC so false triggers can be studied and trained away.
        self.events_dir = os.path.join(base_dir, ".runtime", "wake_events") if model is None else None
        self.recent = collections.deque(maxlen=38)  # ~3 s

    def reset(self):
        """Forget recent audio. The detector isn't fed while Winston listens to a request, so its
        buffer would otherwise still hold the old wake word and fire on it again."""
        self.feats.reset()
        self.hits = 0
        self.levels.clear()
        self.recent.clear()

    def score(self, chunk):
        self.feats(chunk)
        f = self.feats.get_features(16).reshape(1, -1)
        return float(self.clf.predict_proba(self.scaler.transform(f))[0, 1])

    def heard(self, chunk, speaking=False):
        s = self.score(chunk)
        self.levels.append(_rms(chunk))
        self.recent.append(chunk.copy())
        if self.cooldown:
            self.cooldown -= 1
            return False
        threshold, need = (self.speaking_threshold, self.speaking_need) if speaking else (self.threshold, self.need)
        self.hits = self.hits + 1 if s >= threshold else 0
        if self.hits >= need:
            self.hits = 0
            voiced = sum(x > self.voiced_rms for x in self.levels)
            if voiced < self.min_voiced:
                return False
            self.cooldown = 15  # ~1.2 s
            self._save_event(s, voiced, speaking)
            return True
        return False

    def _save_event(self, score, voiced, speaking):
        if not self.events_dir:
            return
        try:
            os.makedirs(self.events_dir, exist_ok=True)
            name = time.strftime("%Y%m%d-%H%M%S") + ("-talking" if speaking else "-idle") + ".wav"
            with wave.open(os.path.join(self.events_dir, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(np.concatenate(list(self.recent)).astype(np.int16).tobytes())
            for old in sorted(os.listdir(self.events_dir))[:-30]:
                os.remove(os.path.join(self.events_dir, old))
            print(f"[wake] {name} score {score:.2f}, {voiced} chunks of sound")
        except OSError as e:
            print(f"[wake] couldn't save the wake audio: {e}")


class NameWake:
    """Wake on Winston's name, spotted with a small Whisper model in each burst of speech.

    No custom wake-word model to train, and you can say the request in the same breath:
    "Winston, what's on my plate today?" feed() returns None when the name wasn't heard,
    otherwise whatever was said after it (possibly '').
    """

    def __init__(self, cfg, refine=None):
        _add_cuda_dlls()
        from faster_whisper import WhisperModel
        self.refine = refine  # the more accurate transcriber, for requests said in the same breath
        try:
            self.model = WhisperModel(cfg["wake_whisper_model"], device=cfg["whisper_device"],
                                      compute_type=cfg["whisper_compute_type"])
        except Exception:
            self.model = WhisperModel(cfg["wake_whisper_model"], device="cpu", compute_type="int8")
        self.name = cfg["name"]
        self.rx = re.compile(r"\b(" + "|".join(cfg["wake_words"]) + r")\b[\s,.!?]*", re.I)
        self.interrupt_rx = re.compile(r"\b(stop|wait|hold on|shut up|quiet)\b", re.I)
        self.label = f"Hey {self.name}"
        self.reset()

    def reset(self):
        self.frames, self.active, self.quiet = [], False, 0.0
        self.roll, self.since = [], 0

    def heard_over_speech(self, chunk):
        """While Winston is talking the mic never goes quiet, so instead of waiting for a pause,
        check the last ~2.5 s of audio for the name once a second."""
        self.frames, self.active, self.quiet = [], False, 0.0  # normal pause-based spotting is off
        self.roll = (self.roll + [chunk])[-25:]   # last 2 s
        self.since += 1
        if self.since < 6 or len(self.roll) < 8:  # check twice a second
            return None
        self.since = 0
        audio = np.concatenate(self.roll).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                            condition_on_previous_text=False)
        text = " ".join(s.text for s in segments).strip()
        if text:
            print(f"  [over speech] {text}")
        m = self.rx.search(text) or self.interrupt_rx.search(text)
        if m:
            self.reset()
            return m.group(0).strip(" ,.!?").lower()
        return None

    def feed(self, chunk, noise):
        loud = _rms(chunk) > max(noise * 3.0, SPEECH_FLOOR)
        step = CHUNK / RATE
        self.roll, self.since = [], 0  # the over-speech window only covers the current reply
        if not self.active:
            self.frames = (self.frames + [chunk])[-LEAD_IN:]
            self.active = loud
            return None
        self.frames.append(chunk)
        self.quiet = 0.0 if loud else self.quiet + step
        length = len(self.frames) * step
        if self.quiet < 0.55 and length < 10:
            return None
        audio = np.concatenate(self.frames).astype(np.float32) / 32768.0
        self.reset()
        if length < 0.5:
            return None
        # no hotwords= here: biasing towards the name makes Whisper swallow it at the start
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                            condition_on_previous_text=False)
        text = " ".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()
        m = self.rx.search(text)
        if not m:
            return None
        after = text[m.end():].strip(" ,.!?")
        if self.refine and len(after.split()) >= 2:
            better = self.refine(audio)
            m2 = self.rx.search(better)
            after = (better[m2.end():] if m2 else better).strip(" ,.!?") or after
        return after


def record_utterance(mic, cfg, max_wait=None, on_level=None):
    """Record from the moment speech starts until a pause. Returns float32 audio or None."""
    threshold = max(mic.noise * 3.0, SPEECH_FLOOR)
    frames, started, quiet, waited, total = [], False, 0.0, 0.0, 0.0
    step = CHUNK / RATE
    max_wait = max_wait or cfg["max_wait_for_speech"]
    while True:
        chunk = mic.read(timeout=2)
        rms = _rms(chunk)
        if on_level:
            on_level(rms / 3000.0)
        loud = rms > threshold
        if not started:
            waited += step
            frames = (frames + [chunk])[-LEAD_IN:]  # keep the run-up so soft first words ("Switch...") aren't clipped
            if loud:
                started = True
            elif waited > max_wait:
                return None
            continue
        frames.append(chunk)
        total += step
        quiet = 0.0 if loud else quiet + step
        if quiet >= cfg["silence_seconds"] or total >= cfg["max_utterance_seconds"]:
            break
    return np.concatenate(frames).astype(np.float32) / 32768.0


HALLUCINATION = re.compile(r"^\W*(thanks? (you )?for watching|please subscribe|subscribe to|"
                           r"see you (in the )?next (video|time)|you)\W*$", re.I)


class Transcriber:
    def __init__(self, cfg):
        _add_cuda_dlls()
        # the vocabulary prompt roughly halves errors on project names
        self.prompt = cfg.get("vocabulary")
        self.beam = cfg.get("whisper_beam_size", 5)
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(cfg["whisper_model"], device=cfg["whisper_device"],
                                      compute_type=cfg["whisper_compute_type"])
        except Exception as e:  # no usable GPU: fall back to CPU
            print(f"[whisper] GPU load failed ({e}); using CPU")
            self.model = WhisperModel(cfg["whisper_model"], device="cpu", compute_type="int8")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, language="en", beam_size=self.beam, vad_filter=True,
                                            initial_prompt=self.prompt, condition_on_previous_text=False)
        # On noise or music Whisper invents text ("Thanks for watching!", "oh, oh, oh, oh"), so
        # drop anything it isn't confident about or that repeats itself.
        text = " ".join(s.text for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0
                        and s.compression_ratio < 2.4).strip()
        return "" if HALLUCINATION.search(text) else text


def speakable(text):
    """Turn Claude's markdown into something that sounds right read aloud."""
    text = re.sub(r"```.*?```", " I've put the code on screen. ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"^\s*[#>*\-]+\s*", "", text, flags=re.M)
    text = re.sub(r"[*_#|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _elevenlabs_key(base_dir):
    """ELEVENLABS_API_KEY from the environment or from assistant\\.env (KEY=value lines)."""
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"].strip()
    path = os.path.join(base_dir or ".", ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            key, _, value = line.partition("=")
            if key.strip() == "ELEVENLABS_API_KEY" and value.strip():
                return value.strip().strip('"\'')
    return None


class Speaker:
    """Speaks queued text sentence by sentence; the next sentence's audio is fetched while
    the current one plays. stop() cuts it off mid-word and drops anything queued."""

    def __init__(self, cfg, base_dir=None, echo=None):
        self.voice, self.rate = cfg["voice"], cfg["voice_rate"]
        self.echo = echo  # told everything we play, so the mic can subtract it
        self.eleven = cfg.get("elevenlabs", {})
        self.eleven_key = _elevenlabs_key(base_dir) if self.eleven.get("enabled") else None
        if self.eleven.get("enabled") and not self.eleven_key:
            print("[tts] ElevenLabs is enabled but no key found in .env; using the free voice")
        self.cache = {}              # pre-made audio for the stock phrases ("Yes?", "Okay.")
        self.recent = []             # (time, text) Winston said lately, to ignore its own voice
        self.texts = queue.Queue()   # sentences waiting to be synthesised
        self.clips = queue.Queue()   # (generation, audio or text-for-SAPI) waiting to play
        self._gen = 0                # bumped by stop() so stale audio is discarded
        self._pending = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._fetch_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def say(self, text):
        text = speakable(text)
        if not text:
            return
        print(f"\n  Winston: {text}")
        now = time.time()
        self.recent = [(t, s) for t, s in self.recent if now - t < 20] + [(now, text)]
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if sentence.strip():
                with self._lock:
                    self._pending += 1
                self.texts.put((self._gen, sentence))

    def beep(self):
        """Short two-note 'I'm listening' chime; doesn't block."""
        t = np.arange(int(24000 * 0.07)) / 24000
        note = lambda f: np.sin(2 * np.pi * f * t) * np.hanning(len(t))
        tone = (np.concatenate([note(880), note(1320)]) * 6000).astype(np.int16)
        self._stop.clear()  # usually right after an interrupt, which set it
        threading.Thread(target=self._play_audio, args=(tone,), daemon=True).start()

    def said_recently(self, word):
        now = time.time()
        return any(word.lower() in s.lower() for t, s in self.recent if now - t < 20)

    def prepare(self, phrases):
        """Synthesise stock phrases in the background so they play instantly later."""
        def work():
            for p in phrases:
                try:
                    self.cache[p] = self._synthesise(p)
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        with self._lock:
            self._gen += 1
            self._pending = 0
        self._stop.set()
        sd.stop()

    def speaking(self):
        return self._pending > 0

    def wait(self, timeout=120):
        end = time.time() + timeout
        while self.speaking() and time.time() < end:
            time.sleep(0.05)

    def _done_one(self, gen):
        with self._lock:
            if gen == self._gen and self._pending > 0:
                self._pending -= 1

    def _fetch_loop(self):
        while True:
            gen, sentence = self.texts.get()
            if gen != self._gen:
                continue
            try:
                clip = self.cache.get(sentence)
                if clip is None:
                    clip = self._synthesise(sentence)
            except Exception as e:
                print(f"[tts] neural voice failed ({e}); using Windows voice")
                clip = sentence
            self.clips.put((gen, clip))

    def _play_loop(self):
        while True:
            gen, clip = self.clips.get()
            if gen != self._gen:
                continue
            self._stop.clear()
            try:
                if isinstance(clip, str):
                    self._play_sapi(clip)
                else:
                    self._play_audio(clip)
            finally:
                self._done_one(gen)

    def _synthesise(self, text):
        if self.eleven_key:
            try:
                return self._synthesise_eleven(text)
            except Exception as e:  # out of credits, offline, bad key: fall back to the free voice
                print(f"[tts] ElevenLabs failed ({e}); using the free voice")
        return self._synthesise_edge(text)

    def _synthesise_eleven(self, text):
        import urllib.request
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{self.eleven['voice_id']}/stream"
               "?output_format=pcm_24000")
        body = json.dumps({"text": text, "model_id": self.eleven["model"],
                           "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "xi-api-key": self.eleven_key, "Content-Type": "application/json", "Accept": "audio/pcm"})
        with urllib.request.urlopen(req, timeout=20) as r:
            pcm = r.read()
        return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16)  # raw 24 kHz 16-bit mono

    def _synthesise_edge(self, text):
        import edge_tts
        import miniaudio

        async def fetch():
            mp3 = b""
            async for part in edge_tts.Communicate(text, self.voice, rate=self.rate).stream():
                if part["type"] == "audio":
                    mp3 += part["data"]
            return mp3

        pcm = miniaudio.decode(asyncio.run(fetch()), output_format=miniaudio.SampleFormat.SIGNED16,
                               nchannels=1, sample_rate=24000)
        return np.frombuffer(pcm.samples, dtype=np.int16)

    def _play_audio(self, audio, rate=24000):
        # played block by block (10 ms) so each block can be handed to the echo canceller
        # at the moment it reaches the speaker
        pos = [0]
        done = threading.Event()

        def cb(outdata, frames, t, status):
            block = audio[pos[0]:pos[0] + frames] if not self._stop.is_set() else audio[:0]
            outdata[:len(block), 0] = block
            outdata[len(block):, 0] = 0
            pos[0] += frames
            if self.echo:
                self.echo.render(outdata[:, 0], rate)
            if len(block) < frames:
                raise sd.CallbackStop

        with sd.OutputStream(samplerate=rate, channels=1, dtype="int16", blocksize=rate // 100,
                             callback=cb, finished_callback=done.set):
            while not done.wait(0.03):
                if self._stop.is_set():
                    break

    def _play_sapi(self, text):
        safe = text.replace("'", "''")
        p = subprocess.Popen(["powershell", "-NoProfile", "-Command",
                              "Add-Type -AssemblyName System.Speech; "
                              f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"],
                             creationflags=subprocess.CREATE_NO_WINDOW)
        while p.poll() is None:
            if self._stop.is_set():
                p.kill()
                break
            time.sleep(0.05)
