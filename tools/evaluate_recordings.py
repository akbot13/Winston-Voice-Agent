"""Score speech models and the interrupt on your real recordings (from record_test.py)."""
import difflib
import json
import os
import re
import sys
import time
import wave

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
REC = os.path.join(BASE, ".runtime", "recordings")
from winston.settings import load  # noqa: E402

cfg = load(BASE)
meta = json.load(open(os.path.join(REC, "meta.json")))


def load(name):
    with wave.open(os.path.join(REC, name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def words(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    sm = difflib.SequenceMatcher(None, r, h)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal") / len(r)


def stt_report():
    from winston.audio import _add_cuda_dlls
    _add_cuda_dlls()
    from faster_whisper import WhisperModel
    cmds = [(c, load(f"cmd{i}.wav").astype(np.float32) / 32768) for i, c in enumerate(meta["commands"], 1)]

    def run(label, fn):
        fn(cmds[0][1])  # warm-up
        errs, secs = [], []
        print(f"\n{label}")
        for ref, a in cmds:
            t = time.time()
            out = fn(a)
            secs.append(time.time() - t)
            errs.append(wer(ref, out))
            print(f"   {'OK ' if errs[-1] == 0 else 'ERR'} {secs[-1]:.2f}s  {out}")
        print(f"   => word error {np.mean(errs):.0%}, avg {np.mean(secs):.2f}s")

    for name in ["small.en", "large-v3-turbo"]:
        m = WhisperModel(name, device="cuda", compute_type="int8_float16")
        run(f"Whisper {name} + vocabulary", lambda a: " ".join(
            s.text for s in m.transcribe(a, language="en", beam_size=5, vad_filter=True,
                                         initial_prompt=cfg["vocabulary"],
                                         condition_on_previous_text=False)[0]).strip())
        del m

    import sherpa_onnx
    d = os.path.join(BASE, "models", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
        tokens=f"{d}/tokens.txt", model_type="nemo_transducer", num_threads=6)

    def parakeet(a):
        s = rec.create_stream()
        s.accept_waveform(16000, a)
        rec.decode_stream(s)
        return s.result.text
    run("NVIDIA Parakeet TDT 0.6B v2 (CPU)", parakeet)


def barge_report():
    from winston.audio import CHUNK, EchoCanceller, WakeWord
    ref = load("winston_reference.wav")
    print("\nInterrupt test: 'Hey Jarvis' over Winston's voice (wake score; 0.5 = triggers)")
    for name, go in meta["go_at"].items():
        mic = load(name)
        for use_aec in (False, True):
            w = WakeWord(cfg, BASE)
            aec = EchoCanceller() if use_aec else None
            scores = []
            for i in range(0, len(mic) - CHUNK, CHUNK):
                c = mic[i:i + CHUNK]
                if aec:
                    r = ref[i:i + CHUNK]
                    aec.render(np.pad(r, (0, CHUNK - len(r))), 16000)
                    c = aec.capture(c)
                scores.append(max(w.model.predict(c).values()))
            t = np.arange(len(scores)) * CHUNK / 16000
            near = [s for s, x in zip(scores, t) if go <= x <= go + 3.5]
            far = [s for s, x in zip(scores, t) if not go - 0.5 <= x <= go + 4]
            print(f"   {name} aec={'on ' if use_aec else 'off'}  after GO max {max(near):.2f}   "
                  f"elsewhere max {max(far) if far else 0:.2f}   level {np.sqrt(np.mean(mic.astype(float)**2)):.0f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["stt", "barge"]
    if "barge" in which:
        barge_report()
    if "stt" in which:
        stt_report()
