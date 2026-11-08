"""Train a personal "Winston" / "Hey Winston" detector on your own recordings.

It reuses openWakeWord's speech-embedding front end (the same one behind "Hey Jarvis") and fits a
small classifier on top: positives are you saying "Hey Winston", negatives are everything else you
recorded, Winston's own voice and synthetic speech full of sound-alikes.

    python tools\\train_wake.py          -> trains, reports held-out accuracy, saves models\\hey_winston.pkl
"""
import asyncio
import glob
import json
import os
import pickle
import shutil
import sys
import wave

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
RATE = 16000
WIN = int(RATE * 2.0)  # gives 16 embedding frames, the same span PersonalWake scores live
SAMPLES = os.path.join(BASE, ".runtime", "wake_samples")
RECORDINGS = os.path.join(BASE, ".runtime", "recordings")
OUT = os.path.join(BASE, "models", "hey_winston.pkl")
rng = np.random.default_rng(7)


def load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def utterances(audio, floor=60.0, gap=0.35, min_len=0.35):
    """(start, end) seconds of each burst of speech, found by loudness."""
    step = 160  # 10 ms
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(step) / step, mode="same")[::step])
    loud = rms > max(floor, np.percentile(rms, 20) * 3)
    out, start, quiet = [], None, 0
    for i, on in enumerate(loud):
        if on:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * 0.01 >= gap:
                end = i - quiet + 1
                if (end - start) * 0.01 >= min_len:
                    out.append((start * 0.01, end * 0.01))
                start, quiet = None, 0
    return out


def find_wake_ends():
    """Where each 'Hey Winston' ends in the positive takes.

    Every utterance in those takes is the wake phrase by construction, so loudness alone finds them.
    (Whisper can't be used to check: it can write "Hey Winston" as "Havenston", "Heaven's done",
    "instead"... which is exactly why wake detection has to work on sound, not text.)
    Longer utterances are "Hey Winston, open the website session": the phrase is their first part.
    """
    stream = np.concatenate([load(p) for p in sorted(glob.glob(os.path.join(SAMPLES, "pos_*.wav")))])
    utts = utterances(stream)
    short = [e - s for s, e in utts if e - s <= 1.4]
    phrase = float(np.median(short)) if short else 0.85
    ends = [e if e - s <= 1.4 else s + phrase for s, e in utts]
    print(f"{len(utts)} utterances; a 'Hey Winston' takes about {phrase:.2f}s")
    return stream, ends


def find_bare_ends():
    """Where each lone "Winston" ends in the takes recorded by record_bare.py."""
    stream = np.concatenate([load(p) for p in sorted(glob.glob(os.path.join(SAMPLES, "bare_*.wav")))])
    ends = [e for s, e in utterances(stream) if 0.3 <= e - s <= 1.2]
    print(f"{len(ends)} lone 'Winston' utterances")
    return stream, ends


def windows_ending_at(stream, t_end, jitter):
    out = []
    for j in jitter:
        e = int((t_end + j) * RATE)
        s = e - WIN
        if s >= 0 and e <= len(stream):
            out.append(stream[s:e])
    return out


def sliding(audio, hop=0.25):
    h = int(hop * RATE)
    return [audio[i:i + WIN] for i in range(0, len(audio) - WIN, h)]


def augment(clip, noise_bank):
    gain = rng.uniform(0.4, 2.5)
    x = clip * gain
    if noise_bank and rng.random() < 0.7:
        n = noise_bank[rng.integers(len(noise_bank))]
        x = x + n * rng.uniform(0.2, 1.0)
    x = x + rng.normal(0, rng.uniform(5, 40), len(x))
    return np.clip(x, -32768, 32767)


def synth_negatives():
    """Synthetic sentences with sound-alikes, several voices (edge-tts, cached)."""
    cache = os.path.join(BASE, ".runtime", "synthetic_negatives.npy")
    if os.path.exists(cache):
        return list(np.load(cache, allow_pickle=True))
    import edge_tts
    import miniaudio
    lines = ["Hey Wilson, how are you", "the western side of town", "let's win this game", "is it instant",
             "hey there, what's up", "winter is coming soon", "Winchester is lovely", "hey, wait a second",
             "switch to the billing session", "open the website session", "ask the agent for ideas",
             "when's the meeting", "Weston and Kingston", "hey listen to this", "the window is open",
             "we won the match", "Hey Jarvis", "hey Siri", "okay Google", "Alexa, play music",
             "send the invoice by Wednesday", "the wind is strong today", "Hey man, nice work"]
    voices = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-GB-RyanNeural", "en-US-GuyNeural"]
    out = []
    for v in voices:
        for line in lines:
            async def f():
                b = b""
                async for p in edge_tts.Communicate(line, v).stream():
                    if p["type"] == "audio":
                        b += p["data"]
                return b
            try:
                mp3 = asyncio.run(f())
            except Exception:  # the TTS service occasionally returns nothing; skip that line
                continue
            pcm = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=1, sample_rate=RATE)
            out.append(np.frombuffer(pcm.samples, dtype=np.int16).astype(np.float32) * 0.3)
    np.save(cache, np.array(out, dtype=object), allow_pickle=True)
    return out


def embed(clips, feats):
    x = np.stack([np.pad(c, (0, max(0, WIN - len(c))))[:WIN] for c in clips]).astype(np.int16)
    e = feats.embed_clips(x, batch_size=64)  # (n, frames, 96)
    assert e.shape[1] >= 16, f"clips give only {e.shape[1]} embedding frames; lengthen WIN"
    return e[:, -16:, :].reshape(len(clips), -1)


def main():
    from openwakeword.utils import AudioFeatures
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    stream, ends = find_wake_ends()
    print(f"found {len(ends)} 'Winston' occurrences in the positive takes")
    bare_stream, bare_ends = find_bare_ends()

    room = load(os.path.join(SAMPLES, "neg_room.wav"))
    winston_voice = load(os.path.join(RECORDINGS, "winston_reference.wav")) * 0.2
    noise_bank = sliding(room, 0.5) + sliding(winston_voice, 0.5)

    # Negative audio, split by *source* so the test uses recordings the model never saw.
    read = load(os.path.join(SAMPLES, "neg_read.wav"))
    half = len(read) // 2
    rec = {os.path.basename(p): load(p) for p in glob.glob(os.path.join(RECORDINGS, "*.wav"))}
    test_rec = {"barge3.wav", "cmd4.wav", "cmd5.wav", "cmd6.wav"}
    synth = synth_negatives()
    synth_test = synth[3::4]
    synth_train = [s for i, s in enumerate(synth) if i % 4 != 3]
    neg_train_audio = [read[:half], room] + [a for n, a in rec.items() if n not in test_rec]
    neg_test_audio = [read[half:]] + [a for n, a in rec.items() if n in test_rec]
    neg_test_audio.append(np.concatenate([np.r_[s, np.zeros(RATE // 2)] for s in synth_test]))

    neg_train = [w for a in neg_train_audio for w in sliding(a, 0.1)]
    neg_train += [augment(w, noise_bank) for w in neg_train[::2]]
    for i in range(0, len(stream) - WIN, int(0.25 * RATE)):  # the takes, away from any 'Winston'
        if all(abs((i + WIN) / RATE - e) > 1.0 for e in ends):
            neg_train.append(stream[i:i + WIN])
    for i in range(0, len(bare_stream) - WIN, int(0.25 * RATE)):
        if all(abs((i + WIN) / RATE - e) > 1.0 for e in bare_ends):
            neg_train.append(bare_stream[i:i + WIN])
    neg_train += [w for s in synth_train for w in sliding(np.pad(s, (WIN // 2, WIN // 2)), 0.3)]

    test_idx = set(range(2, len(ends), 5))  # every 5th 'Hey Winston' is held out
    pos_train = []
    for k, e in enumerate(ends):
        if k not in test_idx:
            for c in windows_ending_at(stream, e, [0.0, 0.08, 0.16, 0.24, 0.32]):
                pos_train += [c] + [augment(c, noise_bank) for _ in range(6)]
    for s_, _ in utterances(stream):  # "Hey" on its own must not fire
        neg_train += windows_ending_at(stream, s_ + 0.25, [0.0, 0.08, 0.16])
    bare_test = set(range(2, len(bare_ends), 5))  # every 5th lone 'Winston' is held out too
    for k, e in enumerate(bare_ends):
        if k not in bare_test:
            for c in windows_ending_at(bare_stream, e, [0.0, 0.08, 0.16, 0.24, 0.32]):
                pos_train += [c] + [augment(c, noise_bank) for _ in range(6)]
    print(f"train: {len(pos_train)} positive / {len(neg_train)} negative windows")

    feats = AudioFeatures(inference_framework="onnx")
    X = np.vstack([embed(pos_train, feats), embed(neg_train, feats)])
    y = np.r_[np.ones(len(pos_train)), np.zeros(len(neg_train))]

    def fit(X, y):
        scaler = StandardScaler().fit(X)
        return {"scaler": scaler, "frames": 16,
                "clf": LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
                .fit(scaler.transform(X), y)}

    # live-style test: stream audio through the detector exactly as Winston would hear it
    from winston.audio import CHUNK, PersonalWake
    model = fit(X, y)
    neg_minutes = sum(len(a) for a in neg_test_audio) / RATE / 60
    for th, need in ((0.8, 3), (0.9, 2), (0.9, 3), (0.95, 3)):
        if True:
            cfg = {"name": "Winston", "personal_wake_threshold": th, "personal_wake_frames": need}
            def catches(source, times, idx, lead, voice=False):
                n = 0
                for k in sorted(idx):
                    e = times[k]
                    clip = source[max(0, int((e - lead) * RATE)):int((e + 0.9) * RATE)]
                    if voice:  # what is left of Winston's own voice after echo cancelling
                        v = winston_voice[(k * RATE) % (len(winston_voice) - len(clip)):][:len(clip)]
                        clip = clip + v * 0.5
                    w = PersonalWake(cfg, BASE, model)
                    clip = np.r_[np.zeros(RATE, dtype=np.float32), clip]  # let the feature buffer fill
                    n += any(w.heard(clip[i:i + CHUNK].astype(np.int16))
                             for i in range(0, len(clip) - CHUNK, CHUNK))
                return n
            caught = catches(stream, ends, test_idx, 1.8)
            bare_caught = catches(bare_stream, bare_ends, bare_test, 1.0)
            bare_voice = catches(bare_stream, bare_ends, bare_test, 1.0, voice=True)
            hey_voice = catches(stream, ends, test_idx, 1.8, voice=True)
            per_source = []
            for a in neg_test_audio:
                w = PersonalWake(cfg, BASE, model)
                per_source.append(sum(w.heard(a[i:i + CHUNK].astype(np.int16)) for i in range(0, len(a) - CHUNK, CHUNK)))
            false = sum(per_source)
            print(f"  threshold {th}, {need} hits in a row: 'Hey Winston' {caught}/{len(test_idx)}; "
                  f"'Winston' {bare_caught}/{len(bare_test)}, over Winston's voice "
                  f"{bare_voice}/{len(bare_test)} ('Hey Winston' {hey_voice}/{len(test_idx)}); "
                  f"{false} false wakes in {neg_minutes:.1f} min {per_source}")

    # final model: train on everything, including what was held out
    held = [c for k in test_idx for c in windows_ending_at(stream, ends[k], [0.0, 0.16, 0.32])]
    held += [c for k in bare_test for c in windows_ending_at(bare_stream, bare_ends[k], [0.0, 0.16, 0.32])]
    X = np.vstack([X, embed(held, feats), embed([w for a in neg_test_audio for w in sliding(a)], feats)])
    y = np.r_[y, np.ones(len(held)), np.zeros(len(X) - len(y) - len(held))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):  # keep the previous model so a bad retrain can be undone
        shutil.copy(OUT, OUT.replace(".pkl", "_previous.pkl"))
    pickle.dump(fit(X, y), open(OUT, "wb"))
    print("saved", OUT)


if __name__ == "__main__":
    main()
