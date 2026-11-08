"""Records your real voice so Winston's speech recognition and interrupts can be tuned on it.

Part A: read 6 commands aloud.  Part B: say "Hey Jarvis" while Winston is talking (3 times).
Saves WAV files to assistant\\.runtime\\recordings\\ â€” nothing leaves this PC.
"""
import asyncio
import json
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

RATE = 16000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runtime", "recordings")
COMMANDS = [
    "Switch to the billing session.",
    "Write a prompt to fix the login bug on my website.",
    "Ask the research agent for three new ideas.",
    "What did Claude say in the website session?",
    "Tell it to start building the sales dashboard.",
    "Open Chrome and go to my dashboard.",
]
REPLY = ("The research agent says the latest work was a study on turning long videos into shorts. The plain clip "
         "channels were a no-go, and the paid campaign version was only good enough to test, because it "
         "needs a few hours of your time each week. The last full recommendation is in the report.")


def save(name, audio, rate=RATE):
    with wave.open(os.path.join(OUT, name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())


def record_until_pause(max_s=10):
    """Record from the first sound until 1.2 s of quiet."""
    chunks, started, quiet = [], False, 0.0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=1280) as s:
        floor = np.sqrt(np.mean(s.read(4000)[0].astype(np.float32) ** 2)) + 1
        t0 = time.time()
        while time.time() - t0 < max_s:
            c = s.read(1280)[0][:, 0]
            loud = np.sqrt(np.mean(c.astype(np.float32) ** 2)) > max(floor * 3, 300)
            chunks.append(c)
            started = started or loud
            quiet = 0 if loud else quiet + 0.08
            if started and quiet > 1.2:
                break
    return np.concatenate(chunks)


def winston_voice():
    import edge_tts
    import miniaudio

    async def f():
        b = b""
        async for p in edge_tts.Communicate(REPLY, "en-GB-RyanNeural", rate="+5%").stream():
            if p["type"] == "audio":
                b += p["data"]
        return b
    pcm = miniaudio.decode(asyncio.run(f()), output_format=miniaudio.SampleFormat.SIGNED16,
                           nchannels=1, sample_rate=RATE)
    return np.frombuffer(pcm.samples, dtype=np.int16)


def play_and_record(ref, go_at):
    """Play ref and record the mic in the same audio callback, so they stay sample-aligned."""
    mic = np.zeros(len(ref) + RATE, dtype=np.int16)
    pos = [0]

    def cb(indata, outdata, frames, t, status):
        p = pos[0]
        block = ref[p:p + frames]
        outdata[:len(block), 0] = block
        outdata[len(block):, 0] = 0
        mic[p:p + frames] = indata[:frames, 0][: max(0, min(frames, len(mic) - p))]
        pos[0] += frames

    with sd.Stream(samplerate=RATE, channels=1, dtype="int16", blocksize=160, callback=cb):
        t0 = time.time()
        shown = False
        while pos[0] < len(mic) - 160:
            if not shown and time.time() - t0 >= go_at:
                print("\n        >>>>>>>>>>   GO  â€”  say  \"HEY JARVIS\"  now   <<<<<<<<<<\n", flush=True)
                shown = True
            time.sleep(0.02)
    return mic


def main():
    os.makedirs(OUT, exist_ok=True)
    print("\n=== Winston voice test (about 2 minutes) ===")
    print("Speak normally, at your usual distance from the laptop. Nothing is uploaded.\n")

    print("PART A: read each command aloud when it appears.\n")
    for i, cmd in enumerate(COMMANDS, 1):
        input(f"  [{i}/6] Press Enter, then say:   {cmd}")
        print("        recording...", flush=True)
        save(f"cmd{i}.wav", record_until_pause())
        print("        got it.\n")

    print("PART B: Winston will talk. When you see GO, say \"Hey Jarvis\" over it (normal voice).\n")
    ref = winston_voice()
    save("winston_reference.wav", ref)
    meta = {"commands": COMMANDS, "go_at": {}}
    for i, go in enumerate((3.0, 6.0, 9.0), 1):
        input(f"  [{i}/3] Press Enter to start Winston talking...")
        mic = play_and_record(ref, go)
        save(f"barge{i}.wav", mic)
        meta["go_at"][f"barge{i}.wav"] = go
        print("        got it.\n")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"All done. Recordings saved in {OUT}\nYou can close this window and tell Claude you're done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
