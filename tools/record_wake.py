"""Records you saying "Hey Winston" (and some normal speech) to train a personal wake-word model.

About 3 minutes. Saves to assistant\\.runtime\\wake_samples\\ — nothing leaves this PC.
"""
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

RATE = 16000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runtime", "wake_samples")
ROUNDS = [
    ("normal voice, as you'd normally call it", 10, "Hey Winston"),
    ("a bit quieter, or leaning back from the laptop", 8, "Hey Winston"),
    ("followed by a request, in one breath", 7, "Hey Winston, open the website session"),
    ("quick and casual", 5, "Hey Winston"),
]
NEGATIVE_TEXT = """Read this aloud at a normal pace (about 40 seconds):

  "Hey, can you check the Western Union payment? I was thinking about Wilson's idea
   for the new app. Let's win this one before the weekend. Is it instant, or does
   it take a while? Switch to the billing session and tell it to start building.
   The winter sale starts on Wednesday. Hey there, how's the weather in Winchester?
   I want the sales dashboard finished, and ask the agent for three new ideas.
   Wait, hold on, stop for a second. Okay, that sounds good, thanks."

Press Enter when you're ready, and Enter again when you've finished."""


def save(name, audio):
    with wave.open(os.path.join(OUT, name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


class Recorder:
    """Keeps the mic open the whole time (Windows takes ~0.7 s to wake a newly opened mic)."""

    def __init__(self):
        self.buf = []
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=1600,
                                     callback=lambda d, f, t, s: self.buf.append(d[:, 0].copy()))
        self.stream.start()
        time.sleep(1.5)

    def take(self):
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, dtype=np.int16)
        self.buf = []
        return audio


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = Recorder()
    print("\n=== 'Hey Winston' wake-word recording (about 3 minutes) ===\n")
    n = 0
    for how, count, phrase in ROUNDS:
        input(f"\nNext: say \"{phrase}\" {count} times — {how}.\n"
              f"A prompt appears every 3 seconds. Press Enter to start...")
        rec.take()
        for i in range(count):
            n += 1
            print(f"   [{n:2d}]  say:  {phrase}", flush=True)
            time.sleep(3.0)
            save(f"pos_{n:02d}.wav", rec.take())
        print("   done.")
    print("\n" + NEGATIVE_TEXT)
    input()
    rec.take()
    print("   recording... press Enter when finished.", flush=True)
    input()
    save("neg_read.wav", rec.take())
    print("\n   Last one: 20 seconds of silence / normal room noise. Just stay quiet...", flush=True)
    rec.take()
    time.sleep(20)
    save("neg_room.wav", rec.take())
    print(f"\nAll done — {n} samples saved in {OUT}\nTell Claude you're done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
