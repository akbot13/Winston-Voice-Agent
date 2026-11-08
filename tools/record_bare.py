"""Records you saying just "Winston", so the wake model learns the single word too.

About 2 minutes. Saves to .runtime\\wake_samples\\bare_XX.wav; nothing leaves this PC.
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
    ("normal voice, as you'd call someone's name", 12),
    ("a bit quieter, or leaning back from the laptop", 8),
    ("quick and casual", 6),
    ("louder, as if Winston is talking and you're cutting in", 4),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    buf = []
    stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=1600,
                            callback=lambda d, f, t, s: buf.append(d[:, 0].copy()))
    stream.start()
    time.sleep(1.5)  # Windows takes ~0.7 s to wake a newly opened mic
    print("\n=== Recording just the word \"Winston\" (about 2 minutes) ===\n")
    n = 0
    for how, count in ROUNDS:
        input(f"\nNext: say only \"Winston\", {count} times: {how}.\n"
              "A prompt appears every 2.5 seconds. Press Enter to start...")
        buf.clear()
        for _ in range(count):
            n += 1
            print(f"   [{n:2d}]  say:  Winston", flush=True)
            time.sleep(2.5)
            audio = np.concatenate(buf)
            buf.clear()
            with wave.open(os.path.join(OUT, f"bare_{n:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(audio.astype(np.int16).tobytes())
        print("   done.")
    print(f"\nAll done: {n} samples saved. Tell Claude you're done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
