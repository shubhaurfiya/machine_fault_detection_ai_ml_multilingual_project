from pathlib import Path
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).parent
OUT = ROOT / "sample_audio"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
sr = 16000
duration = 2.0
t = np.arange(int(sr * duration)) / sr

def make_signal(kind):
    # These are synthetic academic demo signals, not recordings of real machines.
    if kind == "Normal":
        base = 220 + rng.normal(0, 4)
        sig = 0.55*np.sin(2*np.pi*base*t) + 0.12*np.sin(2*np.pi*440*t)
        noise = 0.025*rng.normal(size=len(t))
    elif kind == "Bearing Fault":
        base = 220 + rng.normal(0, 5)
        impulses = np.zeros(len(t))
        for k in range(0, len(t), int(sr/55)):
            width = min(80, len(t)-k)
            impulses[k:k+width] += np.hanning(width) * 0.9
        sig = 0.45*np.sin(2*np.pi*base*t) + 0.55*impulses*np.sin(2*np.pi*1800*t)
        noise = 0.08*rng.normal(size=len(t))
    elif kind == "Motor Fault":
        base = 300 + rng.normal(0, 7)
        sig = 0.42*np.sin(2*np.pi*base*t) + 0.32*np.sin(2*np.pi*(base*2)*t)
        sig += 0.25*np.sin(2*np.pi*1200*t)
        noise = 0.10*rng.normal(size=len(t))
    else:  # Fan Fault
        base = 120 + rng.normal(0, 4)
        sig = 0.5*np.sin(2*np.pi*base*t) + 0.25*np.sin(2*np.pi*360*t)
        sig += 0.15*np.sin(2*np.pi*900*t)
        noise = 0.06*rng.normal(size=len(t))

    x = sig + noise
    x = x / max(np.max(np.abs(x)), 1e-9)
    return (x * 32767).astype(np.int16)

# Create multiple variations per class.
classes = ["Normal", "Bearing Fault", "Motor Fault", "Fan Fault"]
for cls in classes:
    prefix = cls.replace(" ", "-")
    for i in range(30):
        sr_i = sr
        audio = make_signal(cls)
        # Small time-independent amplitude variation
        audio = np.clip(audio.astype(np.float64) * rng.uniform(0.85, 1.0), -32767, 32767).astype(np.int16)
        wavfile.write(OUT / f"{prefix}_{i:02d}.wav", sr_i, audio)

print(f"Generated {len(list(OUT.glob('*.wav')))} synthetic WAV files in {OUT}")
print("Note: replace these demo sounds with real labeled machine recordings for a real-world project.")
