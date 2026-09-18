from pathlib import Path
import numpy as np
import joblib
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).parent
DATA = ROOT / "sample_audio"
MODEL = ROOT / "models" / "machine_fault_model.pkl"

def features(audio, sr):
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(audio[:-1] * audio[1:] < 0)

    freqs, _, Z = stft(audio, fs=sr, nperseg=1024)
    mag = np.abs(Z) + 1e-12
    spectrum = np.mean(mag, axis=1)
    total = spectrum.sum()
    centroid = (freqs * spectrum).sum() / total
    bandwidth = np.sqrt(((freqs - centroid)**2 * spectrum).sum() / total)
    cumulative = np.cumsum(spectrum)
    idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    rolloff = freqs[min(idx, len(freqs)-1)]
    dominant = freqs[np.argmax(spectrum)]
    return [rms, zcr, centroid, bandwidth, rolloff, dominant]

X, y = [], []
for wav in DATA.glob("*.wav"):
    sr, audio = wavfile.read(wav)
    X.append(features(audio, sr))
    y.append(wav.stem.split("_")[0].replace("-", " "))

X = np.array(X)
y = np.array(y)

if len(set(y)) < 2 or len(X) < 20:
    raise RuntimeError("Not enough training samples. Run generate_dataset.py first.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(y_test, pred))

joblib.dump(model, MODEL)
print(f"Saved model to: {MODEL}")
