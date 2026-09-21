import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os

from gtts import gTTS

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    RandomForestClassifier = None
    StandardScaler = None
    Pipeline = None


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Machine Fault Detection",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 AI-Based Machine Fault Detection")
st.subheader("Audio Signal Analysis & Predictive Maintenance")


# =========================================================
# MULTI-LANGUAGE FAULT MESSAGES
# =========================================================

LANGUAGES = {
    "English": "en",
    "Kannada": "kn",
    "Hindi": "hi"
}


FAULTS = {

    "Normal": {

        "severity": "Normal",

        "location": {
            "English": "No abnormal location detected",
            "Kannada": "ಯಾವುದೇ ಅಸಾಮಾನ್ಯ ಸಮಸ್ಯೆಯ ಸ್ಥಳ ಪತ್ತೆಯಾಗಿಲ್ಲ",
            "Hindi": "किसी असामान्य समस्या का स्थान नहीं मिला"
        },

        "recommendation": {
            "English":
                "Machine operation appears normal. Continue routine inspection.",

            "Kannada":
                "ಯಂತ್ರದ ಕಾರ್ಯಾಚರಣೆ ಸಾಮಾನ್ಯವಾಗಿದೆ. ನಿಯಮಿತ ಪರಿಶೀಲನೆಯನ್ನು ಮುಂದುವರಿಸಿ.",

            "Hindi":
                "मशीन का संचालन सामान्य है। नियमित निरीक्षण जारी रखें।"
        },

        "voice": {
            "English":
                "The machine condition is normal. No major abnormal sound was detected.",

            "Kannada":
                "ಯಂತ್ರದ ಸ್ಥಿತಿ ಸಾಮಾನ್ಯವಾಗಿದೆ. ಯಾವುದೇ ಪ್ರಮುಖ ಅಸಾಮಾನ್ಯ ಶಬ್ದ ಪತ್ತೆಯಾಗಿಲ್ಲ.",

            "Hindi":
                "मशीन की स्थिति सामान्य है। कोई बड़ी असामान्य आवाज़ नहीं मिली।"
        }
    },


    "Bearing Fault": {

        "severity": "Critical",

        "location": {
            "English": "Motor bearing / rotating shaft",
            "Kannada": "ಮೋಟಾರ್ ಬೇರಿಂಗ್ / ತಿರುಗುವ ಶಾಫ್ಟ್",
            "Hindi": "मोटर बेयरिंग / घूमने वाला शाफ्ट"
        },

        "recommendation": {
            "English":
                "Inspect bearing lubrication, bearing wear and shaft alignment. Replace the bearing if damaged.",

            "Kannada":
                "ಬೇರಿಂಗ್‌ನ ಲೂಬ್ರಿಕೇಶನ್, ಹಾನಿ ಮತ್ತು ಶಾಫ್ಟ್ ಅಲೈನ್‌ಮೆಂಟ್ ಪರಿಶೀಲಿಸಿ. ಬೇರಿಂಗ್ ಹಾನಿಯಾಗಿದ್ದರೆ ಬದಲಾಯಿಸಿ.",

            "Hindi":
                "बेयरिंग की लुब्रिकेशन, घिसाव और शाफ्ट अलाइनमेंट की जांच करें। बेयरिंग खराब होने पर उसे बदलें।"
        },

        "voice": {
            "English":
                "A possible bearing fault is detected. The problem is likely near the motor bearing or rotating shaft. Please inspect the bearing and lubrication.",

            "Kannada":
                "ಬೇರಿಂಗ್‌ನಲ್ಲಿ ಸಂಭವನೀಯ ದೋಷ ಪತ್ತೆಯಾಗಿದೆ. ಸಮಸ್ಯೆಯು ಮೋಟಾರ್ ಬೇರಿಂಗ್ ಅಥವಾ ತಿರುಗುವ ಶಾಫ್ಟ್ ಬಳಿ ಇರಬಹುದು. ಬೇರಿಂಗ್ ಮತ್ತು ಲೂಬ್ರಿಕೇಶನ್ ಪರಿಶೀಲಿಸಿ.",

            "Hindi":
                "संभावित बेयरिंग खराबी का पता चला है। समस्या मोटर बेयरिंग या घूमने वाले शाफ्ट के पास हो सकती है। बेयरिंग और लुब्रिकेशन की जांच करें।"
        }
    },


    "Loose Connection": {

        "severity": "Warning",

        "location": {
            "English": "Electrical connection / motor terminal",
            "Kannada": "ವಿದ್ಯುತ್ ಸಂಪರ್ಕ / ಮೋಟಾರ್ ಟರ್ಮಿನಲ್",
            "Hindi": "विद्युत कनेक्शन / मोटर टर्मिनल"
        },

        "recommendation": {
            "English":
                "Switch off the power safely and inspect terminals, wiring and loose connectors.",

            "Kannada":
                "ಸುರಕ್ಷಿತವಾಗಿ ವಿದ್ಯುತ್ ಸಂಪರ್ಕವನ್ನು ಆಫ್ ಮಾಡಿ. ಟರ್ಮಿನಲ್‌ಗಳು, ವೈರ್‌ಗಳು ಮತ್ತು ಸಡಿಲ ಸಂಪರ್ಕಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",

            "Hindi":
                "सुरक्षित रूप से बिजली बंद करें। टर्मिनल, वायरिंग और ढीले कनेक्शन की जांच करें।"
        },

        "voice": {
            "English":
                "A possible loose electrical connection is detected. Check the motor terminals and wiring after safely switching off the power.",

            "Kannada":
                "ಸಂಭವನೀಯ ಸಡಿಲ ವಿದ್ಯುತ್ ಸಂಪರ್ಕ ಪತ್ತೆಯಾಗಿದೆ. ವಿದ್ಯುತ್ ಅನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಆಫ್ ಮಾಡಿದ ನಂತರ ಮೋಟಾರ್ ಟರ್ಮಿನಲ್ ಮತ್ತು ವೈರಿಂಗ್ ಪರಿಶೀಲಿಸಿ.",

            "Hindi":
                "संभावित ढीला विद्युत कनेक्शन पाया गया है। बिजली सुरक्षित रूप से बंद करने के बाद मोटर टर्मिनल और वायरिंग की जांच करें।"
        }
    },


    "Motor Imbalance": {

        "severity": "Warning",

        "location": {
            "English": "Motor shaft / rotating assembly",
            "Kannada": "ಮೋಟಾರ್ ಶಾಫ್ಟ್ / ತಿರುಗುವ ಭಾಗ",
            "Hindi": "मोटर शाफ्ट / घूमने वाला भाग"
        },

        "recommendation": {
            "English":
                "Check shaft alignment, motor mounting, rotor balance and vibration.",

            "Kannada":
                "ಶಾಫ್ಟ್ ಅಲೈನ್‌ಮೆಂಟ್, ಮೋಟಾರ್ ಮೌಂಟಿಂಗ್, ರೋಟರ್ ಬ್ಯಾಲೆನ್ಸ್ ಮತ್ತು ಕಂಪನವನ್ನು ಪರಿಶೀಲಿಸಿ.",

            "Hindi":
                "शाफ्ट अलाइनमेंट, मोटर माउंटिंग, रोटर बैलेंस और कंपन की जांच करें।"
        },

        "voice": {
            "English":
                "A possible motor imbalance is detected. The likely problem area is the motor shaft or rotating assembly. Check alignment and balance.",

            "Kannada":
                "ಮೋಟಾರ್‌ನಲ್ಲಿ ಸಂಭವನೀಯ ಅಸಮತೋಲನ ಪತ್ತೆಯಾಗಿದೆ. ಸಮಸ್ಯೆಯು ಮೋಟಾರ್ ಶಾಫ್ಟ್ ಅಥವಾ ತಿರುಗುವ ಭಾಗದಲ್ಲಿ ಇರಬಹುದು. ಅಲೈನ್‌ಮೆಂಟ್ ಮತ್ತು ಬ್ಯಾಲೆನ್ಸ್ ಪರಿಶೀಲಿಸಿ.",

            "Hindi":
                "मोटर में संभावित असंतुलन पाया गया है। समस्या मोटर शाफ्ट या घूमने वाले भाग में हो सकती है। अलाइनमेंट और बैलेंस की जांच करें।"
        }
    }
}


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(y, sr):

    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return np.zeros(29)

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(y)

    rms = librosa.feature.rms(y=y)

    return np.r_[
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        chroma.mean(axis=1)[:5],
        [
            spectral_centroid.mean(),
            spectral_bandwidth.mean(),
            spectral_rolloff.mean(),
            zcr.mean(),
            rms.mean()
        ]
    ]


# =========================================================
# CREATE DEMO DATASET
# =========================================================

def create_demo_dataset():

    rng = np.random.default_rng(42)

    X = []
    labels = []

    sr = 22050
    duration = 2

    t = np.linspace(
        0,
        duration,
        sr * duration,
        endpoint=False
    )

    machine_types = {

        "Normal": [120, 240],

        "Bearing Fault": [
            120,
            240,
            480,
            900
        ],

        "Loose Connection": [
            80,
            160
        ],

        "Motor Imbalance": [
            120,
            180,
            300
        ]
    }

    for label, frequencies in machine_types.items():

        for _ in range(30):

            audio = np.zeros_like(t)

            for frequency in frequencies:

                amplitude = (
                    0.15
                    if label == "Normal"
                    else 0.22
                )

                audio += (
                    amplitude *
                    np.sin(
                        2 * np.pi *
                        frequency *
                        t
                        +
                        rng.random() * 6.28
                    )
                )

            # Bearing fault
            if label == "Bearing Fault":

                positions = rng.integers(
                    0,
                    len(t),
                    20
                )

                for position in positions:

                    end = min(
                        len(t),
                        position + 100
                    )

                    audio[position:end] += (
                        0.5 *
                        np.exp(
                            -np.arange(
                                end - position
                            ) / 25
                        )
                    )

            # Loose connection
            elif label == "Loose Connection":

                mask = (
                    rng.random(len(t)) > 0.97
                ).astype(float)

                audio += (
                    0.25 *
                    mask *
                    np.sin(
                        2 * np.pi *
                        1000 *
                        t
                    )
                )

            # Motor imbalance
            elif label == "Motor Imbalance":

                audio += (
                    0.18 *
                    np.sin(
                        2 * np.pi *
                        55 *
                        t
                    )
                )

            audio += rng.normal(
                0,
                0.03,
                len(t)
            )

            X.append(
                extract_features(
                    audio,
                    sr
                )
            )

            labels.append(label)

    return np.array(X), np.array(labels)


# =========================================================
# TRAIN MODEL
# =========================================================

class NumpyNearestCentroid:

    def fit(self, X, y):

        self.classes_ = np.unique(y)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        normalized = (X - self.mean_) / self.scale_

        self.centroids_ = np.array(
            [normalized[y == label].mean(axis=0) for label in self.classes_]
        )

        return self

    def _probabilities(self, X):

        normalized = (X - self.mean_) / self.scale_
        distances = np.linalg.norm(
            normalized[:, None, :] - self.centroids_[None, :, :],
            axis=2
        )
        scores = np.exp(-distances)

        return scores / scores.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._probabilities(X)

    def predict(self, X):
        probabilities = self._probabilities(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

@st.cache_resource
def train_model():

    X, y = create_demo_dataset()

    if RandomForestClassifier is not None:
        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=180,
                        random_state=42
                    )
                )
            ]
        )
    else:
        model = NumpyNearestCentroid()

    model.fit(X, y)

    return model


# =========================================================
# ANALYZE AUDIO
# =========================================================

def analyze_audio(y, sr):

    model = train_model()

    features = extract_features(
        y,
        sr
    )

    features = features.reshape(
        1,
        -1
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    confidence = (
        max(probabilities) * 100
    )

    return (
        prediction,
        confidence,
        probabilities,
        model
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌐 Select Voice Language")

    selected_language = st.selectbox(
        "AI Voice Language",
        [
            "English",
            "Kannada",
            "Hindi"
        ]
    )

    language_code = LANGUAGES[
        selected_language
    ]

    st.divider()

    st.header("📌 Project Modules")

    st.write("🎙️ Machine Audio Input")
    st.write("🧹 Audio Preprocessing")
    st.write("🧠 Feature Extraction")
    st.write("🤖 Random Forest ML")
    st.write("🔧 Fault Detection")
    st.write("📍 Fault Location")
    st.write("⚠️ Severity Detection")
    st.write("📊 Waveform & Spectrogram")
    st.write("📈 Probability Graph")
    st.write("🛠️ Maintenance Recommendation")
    st.write("🗣️ Multilingual AI Voice")


# =========================================================
# AUDIO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "🎙️ Upload Machine Audio",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ]
)


# =========================================================
# PROCESS AUDIO
# =========================================================

if uploaded_file:

    try:

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        audio, sample_rate = librosa.load(
            temp_path,
            sr=None,
            mono=True
        )

        os.remove(temp_path)

        st.subheader("🎧 Uploaded Machine Sound")

        st.audio(
            uploaded_file
        )


        # =================================================
        # AI ANALYSIS
        # =================================================

        prediction, confidence, probabilities, model = (
            analyze_audio(
                audio,
                sample_rate
            )
        )

        fault_info = FAULTS[
            prediction
        ]


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🔧 Machine Condition",
                prediction
            )

        with col2:

            st.metric(
                "🎯 Confidence",
                f"{confidence:.1f}%"
            )

        with col3:

            st.metric(
                "⚠️ Severity",
                fault_info["severity"]
            )


        # =================================================
        # LOCATION
        # =================================================

        st.subheader(
            "📍 Possible Problem Location"
        )

        st.success(
            fault_info["location"][
                selected_language
            ]
        )


        # =================================================
        # MAINTENANCE
        # =================================================

        st.subheader(
            "🛠️ Maintenance Recommendation"
        )

        st.info(
            fault_info["recommendation"][
                selected_language
            ]
        )


        # =================================================
        # WAVEFORM
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "📈 Audio Waveform"
            )

            fig, ax = plt.subplots()

            librosa.display.waveshow(
                audio,
                sr=sample_rate,
                ax=ax
            )

            ax.set_title(
                "Machine Audio Waveform"
            )

            ax.set_xlabel(
                "Time (seconds)"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            st.pyplot(fig)

            plt.close(fig)


        # =================================================
        # SPECTROGRAM
        # =================================================

        with col2:

            st.subheader(
                "🌈 Spectrogram"
            )

            spectrogram = (
                librosa.amplitude_to_db(
                    np.abs(
                        librosa.stft(audio)
                    ),
                    ref=np.max
                )
            )

            fig, ax = plt.subplots()

            librosa.display.specshow(
                spectrogram,
                sr=sample_rate,
                x_axis="time",
                y_axis="hz",
                ax=ax
            )

            ax.set_title(
                "Machine Sound Spectrogram"
            )

            st.pyplot(fig)

            plt.close(fig)


        # =================================================
        # PROBABILITY GRAPH
        # =================================================

        st.divider()

        st.subheader(
            "📊 AI Prediction Probability"
        )

        probability_data = pd.DataFrame(
            {
                "Condition":
                    model.classes_,

                "Probability":
                    probabilities
            }
        )

        probability_data = (
            probability_data
            .set_index(
                "Condition"
            )
        )

        st.bar_chart(
            probability_data
        )


        # =================================================
        # MULTILINGUAL VOICE
        # =================================================

        st.divider()

        st.subheader(
            f"🗣️ AI Voice — {selected_language}"
        )

        voice_text = fault_info[
            "voice"
        ][
            selected_language
        ]

        st.write(
            voice_text
        )

        if st.button(
            "🔊 Generate AI Voice"
        ):

            try:

                tts = gTTS(
                    text=voice_text,
                    lang=language_code
                )

                voice_path = os.path.join(
                    tempfile.gettempdir(),
                    "machine_fault_voice.mp3"
                )

                tts.save(
                    voice_path
                )

                with open(
                    voice_path,
                    "rb"
                ) as audio_file:

                    st.audio(
                        audio_file.read(),
                        format="audio/mp3"
                    )

                st.success(
                    f"AI voice generated in {selected_language}."
                )

            except Exception as e:

                st.warning(
                    "Voice generation requires an internet connection."
                )

                st.write(
                    str(e)
                )


        # =================================================
        # FINAL SUMMARY
        # =================================================

        st.divider()

        st.subheader(
            "📋 Diagnostic Summary"
        )

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Machine Condition",
                    "Confidence",
                    "Severity",
                    "Problem Location",
                    "Recommendation",
                    "Voice Language"
                ],

                "Result": [
                    prediction,
                    f"{confidence:.1f}%",
                    fault_info["severity"],
                    fault_info["location"][
                        selected_language
                    ],
                    fault_info["recommendation"][
                        selected_language
                    ],
                    selected_language
                ]
            }
        )

        st.table(
            summary
        )


    except Exception as error:

        st.error(
            "❌ Error while processing audio"
        )

        st.code(
            str(error)
        )


# =========================================================
# BEFORE AUDIO UPLOAD
# =========================================================

else:

    st.warning(
        "⬆️ Please upload a machine audio file to start."
    )

    st.subheader(
        "🔄 How the System Works"
    )

    st.markdown(
        """
        **Machine Sound**

        ↓

        **Audio Preprocessing**

        ↓

        **Feature Extraction**

        MFCC + Spectral Features + RMS + ZCR

        ↓

        **Random Forest ML**

        ↓

        **Fault Detection**

        ↓

        **Fault Location + Severity**

        ↓

        **Maintenance Recommendation + Multilingual AI Voice**
        """
    )