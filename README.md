# AI-Based Machine Fault Detection and Predictive Maintenance Using Audio Signal Analysis

## Project idea
The system listens to a machine sound and uses machine-learning features from the audio signal to classify the condition as:
- Normal
- Bearing Fault
- Motor Fault
- Fan Fault

The Streamlit interface shows the prediction, confidence, severity and a maintenance recommendation.

## Technology
- Python
- NumPy
- SciPy
- Scikit-learn
- Random Forest
- Streamlit

## How to run on Windows

### 1. Install Python
Use Python 3.10–3.12. Do not install "Python 12.8"; Python versions are 3.x.

### 2. Open Command Prompt in this project folder
Example:
`cd path\to\machine_fault_detection_ai_ml`

### 3. Create virtual environment
`python -m venv venv`

### 4. Activate it
`venv\Scripts\activate`

### 5. Install packages
`pip install -r requirements.txt`

### 6. Generate demo audio dataset
`python generate_dataset.py`

### 7. Train the model
`python train_model.py`

### 8. Start the web application
`streamlit run app.py`

The browser should open the Streamlit application automatically.

## Important
The included WAV files are synthetic signals created only to make the academic prototype runnable without an external dataset. For a final project, train the model with real labeled machine audio (normal and fault recordings).

## Project flow
Machine sound -> WAV input -> preprocessing -> feature extraction -> Random Forest -> fault class -> confidence/severity -> maintenance recommendation.

## Suggested presentation points
1. Problem: machine failures cause downtime and maintenance cost.
2. Input: machine sound from microphone or recorded WAV.
3. Processing: clean/normalize audio.
4. Features: RMS, Zero Crossing Rate, spectral centroid, bandwidth, rolloff and dominant frequency.
5. ML: Random Forest learns patterns from labeled audio.
6. Output: predicted condition, confidence, severity and recommendation.
7. Benefit: low-cost, non-contact monitoring prototype.
