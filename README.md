# Emotion Detection Suite

This repository contains two emotion detection projects:

1. **Facial Emotion Detection** using a pre-trained Mini-XCEPTION model.
2. **Speech-based Emotion Detection** using Whisper and audio feature analysis.

---

## 📸 Facial Emotion Detection (Mini-XCEPTION)

# Emotion Detection with Mini-XCEPTION

This project uses a pre-trained Mini-XCEPTION model to detect facial emotions from uploaded images. It leverages OpenCV for face detection and Keras for deep learning inference, running entirely within Google Colab.

## 🔧 Features

- Face detection using OpenCV's Haar cascades.
- Emotion classification using a pre-trained Mini-XCEPTION model trained on the FER-2013 dataset.
- Real-time inference on static images.
- Compatible with Google Colab.

## 🧠 Emotion Labels

The model can classify faces into the following emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 🚀 Getting Started

### 1. Clone this repository (optional)

```bash
git clone https://github.com/your-username/emotion-detection-mini-xception.git
cd emotion-detection-mini-xception
```

### 2. Open the notebook in Google Colab

Upload or copy the code into a Google Colab notebook.

### 3. Install dependencies

```python
!pip install -q keras opencv-python
```

### 4. Download the pre-trained model

```python
!wget -q https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 -O emotion_model.h5
```

### 5. Upload an image

Use the built-in file uploader:

```python
from google.colab import files
uploaded = files.upload()
```

### 6. Run the detection pipeline

- The uploaded image will be processed.
- Faces will be detected and labeled with the predicted emotion.

## 📷 Example Output

> Faces in your uploaded image will be outlined with a rectangle and the predicted emotion will be labeled above.

## 📁 Files

- `emotion_model.h5`: Pre-trained Mini-XCEPTION model weights.
- Main Colab script: Face detection and emotion classification logic.

## 📜 License

This project is licensed under the MIT License. Model originally from the [face_classification](https://github.com/oarriaga/face_classification) project by Octavio Arriaga.

---

## 🎤 Speech-Based Emotion Detection (Whisper + Audio Features)

# Emotion-Aware Speech Recognition using Whisper and Audio Features

This project performs speech transcription and basic emotion recognition from audio files using OpenAI's Whisper model and extracted audio features. It demonstrates how to combine speech recognition with emotional context analysis in a simple machine learning pipeline.

## 🔧 Features

- Transcribes speech from audio using Whisper.
- Extracts MFCC and pitch features from audio.
- Classifies emotional tone using a dummy SVM-based emotion classifier.
- Automatically handles audio format conversion and preprocessing.

## 🧠 Emotion Labels

The dummy classifier randomly trains on synthetic data with the following emotions:

- Happy
- Sad
- Angry
- Neutral

> Note: You can replace the dummy classifier with a real one using labeled emotional speech datasets like RAVDESS or IEMOCAP.

## 🚀 Getting Started

### 1. Install Dependencies

```python
!pip install -q openai-whisper librosa==0.10.0.post2 scikit-learn numpy==1.23.5 pydub ffmpeg-python
```

### 2. Load the Whisper Model

```python
import whisper
model = whisper.load_model("base")
```

### 3. Upload Audio File

Use Google Colab's uploader to upload audio files:

```python
from google.colab import files
uploaded = files.upload()
```

### 4. Run Full Pipeline

```python
emotion_aware_speech_recognition("/content/your_audio_file.wav")
```

## 📁 Files

- `emotion_aware_speech_recognition()`: Main function to transcribe and classify emotion.
- `convert_to_wav()`: Converts any input audio to mono WAV at 16kHz.
- `extract_audio_features()`: Extracts MFCC and pitch-based features.
- `train_emotion_classifier()`: Trains a basic SVM model on dummy data for emotion prediction.
- `classify_emotion()`: Predicts emotion from audio features.

## 📜 License

This project is for educational and experimental purposes. Whisper is by OpenAI, and audio processing libraries used are open-source.

---
