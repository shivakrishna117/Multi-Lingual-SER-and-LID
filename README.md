# Multi-Lingual-SER-and-LID
"I built a multilingual speech emotion recognition system from scratch — no pre-trained models — that automatically detects whether someone is speaking Telugu, Hindi, German, or English, then predicts their emotion with 75% accuracy using a novel Multi-Head Attention + CRNN + BiGRU ensemble. 
# 🎙️ Multilingual SER + LID (Speech Emotion Recognition & Language Identification)

<p align="center">
  <img src="assets/banner.png" alt="Project Banner" width="800"/>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Research](https://img.shields.io/badge/Focus-Speech%20AI-purple)

</p>

---

## 🚀 Overview

An end-to-end **multilingual speech understanding system** that performs:

* 🌍 **Language Identification (LID)** — Telugu, Hindi, German, English
* 🎭 **Speech Emotion Recognition (SER)** — Angry, Happy, Sad, Neutral

Built **from scratch (no pre-trained models)** using a **CRNN + Multi-Head Attention + BiGRU ensemble**, and deployed as a **real-time app**.

---

## 🎯 Key Results

| Task                    | Accuracy  |
| ----------------------- | --------- |
| Language Identification | **99.4%** |
| Emotion Recognition     | **75%**   |

---

## 🧠 Architecture

<p align="center">
  <img src="assets/architecture.png" width="700"/>
</p>

```text
Audio → Feature Extraction → CNN → Multi-Head Attention → BiGRU → Output
```

---

## 🎥 Demo

<p align="center">
  <img src="assets/demo.gif" width="700"/>
</p>

👉 Real-time prediction from microphone input via Streamlit

---

## 🔬 Feature Engineering

* MFCC (40)
* Delta + Delta² (80)
* Log Mel Spectrogram (128)
* Pitch (F0)
* RMS Energy

👉 Final feature vector: **250 dimensions per frame**

---

## 🏗️ Model Highlights

* 🔹 CRNN (CNN + BiGRU)
* 🔹 Multi-Head Attention (8 heads × 2 layers)
* 🔹 Ensemble (3 seeds for stability)
* 🔹 No pre-trained models (fully trained from scratch)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ser-lid.git
cd ser-lid
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run App

```bash
streamlit run app.py
```

### Train Model

```bash
python train.py
```

---

## 📊 Results Visualization

<p align="center">
  <img src="assets/confusion_matrix.png" width="400"/>
  <img src="assets/loss_curve.png" width="400"/>
</p>

---

## 🧪 Tech Stack

* Python
* PyTorch
* NumPy / Librosa
* Streamlit

---

## 📌 Future Work

* Add low-resource languages
* Transformer-based architectures
* Edge deployment (mobile / IoT)

---

## 🤝 Contributing

PRs are welcome!

---

## 📜 License

MIT License

---

## 👤 Author

**Shiva Krishna**
Speech Processing | NLP | Deep Learning

---
