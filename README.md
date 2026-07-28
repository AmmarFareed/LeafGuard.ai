# 🌿 LeafGuard AI

**Early Detection of Crop Diseases from Leaf Images**

An AI-powered computer vision tool that diagnoses plant leaf diseases from a single photo — giving farmers a fast, free, first-line diagnosis without waiting for expert access.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()

---

## 📖 Overview

LeafGuard AI is a final certificate project for an Artificial Intelligence course. It uses **transfer learning** on **MobileNetV2** to classify plant leaf images into 38 categories spanning 14 crop species — identifying whether a leaf is healthy or diagnosing a specific disease.

Upload a photo of a leaf, and the app returns:
- A predicted disease (or healthy) label
- A confidence score
- A ranked top-3 breakdown for transparency

## 🎯 Problem It Solves

Farmers — especially smallholders — often can't identify a crop disease early enough to act, and agricultural experts aren't always available on-site. LeafGuard AI offers an instant, accessible first-line diagnosis, helping catch disease before it spreads and reduces yield.

## 🧠 How It Works

1. **Capture** — user uploads a leaf photo
2. **Analyze** — a fine-tuned MobileNetV2 CNN classifies the image
3. **Diagnose** — the app returns a label + confidence score

## 📊 Dataset

- **Source:** [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle, built on the original PlantVillage corpus
- **Classes:** 38 (healthy + disease labels across 14 crop species)
- **Size:** ~87,000 labeled leaf images

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Modeling | TensorFlow / Keras (MobileNetV2 transfer learning) |
| Evaluation | scikit-learn, Matplotlib, Seaborn |
| Demo App | Streamlit |
| Image Handling | Pillow |
| Training Environment | Google Colab (GPU) |
| Data Source | Kaggle API |

## 📁 Project Structure

```
LeafGuard-AI/
├── LeafGuard_AI.ipynb       # Full training notebook (data → model → evaluation)
├── streamlit_app.py         # Demo web app
├── requirements.txt         # Python dependencies
├── leafguard_model.keras    # Trained model (generated after training)
├── class_names.json         # Class label mapping (generated after training)
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/LeafGuard-AI.git
cd LeafGuard-AI
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run streamlit_app.py
```
The app opens at `http://localhost:8501`.

> **Note:** `leafguard_model.keras` and `class_names.json` must be trained first — see the notebook, or download pre-trained files if provided in [Releases](../../releases).

## 🧪 Model Training

The full pipeline — data loading, augmentation, transfer learning, fine-tuning, and evaluation — lives in `LeafGuard_AI.ipynb`, designed to run on Google Colab with a free GPU runtime.

**Approach:**
- Base model: MobileNetV2 (pretrained on ImageNet), frozen initially
- Classification head: Global Average Pooling → Dropout (0.3) → Dense (38, softmax)
- Fine-tuning: last 20 layers unfrozen at a lower learning rate
- Regularization: dropout, early stopping, data augmentation (flip, rotation, zoom, brightness)

## ⚠️ Known Limitations

- Trained on lab-quality images (single leaf, plain background) — accuracy may drop on cluttered, real-world field photos
- Class imbalance across the 38 categories may affect per-class performance
- Not a replacement for expert diagnosis — intended as a fast, first-line triage tool

## 🔭 Future Improvements

- [ ] Grad-CAM visual explainability
- [ ] Fine-tune on field-condition datasets (e.g. PlantDoc)
- [ ] Compare against EfficientNet / ResNet architectures
- [ ] Add treatment/care recommendations per diagnosis
- [ ] Public deployment (Streamlit Community Cloud / Hugging Face Spaces)

## 👤 Author

**Ammar Fareed**
Higher Diploma in IT, SLIIT City University
[Portfolio](https://ammarfareed.vercel.app) · [LinkedIn](https://linkedin.com/in/ammar-fareed) · [GitHub](https://github.com/AmmarFareed)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
