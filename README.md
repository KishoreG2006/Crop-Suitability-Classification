# 🌱 Crop‑Suitability‑Classification  

[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Streamlit Deploy](https://static.streamlit.io/badges/streamlit_badge_black.svg)](https://share.streamlit.io/kishoreg2006/crop-suitability-classification/main/app.py)

> **Predict the most suitable crop** for a given set of soil nutrients and environmental conditions – all with a lightweight, interpretable machine‑learning model that runs locally for free.

---

## 📖 Table of Contents
| # | Section |
|---|---------|
| 1 | [Project Overview](#-project-overview) |
| 2 | [Dataset](#-dataset) |
| 3 | [Methodology & Models](#-methodology--models) |
| 4 | [Results & Evaluation](#-results--evaluation) |
| 5 | [Installation](#-installation) |
| 6 | [Usage](#-usage) |
| 7 | [Folder Structure](#-folder-structure) |
| 8 | [Deploy the App](#-deploy-the-app) |
| 9 | [Advantages & Applications](#-advantages--applications) |
|10| [Conclusion](#-conclusion) |
|11| [Contributing](#-contributing) |
|12| [License](#-license) |

---

## 🌍 Project Overview
- **Domain:** Agriculture  
- **Task:** Multi‑class classification (≈ 30 crop categories)  
- **Goal:** Given **N, P, K, temperature, humidity, pH, rainfall** → recommend the crop that will yield the highest productivity on those conditions.  
- **Why?** Farmers often have soil‑test results but lack a quick, data‑driven way to decide which crop to grow. This project turns a small, publicly‑available dataset into a ready‑to‑use decision‑support tool.

---

## 📂 Dataset
| File | Description |
|------|------------|
| `data/Crop_recommendation.csv` | 1 500 + rows, 8 columns: 7 numeric features (`N, P, K, temperature, humidity, ph, rainfall`) and a categorical target (`label`). |
| **Source** | Kaggle – *Crop Recommendation Dataset* (≈ 150 KB). |

The dataset already contains a balanced mixture of **rice, maize, chickpea, kidney beans, moth beans, etc.** – perfect for a beginner‑level ML project.

---

## 🛠️ Methodology & Models
1. **Data inspection** – check for missing values, duplicates, and class distribution.  
2. **Train/‑test split** – 80 % train / 20 % test, stratified by `label`.  
3. **Pre‑processing** – `StandardScaler` (important for distance‑based models).  
4. **Models evaluated**  
   - **Logistic Regression** – baseline linear classifier.  
   - **K‑Nearest Neighbours (k = 5)** – non‑parametric, strong on small datasets.  
   - **Decision Tree** – rule‑based, fully interpretable.  

All three are wrapped in a **scikit‑learn Pipeline** (`scaler → classifier`) to avoid data leakage.

### Why K‑Nearest Neighbours?
- Highest test accuracy (**≈ 97.95 %**).  
- Simple to understand: predicts the majority class among the *k* most similar historic observations.  
- No heavy hyper‑parameter tuning required for a small dataset.

---

## 📊 Results & Evaluation

| Model                | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|----------------------|----------|----------------------|-------------------|---------------|
| Logistic Regression  | 0.9727   | 0.9740               | 0.9727            | 0.9725        |
| **K‑Nearest Neighbours** | **0.9795** | **0.9804**           | **0.9795**        | **0.9793**    |
| Decision Tree        | 0.9795   | 0.9806               | 0.9795            | 0.9794        |

*The KNN model was saved as `models/crop_model.pkl` and is used by the Streamlit app.*

---

## 💻 Installation

```powershell
# 1️⃣ Clone the repo (you already have it)
git clone https://github.com/KishoreG2006/Crop-Suitability-Classification.git
cd Crop-Suitability-Classification

# 2️⃣ Create a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or: source .venv/bin/activate  # macOS/Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt
