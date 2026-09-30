# Crop Suitability Classification

A end‑to‑end machine‑learning project that predicts the most suitable crop given soil nutrients and environmental conditions.

## Project Structure
```
crop‑suitability‑classification/
│
├── data/
│   └── Crop_recommendation.csv
│
├── notebooks/
│   └── crop_suitability_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── crop_model.pkl
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

## Quick Start
1. **Create a virtual environment** (optional but recommended)
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # PowerShell
   # or   .venv\Scripts\activate.bat   # cmd
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook** for exploration and model training:
   ```bash
   jupyter notebook notebooks/crop_suitability_analysis.ipynb
   ```
4. **Train the final model** (the notebook already does this, but you can also run the script directly):
   ```bash
   python src/train.py
   ```
5. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

The app will load the saved model (`models/crop_model.pkl`) and expose a simple UI where you can enter nitrogen, phosphorus, potassium, temperature, humidity, pH and rainfall to get a crop recommendation.

## License
This project is provided for educational purposes and is released under the MIT License.
