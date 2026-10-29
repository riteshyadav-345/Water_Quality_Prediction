# AI-Based Water Quality Prediction

This project provides a Streamlit data explorer and a baseline Random Forest classifier using the Water Potability dataset.

## Important dataset finding

`data/water_potability.csv` contains the nine expected features and a `Potability` target (0/1). The earlier `data/water_dataX.csv` is a different river monitoring dataset and is not used for this classifier. The Kaggle dataset has missing measurements; the training pipeline imputes them with training-set medians.

The training script uses a stratified 80/20 train/test split, a median-imputation and Random Forest pipeline, and reports accuracy, precision, recall, F1-score and a confusion matrix. It saves the fitted pipeline to `artifacts/water_potability_model.joblib`.

## Run in VS Code on Windows

1. Open the `Water_Quality_Ml_Project` folder in VS Code.
2. Choose **Terminal → New Terminal**.
3. Create and activate a virtual environment in PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the project packages and train/save the model:

   ```powershell
   pip install -r requirements.txt
   python train_model.py
   ```

5. Start the interface:

   ```powershell
   streamlit run app.py
   ```

The browser opens a form with nine measurement fields. Enter lab readings and click **Predict water class**. The default values are example medians from the training dataset. If PowerShell blocks environment activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.venv\Scripts\python.exe -m streamlit run app.py`.

## Scope and limitations

The current app explores data and missing values; it does not issue drinking-water safety decisions. A dataset classifier is only a preliminary screening aid and cannot replace appropriate laboratory testing, including microbiological and chemical assessment against applicable standards.
