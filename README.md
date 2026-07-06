# Breast Cancer Prediction

A modular, production-style ML pipeline that predicts whether a tumor is **Malignant** or **Benign**, based on the Breast Cancer Wisconsin dataset. Built with a clean `src/` architecture, DVC-managed pipeline stages, and full logging — following the same structure as the Loan Approval Prediction project.

## 🚀 Overview

This project takes raw diagnostic data through a complete ML pipeline — ingestion, preprocessing, feature engineering, training, and evaluation — and produces a trained classification model along with performance metrics.

## 🧠 Model

- **Algorithm:** AdaBoost Classifier
- **Target variable:** `diagnosis` (`M` = Malignant → `1`, `B` = Benign → `0`)
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📂 Project Structure

```
Breast-Cancer-Prediction/
│
├── data/
│   ├── raw/              # Raw ingested data (train.csv, test.csv)
│   ├── processed/        # Cleaned & encoded data
│   └── interim/          # Scaled data (train_scaled.csv, test_scaled.csv)
│
├── src/
│   ├── data_ingestion.py       # Loads raw data, drops unwanted columns, splits
│   ├── data_preprocessing.py   # Removes duplicates/missing values, encodes target
│   ├── feature_engineering.py  # Scales features using StandardScaler
│   ├── model_training.py       # Trains the AdaBoost model, saves it
│   └── model_evaluation.py     # Evaluates the model, saves metrics
│
├── models/
│   ├── scaler.pkl         # Fitted StandardScaler (for future predictions)
│   └── model.pkl          # Trained model artifact
│
├── reports/
│   └── metrics.json       # Accuracy, precision, recall, f1, confusion matrix
│
├── logs/                  # Log files for each pipeline stage
├── dvc.yaml                # DVC pipeline stage definitions
├── requirements.txt
└── README.md
```

## ⚙️ Pipeline Stages (DVC)

| Stage                | Input                          | Output                                   |
|------------------------|----------------------------------|---------------------------------------------|
| `data_ingestion`        | Raw source data                  | `data/raw`                                   |
| `data_preprocessing`    | `data/raw`                       | `data/processed`                             |
| `feature_engineering`   | `data/processed`                 | `data/interim`, `models/scaler.pkl`          |
| `model_training`        | `data/interim`                   | `models/model.pkl`                           |
| `model_evaluation`      | `models/model.pkl`, `data/interim`| `reports/metrics.json`                       |

Run the entire pipeline with:
```bash
dvc repro
```

Force a full re-run (ignoring cache):
```bash
dvc repro --force
```

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** – preprocessing, scaling, model training & evaluation
- **pandas** – data handling
- **DVC** – pipeline versioning & reproducibility
- **joblib** – model/scaler serialization
- **logging** – structured debug/info/error logs (console + file)

## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd Breast-Cancer-Prediction
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline
   ```bash
   dvc repro
   ```

4. Check results
   ```
   reports/metrics.json
   logs/
   ```

## 📌 Future Improvements

- [ ] Move hyperparameters (`n_estimators`, `learning_rate`) to `params.yaml`
- [ ] Add a Flask-based web interface for interactive predictions
- [ ] Add model versioning via DVC remote storage
- [ ] Add unit tests for each pipeline stage

## 👤 Author

**Rahul**
Final-year CSE student, focused on Data Analytics & Machine Learning
GitHub: [Rahul117590](https://github.com/Rahul117590)