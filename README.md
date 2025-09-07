# Olist Delay Forecast Project

Predicting whether an order will arrive late is a common challenge in e‑commerce.
This repository contains a complete pipeline for predicting late deliveries in the Olist dataset, including data preprocessing, feature engineering, model training, hyper‑parameter tuning and evaluation. It provides an end‑to‑end example of building a machine‑learning solution using Python and open‑source libraries.

## Dataset
The project uses the open Olist public dataset from Kaggle, which contains order, customer, seller and geolocation information for a Brazilian e‑commerce platform. The key tables used include:
- `orders`: order timestamps (purchase, approval, shipping limit and delivery), order status and estimated delivery date.
- `order_items`: order items and sellers.
- `customers`: customer city and state.
- `sellers`: seller city and state.
- `products`: product characteristics (category, weight, size).
- `order_payments`: payment type, instalments and value.

The target is `is_late`, a binary flag indicating whether the actual delivery date exceeded the estimated delivery date. The script creates this label by comparing `order_delivered_customer_date` with `order_estimated_delivery_date`.

## Project Structure

```
├── data/                       # Raw and processed datasets (not tracked by git)
├── notebooks/                  # Jupyter notebooks for EDA and prototyping
├── src/
│   ├── preprocessing.py        # Feature engineering and preprocessing utilities
│   ├── models.py               # Model definitions (XGBoost, LightGBM, CatBoost, baseline)
│   ├── tuning.py               # Hyper‑parameter tuning functions (GridSearch, RandomizedSearch, Optuna)
│   ├── evaluation.py           # Evaluation metrics and plots
│   └── utils.py                # Utility functions (data loading, splitting, logging)
├── artifacts/                  # Saved models, thresholds, plots
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and usage instructions
```

## Setup

1. Clone the repository

```bash
git clone https://github.com/olavodd42/delay_forecast_olist.git
cd delay_forecast_olist
```

2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Requirements include pandas, numpy, scikit‑learn, xgboost, lightgbm, catboost, optuna and matplotlib.

4. Download the dataset
- The raw data is not stored in the repository. Download the Olist public dataset from Kaggle and place the `.csv` files in the `data/raw/` directory.
- Run the provided notebook or script in `src/utils.py` to load and merge the tables into a single DataFrame.

## Usage
The pipeline can be run either via notebook or from the command line. The main steps are:

1. Data loading and merging
```PYTHON
from src.utils import load_and_merge_data

df = load_and_merge_data(path_to_data="./data/raw/")
```

2. Feature engineering
```PYTHON
from src.preprocessing import engineer_features

df = engineer_features(df)
# This step creates aggregate features such as number of items, price and freight totals,
# average item price, shipping deadlines and flags for same‑state/region deliveries.
```

3. Train/test split
```PYTHON   
from sklearn.model_selection import train_test_split

X = df.drop(columns=["is_late", "order_id", "customer_id"])
y = df["is_late"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
```


4. Model training
Baseline models are defined in `src/models.py`. Example using XGBoost:
```PYTHON
from src.models import build_xgb_pipeline

pipe = build_xgb_pipeline(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]))
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:,1]
```

5. Hyper‑parameter tuning
`src/tuning.py` contains GridSearchCV, RandomizedSearchCV and Optuna optimizers for each model. Example using Optuna for LightGBM:
```PYTHON
from src.tuning import tune_lightgbm_optuna

best_pipe, study = tune_lightgbm_optuna(X_train, y_train,
    preprocessing=preprocessor,
    n_trials=30,
    cv_folds=3,
    scoring='average_precision')
y_prob = best_pipe.predict_proba(X_test)[:,1]
```

6. Evaluation

Evaluate PR‑AUC, ROC‑AUC and F1‑score at the optimal threshold using `src/evaluation.py`:
```PYTHON
from src.evaluation import evaluate_model, best_threshold_by_f1

evaluate_model(y_test, y_prob)
thr, precision, recall, f1 = best_threshold_by_f1(y_test, y_prob)
```

7. Saving artifacts
Save the trained pipeline and threshold for later inference:
```PYTHON
import joblib, json, os

os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_pipe, "artifacts/lightgbm_pipeline.joblib")
with open("artifacts/thresholds.json", "w") as f:
    json.dump({"threshold_f1": thr}, f)
```

8. Prediction on new data
```PYTHON
from src.models import load_pipeline

pipe = load_pipeline("artifacts/lightgbm_pipeline.joblib")
proba = pipe.predict_proba(new_df)[:,1]
predictions = (proba >= thr).astype(int)
```

## Improvements and Recommendations
The repository already includes pipelines for XGBoost, LightGBM and CatBoost and explores hyper‑parameter search strategies (grid search, random search, Optuna). To further improve the project:

- **Consolidate Feature Engineering** – Move all feature creation into a single module (`preprocessing.py`) with clearly documented functions. This ensures reproducibility and makes it easier to add or modify features.
- **Type Hints and Docstrings** – Add type hints and docstrings to all functions and classes to improve readability and maintenance.
- **Configuration Files** – Use a YAML or JSON configuration file for hyper‑parameters and dataset paths. This decouples code from configuration and simplifies experimentation.
- **GPU Support** – If a CUDA‑compatible GPU is available, enable GPU training in LightGBM (`device='cuda'`) and XGBoost to reduce training time. Ensure the environment is configured with the correct CUDA toolkit.
- **Cross‑Validation Strategy** – For time‑series data, implement a temporal split or TimeSeriesSplit to avoid leakage when evaluating models. The project includes a `temporal_split` function for this purpose.
- **Logging and Reporting** – Integrate logging (e.g., with Python’s logging module) to track training progress and parameter settings. Generate summary reports and plots for each model.
- **Packaging** – Convert the code into an installable Python package with `setup.py` or `pyproject.toml` so users can install it via pip and run the pipeline with a CLI entry point.
- **Unit Tests** – Add unit tests for key functions (feature engineering, model training, evaluation) using `pytest` to ensure reliability.
- **Docker Support** – Provide a `Dockerfile` with the required environment and GPU drivers to make deployment and reproducibility easier.
- **Documentation** – Expand this README with links to notebooks that demonstrate exploratory data analysis (EDA), feature importance visualisations, and comparison of model performance.

## License
VER [LICENSE.md](LICENSE.md))
## Acknowledgements
- Olist for providing the public dataset used in this project.
- Contributors who helped develop this pipeline and fine‑tune the models.