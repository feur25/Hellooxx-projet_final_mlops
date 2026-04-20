from pathlib import Path
from typing import NamedTuple

class SplitRatio(NamedTuple):
    train: float
    val: float
    test: float

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"

RAW_DATA_PATH = DATA_DIR / "diabetes.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
VAL_DATA_PATH = DATA_DIR / "val.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
NEW_DATA_DIR = DATA_DIR / "incoming"

FEATURE_COLS = (
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
)

TARGET_COL = "Outcome"

SPLIT_RATIO = SplitRatio(0.70, 0.15, 0.15)

RANDOM_SEED = 42
CV_FOLDS = 5

EXPERIMENT_NAME = "diabetes_prediction"
MODEL_NAME = "RandomForestClassifier"

GRID_PARAMS = {
    "n_estimators": [10, 20, 50, 100],
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

API_HOST = "0.0.0.0"
API_PORT = 8000