import pandas as pd
import numpy as np
import seraplot as sp
from typing import NamedTuple

from src.config import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    FEATURE_COLS, TARGET_COL, SPLIT_RATIO, RANDOM_SEED
)

class DataSplit(NamedTuple):
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

class ScaledData(NamedTuple):
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    scaler_mean: list
    scaler_scale: list

class DataPipeline(object):
    _IQR_COLS = ("Insulin", "DiabetesPedigreeFunction")
    _IQR_FACTOR = 1.5

    def __init__(self, path=None):
        self._df = pd.read_csv(path or RAW_DATA_PATH)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def clean(self) -> "DataPipeline":
        self._df = self._df.copy()
        tuple(map(self._cap_outliers, self._IQR_COLS))

        return self

    def _cap_outliers(self, col):
        q1, q3 = self._df[col].quantile(0.25), self._df[col].quantile(0.75)
        bounds = (q1 - self._IQR_FACTOR * (q3 - q1), q3 + self._IQR_FACTOR * (q3 - q1))
        self._df[col] = self._df[col].clip(*bounds)

    def _extract_xy(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(self._df[list(FEATURE_COLS)].values, dtype=np.float64)
        y = self._df[TARGET_COL].values.astype(np.int32)

        return x, y

    @staticmethod
    def _as_f64(arr) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=np.float64)

    @staticmethod
    def _as_i32(arr) -> np.ndarray:
        return np.array(arr, dtype=np.int32)

    def split(self) -> DataSplit:
        x, y = self._extract_xy()
        x_tv, x_te, y_tv, y_te = sp.train_test_split(
            x, y, test_size=SPLIT_RATIO.test, random_state=RANDOM_SEED
        )
        val_frac = SPLIT_RATIO.val / (SPLIT_RATIO.train + SPLIT_RATIO.val)
        x_tr, x_va, y_tr, y_va = sp.train_test_split(
            self._as_f64(x_tv), self._as_i32(y_tv),
            test_size=val_frac, random_state=RANDOM_SEED
        )
        return DataSplit(
            *map(self._as_f64, (x_tr, x_va, x_te)),
            *map(self._as_i32, (y_tr, y_va, y_te)),
        )

    def save_splits(self) -> DataSplit:
        ds = self.clean().split()
        cols = list(FEATURE_COLS) + [TARGET_COL]

        tuple(map(
            lambda t: pd.DataFrame(
                np.column_stack([t[1], np.array(t[2]).reshape(-1, 1)]), columns=cols
            ).to_csv(t[0], index=False),
            zip(
                (TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH),
                (ds.x_train, ds.x_val, ds.x_test),
                (ds.y_train, ds.y_val, ds.y_test),
            )
        ))

        return ds

    @classmethod
    def from_csv(cls, path) -> "DataPipeline":
        return cls(path)

    @staticmethod
    def load_split(path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        x = np.ascontiguousarray(df[list(FEATURE_COLS)].values, dtype=np.float64)

        y = df[TARGET_COL].values.astype(np.int32)

        return x, y

    @staticmethod
    def scale(x_train, x_val=None, x_test=None) -> ScaledData:
        scaler = sp.StandardScaler()
        xt = scaler.fit_transform(x_train)

        return ScaledData(
            x_train=xt,
            x_val=scaler.transform(x_val) if x_val is not None else None,
            x_test=scaler.transform(x_test) if x_test is not None else None,
            scaler_mean=list(scaler.mean_),
            scaler_scale=list(scaler.scale_),
        )

    def __repr__(self) -> str:
        return f"DataPipeline(shape={self.shape})"
