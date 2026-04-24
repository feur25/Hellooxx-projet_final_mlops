import numpy as np
from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    pregnancies: float = Field(..., ge=0, le=20, description="Number of times pregnant", examples=[6])
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration after a 2-hour OGTT", examples=[148])
    blood_pressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)", examples=[72])
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)", examples=[35])
    insulin: float = Field(..., ge=0, le=900, description="2-hour serum insulin (mu U/ml)", examples=[0])
    bmi: float = Field(..., ge=0, le=80, description="Body mass index (kg/m^2)", examples=[33.6])
    diabetes_pedigree_function: float = Field(..., ge=0, le=3, description="Diabetes pedigree function", examples=[0.627])
    age: float = Field(..., ge=0, le=120, description="Age in years", examples=[50])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50,
            }]
        }
    }

    def to_array(self) -> np.ndarray:
        return np.array([tuple(self.model_dump().values())], dtype=np.float64)

    @classmethod
    def from_tuple(cls, values: tuple) -> "PredictionRequest":
        keys = tuple(cls.model_fields.keys())
        return cls(**dict(zip(keys, values)))


class BatchPredictionRequest(BaseModel):
    samples: list[PredictionRequest] = Field(
        ..., min_length=1, max_length=10000,
        description="List of patient feature payloads (1 to 10000 entries).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "samples": [
                    {
                        "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
                        "skin_thickness": 35, "insulin": 0, "bmi": 33.6,
                        "diabetes_pedigree_function": 0.627, "age": 50,
                    },
                    {
                        "pregnancies": 1, "glucose": 85, "blood_pressure": 66,
                        "skin_thickness": 29, "insulin": 0, "bmi": 26.6,
                        "diabetes_pedigree_function": 0.351, "age": 31,
                    },
                ]
            }]
        }
    }

    def to_array(self) -> np.ndarray:
        return np.vstack([s.to_array() for s in self.samples])


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class: 0 (non-diabetic) or 1 (diabetic)", examples=[1])
    probability: Optional[list[float]] = Field(
        None, description="Class probabilities [P(class=0), P(class=1)]",
        examples=[[0.21, 0.79]],
    )

    @classmethod
    def from_prediction(cls, pred, proba) -> "PredictionResponse":
        return cls(prediction=int(pred), probability=list(map(float, proba)))


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(..., description="Ordered list of predictions, one per input sample.")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_version: str = Field(..., description="Version stamp of the loaded model.", examples=["1776940497"])
    model_name: str = Field(..., description="Estimator class name.", examples=["RandomForestClassifier"])


class RootResponse(BaseModel):
    name: str = Field(..., examples=["Diabetes Prediction API"])
    version: str = Field(..., examples=["1.0.0"])
    docs: str = Field(..., examples=["/docs"])
    redoc: str = Field(..., examples=["/redoc"])
    openapi: str = Field(..., examples=["/openapi.json"])
    endpoints: list[str]


class RetrainResponse(BaseModel):
    status: str = Field(..., description="One of: `retrained`, `no_new_data`, `merge_failed`.", examples=["retrained"])
    new_files_count: Optional[int] = Field(None, description="Number of new CSV files merged.", examples=[1])
    improved: Optional[bool] = Field(None, description="True if the new model beats the previous one on validation.", examples=[True])
    model_version: Optional[str] = Field(None, description="Version stamp of the freshly trained model.", examples=["1776940497"])


class ModelInfoResponse(BaseModel):
    name: str = Field(..., examples=["RandomForestClassifier"])
    version: str = Field(..., examples=["1776940497"])
    params: dict = Field(..., description="Hyperparameters chosen by GridSearchCV.")
    metrics: dict = Field(..., description="Train / val / test / CV metrics persisted alongside the model.")


class ModelVersionsResponse(BaseModel):
    versions: list[str] = Field(..., description="All model versions persisted in the registry.")
    current: Optional[str] = Field(None, description="Version currently loaded in memory.")


class ModelReloadResponse(BaseModel):
    status: str = Field(..., examples=["reloaded"])
    version: str = Field(..., examples=["1776940497"])
    name: str = Field(..., examples=["RandomForestClassifier"])
