import numpy as np
from pydantic import BaseModel
from typing import Optional


class BaseRequest(BaseModel):
    def to_dict(self) -> dict:
        return self.model_dump()


class PredictionRequest(BaseRequest):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

    def to_array(self) -> np.ndarray:
        return np.array([tuple(self.model_dump().values())], dtype=np.float64)

    @classmethod
    def from_tuple(cls, values: tuple):
        keys = tuple(cls.model_fields.keys())
        return cls(**dict(zip(keys, values)))

    class Config:
        arbitrary_types_allowed = True


class BatchPredictionRequest(BaseRequest):
    samples: list[PredictionRequest]

    def to_array(self) -> np.ndarray:
        return np.vstack(list(map(lambda s: s.to_array(), self.samples)))


class BaseResponse(BaseModel):
    pass


class PredictionResponse(BaseResponse):
    prediction: int
    probability: Optional[list[float]] = None

    @classmethod
    def from_prediction(cls, pred, proba):
        return cls(prediction=int(pred), probability=list(map(float, proba)))


class BatchPredictionResponse(BaseResponse):
    predictions: list[PredictionResponse]


class HealthResponse(BaseResponse):
    status: str
    model_version: str
    model_name: str


class RetrainResponse(BaseResponse):
    status: str
    new_files_count: Optional[int] = None
    improved: Optional[bool] = None
    model_version: Optional[str] = None
