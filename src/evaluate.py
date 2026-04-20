import numpy as np
import seraplot as sp

class Evaluator(object):
    _METRICS: tuple = ()

    def __init__(self, model, x, y_true):
        self._model = model
        self._x = x
        self._y_true = np.array(y_true, dtype=np.int32)
        self._y_pred = np.array(model.predict(x), dtype=np.int32)

    @property
    def ground_truth(self) -> list[int]:
        return self._y_true.tolist()

    @property
    def predictions(self) -> list[int]:
        return self._y_pred.tolist()

    def compute(self) -> dict[str, float]:
        yt, yp = self.ground_truth, self.predictions
        return dict(map(lambda m: (m[0], m[1](yt, yp)), self._METRICS))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self._y_true)})"

class ClassificationEvaluator(Evaluator):
    _METRICS = (
        ("accuracy", lambda yt, yp: sp.accuracy_score(yt, yp)),
        ("f1", lambda yt, yp: sp.f1_score(yt, yp, average="binary")),
        ("precision", lambda yt, yp: sp.precision_score(yt, yp, average="binary")),
        ("recall", lambda yt, yp: sp.recall_score(yt, yp, average="binary")),
    )

    def full_report(self) -> dict[str, float | dict]:
        metrics = self.compute()
        yt, yp = self.ground_truth, self.predictions

        metrics["confusion_matrix"] = sp.confusion_matrix(yt, yp)
        metrics["classification_report"] = sp.classification_report(yt, yp)

        return metrics

    @staticmethod
    def cross_validate(estimator_name, x, y, cv=5, seed=42) -> dict[str, float | list[float]]:
        scores = sp.cross_val_score(estimator_name, x, y, cv=cv, scoring="accuracy", seed=seed)
        
        return {
            "cv_scores": scores,
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
        }
