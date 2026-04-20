import time
import seraplot as sp
import mlflow

from src.config import (
    EXPERIMENT_NAME, MODEL_NAME, GRID_PARAMS,
    CV_FOLDS, RANDOM_SEED, MLRUNS_DIR
)

from src.data_prep import DataPipeline
from src.evaluate import ClassificationEvaluator
from src.model_store import ModelRegistry

class BasePipeline(object):
    _registry = ModelRegistry()

    def __init__(self, data_path=None):
        self._pipeline = DataPipeline(data_path).clean()

    def _prepare(self) -> tuple:
        ds = self._pipeline.split()
        scaled = DataPipeline.scale(ds.x_train, ds.x_val, ds.x_test)

        return ds, scaled

    @staticmethod
    def _build_model(params) -> sp.RandomForestClassifier:
        return sp.RandomForestClassifier(
            **dict(map(lambda kv: (kv[0], int(kv[1])), params.items()))
        )

    @staticmethod
    def _numeric_only(metrics) -> dict:
        return dict(filter(lambda kv: isinstance(kv[1], (int, float)), metrics.items()))

    def run(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pipeline={self._pipeline})"

class TrainingPipeline(BasePipeline):
    def run(self) -> dict:
        mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
        mlflow.set_experiment(EXPERIMENT_NAME)

        ds, scaled = self._prepare()
        scaler_params = {"mean": scaled.scaler_mean, "scale": scaled.scaler_scale}

        with mlflow.start_run(run_name="grid_search") as run:
            tuple(map(
                lambda kv: mlflow.log_param(*kv),
                (("estimator", MODEL_NAME), ("cv_folds", CV_FOLDS), ("seed", RANDOM_SEED)),
            ))

            grid = sp.GridSearchCV(
                MODEL_NAME, GRID_PARAMS,
                cv=CV_FOLDS, seed=RANDOM_SEED, scoring="accuracy",
            )
            grid.fit(scaled.x_train, ds.y_train)

            best_params = dict(grid.best_params_)
            mlflow.log_metric("best_cv_accuracy", grid.best_score_)
            tuple(map(
                lambda kv: mlflow.log_param(f"best_{kv[0]}", kv[1]),
                best_params.items(),
            ))

            model = self._build_model(best_params)
            model.fit(scaled.x_train, ds.y_train)

            evals = tuple(map(
                lambda t: (t[0], ClassificationEvaluator(model, t[1], t[2])),
                (
                    ("train", scaled.x_train, ds.y_train),
                    ("val", scaled.x_val, ds.y_val),
                    ("test", scaled.x_test, ds.y_test),
                ),
            ))

            metrics = {name: ev.compute() for name, ev in evals}
            test_report = evals[2][1].full_report()

            for prefix, m in [("train", metrics["train"]), ("val", metrics["val"]), ("test", test_report)]:
                tuple(map(
                    lambda kv: mlflow.log_metric(f"{prefix}_{kv[0]}", kv[1]),
                    self._numeric_only(m).items(),
                ))

            cv = ClassificationEvaluator.cross_validate(
                MODEL_NAME, scaled.x_train, ds.y_train, cv=CV_FOLDS, seed=RANDOM_SEED,
            )
            mlflow.log_metric("cv_mean_accuracy", cv["cv_mean"])
            mlflow.log_metric("cv_std_accuracy", cv["cv_std"])

            version = str(int(time.time()))
            all_metrics = {
                "best_cv_accuracy": grid.best_score_,
                "train": metrics["train"],
                "val": metrics["val"],
                "test": self._numeric_only(test_report),
                "cv": cv,
            }

            model_path = self._registry.save(
                MODEL_NAME, best_params, scaler_params, all_metrics, version,
            )
            mlflow.log_artifact(str(model_path))
            mlflow.log_param("model_version", version)

        return {
            "best_params": best_params,
            "best_cv_score": grid.best_score_,
            "train_metrics": metrics["train"],
            "val_metrics": metrics["val"],
            "test_metrics": test_report,
            "model_version": version,
            "run_id": run.info.run_id,
        }

if __name__ == "__main__":
    r = TrainingPipeline().run()
    
    tuple(map(print, (
        f"Version: {r['model_version']}",
        f"Best CV: {r['best_cv_score']:.4f}",
        f"Train acc: {r['train_metrics']['accuracy']:.4f}",
        f"Val acc: {r['val_metrics']['accuracy']:.4f}",
        f"Test acc: {r['test_metrics']['accuracy']:.4f}",
    )))
