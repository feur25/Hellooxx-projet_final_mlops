import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import TrainingPipeline

r = TrainingPipeline().run()

tuple(map(print, (
    f"\nModel version: {r['model_version']}",
    f"MLflow run ID: {r['run_id']}",
    f"Best CV accuracy: {r['best_cv_score']:.4f}",
    f"Best params: {r['best_params']}",
    f"Train accuracy: {r['train_metrics']['accuracy']:.4f}",
    f"Val accuracy: {r['val_metrics']['accuracy']:.4f}",
    f"Test accuracy: {r['test_metrics']['accuracy']:.4f}",
    f"Test F1: {r['test_metrics']['f1']:.4f}",
    f"Test precision: {r['test_metrics']['precision']:.4f}",
    f"Test recall: {r['test_metrics']['recall']:.4f}",
)))
