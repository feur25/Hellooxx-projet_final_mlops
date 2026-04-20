import os
import glob
import time
import pandas as pd
from pathlib import Path

from src.config import NEW_DATA_DIR, DATA_DIR
from src.train import TrainingPipeline, BasePipeline

class RetrainPipeline(BasePipeline):
    _archive_dir = Path(NEW_DATA_DIR) / "processed"

    def _detect_new(self) -> tuple:
        Path(NEW_DATA_DIR).mkdir(parents=True, exist_ok=True)
        return tuple(glob.glob(str(Path(NEW_DATA_DIR) / "*.csv")))

    def _merge(self, new_files) -> bool:
        main_csv = Path(DATA_DIR) / "diabetes.csv"
        if not main_csv.exists():
            return False

        existing = pd.read_csv(main_csv)
        combined = pd.concat(
            [existing, *map(pd.read_csv, new_files)], ignore_index=True,
        ).drop_duplicates()

        existing.to_csv(
            Path(DATA_DIR) / f"diabetes_backup_{int(time.time())}.csv", index=False,
        )
        combined.to_csv(main_csv, index=False)

        self._archive_dir.mkdir(parents=True, exist_ok=True)
        tuple(map(
            lambda f: os.rename(f, self._archive_dir / Path(f).name),
            new_files,
        ))
        return True

    def run(self) -> dict:
        new_files = self._detect_new()
        if not new_files:
            return {"status": "no_new_data"}

        if not self._merge(new_files):
            return {"status": "merge_failed"}

        results = TrainingPipeline().run()

        try:
            prev = self._registry.load("latest")
            improved = results["val_metrics"]["accuracy"] > prev["metrics"]["val"]["accuracy"]
        except Exception:
            improved = True

        return {
            "status": "retrained",
            "new_files_count": len(new_files),
            "results": results,
            "improved": improved,
        }

if __name__ == "__main__":
    result = RetrainPipeline().run()

    print(f"Status: {result['status']}")

    if result["status"] == "retrained":
        r = result["results"]

        tuple(map(print, (
            f"Files: {result['new_files_count']}",
            f"Improved: {result['improved']}",
            f"Version: {r['model_version']}",
            f"Val acc: {r['val_metrics']['accuracy']:.4f}",
        )))