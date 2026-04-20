import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrain import RetrainPipeline

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
