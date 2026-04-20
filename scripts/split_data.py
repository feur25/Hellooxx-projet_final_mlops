import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import DataPipeline

ds = DataPipeline().save_splits()
sizes = tuple(map(len, (ds.y_train, ds.y_val, ds.y_test)))

print(f"Train: {sizes[0]} | Val: {sizes[1]} | Test: {sizes[2]}")
print(f"Total: {sum(sizes)}")
