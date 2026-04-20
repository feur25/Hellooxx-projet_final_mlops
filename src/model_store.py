import json
from pathlib import Path
from src.config import MODELS_DIR

class ModelRegistry(object):
    def __init__(self, base_dir=None):
        self._dir = Path(base_dir or MODELS_DIR)

    def _path(self, version) -> Path:
        return self._dir / f"model_{version}.json"

    def save(self, name, params, scaler_params, metrics, version="latest") -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        config = dict(
            model_name=name, params=params, scaler=scaler_params,
            metrics=metrics, version=version,
        )
        tuple(map(
            lambda p: p.write_text(json.dumps(config, indent=2)),
            (self._path(version), self._path("latest")),
        ))
        return self._path(version)

    def load(self, version="latest") -> dict:
        return json.loads(self._path(version).read_text())

    @property
    def versions(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        
        return sorted(filter(
            lambda v: v != "latest",
            map(
                lambda f: f.stem.removeprefix("model_"),
                filter(
                    lambda f: f.name.startswith("model_") and f.suffix == ".json",
                    self._dir.iterdir(),
                ),
            ),
        ))

    def __repr__(self) -> str:
        return f"ModelRegistry(dir={self._dir}, n_versions={len(self.versions)})"