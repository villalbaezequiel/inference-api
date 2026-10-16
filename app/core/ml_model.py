"""Modelo ML embebido: carga única y versionado.

- Carga: en el lifespan, UNA vez por proceso, vía `asyncio.to_thread` para no
  bloquear el event loop mientras se deserializa. Si falta un artefacto, la app
  no arranca (fail-fast: mejor un pod que no pasa readiness que uno que da 500).
- Inferencia: CPU-bound -> corre en un ThreadPoolExecutor dedicado (no en el
  default de anyio que comparten las rutas sync), así nunca bloquea el loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

Prediction = tuple[str, float]


class UnknownModelVersion(Exception):
    def __init__(self, requested: str, available: list[str]):
        super().__init__(f"model version '{requested}' not available; use one of {available}")


@dataclass(slots=True)
class LoadedModel:
    version: str
    pipeline: object
    classes: list[str]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        proba = self.pipeline.predict_proba(texts)
        idx = np.argmax(proba, axis=1)
        return [(self.classes[i], float(proba[row, i])) for row, i in enumerate(idx)]


class ModelRegistry:
    def __init__(self, models_dir: str, versions: list[str], default_version: str, threads: int = 2):
        self.models_dir = Path(models_dir)
        self.versions = versions
        self.default_version = default_version
        self._executor = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ml")
        self._models: dict[str, LoadedModel] = {}

    @property
    def loaded_versions(self) -> list[str]:
        return sorted(self._models)

    def _load_sync(self, version: str) -> LoadedModel:
        path = self.models_dir / f"classifier-{version}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"model artifact not found: {path} (run `python -m app.core.train`)")
        # joblib usa pickle: sólo cargamos artefactos que construimos nosotros
        # en el build de la imagen. Para artefactos de terceros -> ONNX.
        pipeline = joblib.load(path)
        model = LoadedModel(version, pipeline, list(pipeline.classes_))
        model.predict_batch(["warmup"])  # warm-up: primera inferencia sin costo de lazy init
        return model

    async def load(self) -> None:
        loaded = await asyncio.gather(*(asyncio.to_thread(self._load_sync, v) for v in self.versions))
        for model in loaded:
            self._models[model.version] = model

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def resolve(self, requested: str | None) -> str:
        version = (requested or self.default_version).strip().lower()
        if version not in self._models:
            raise UnknownModelVersion(version, self.loaded_versions)
        return version

    async def classify(self, text: str, version: str) -> Prediction:
        model = self._models[version]
        loop = asyncio.get_running_loop()
        [result] = await loop.run_in_executor(self._executor, model.predict_batch, [text])
        return result
