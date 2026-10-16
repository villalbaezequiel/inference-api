"""Entrena y serializa los modelos de clasificación v1 y v2.

Se ejecuta en el build de la imagen Docker (`python -m app.core.train`), así
el contenedor arranca con artefactos listos y la API sólo los *carga*.

    v1: TF-IDF palabras (1-2 gramas) + Logistic Regression
    v2: TF-IDF palabras + caracteres (robusto a typos/flexiones) + Logistic Regression

El dataset es chico y sintético a propósito: el challenge evalúa cómo se sirve el
modelo, no su calidad. En producción el artefacto vendría de un model registry.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

POSITIVE = [
    "excelente producto, lo recomiendo",
    "me encantó el servicio, muy rápido",
    "muy buena atención y calidad",
    "funciona perfecto, estoy feliz",
    "la entrega llegó antes de lo esperado",
    "increíble experiencia, volvería a comprar",
    "súper conforme con la compra",
    "el soporte resolvió todo enseguida",
    "great product, highly recommend",
    "i love it, works perfectly",
    "amazing service and fast delivery",
    "very happy with the quality",
    "todo impecable, gracias",
    "buenísimo, cumple lo que promete",
    "la app es rápida y fácil de usar",
    "atención de diez, muy amables",
    "best purchase i made this year",
    "excellent support team",
    "me sirvió muchísimo, genial",
    "calidad premium a buen precio",
]

NEGATIVE = [
    "pésimo producto, no lo recomiendo",
    "el servicio fue lento y malo",
    "muy mala atención, nunca más",
    "no funciona, estoy decepcionado",
    "la entrega llegó tarde y rota",
    "horrible experiencia, quiero un reembolso",
    "nada conforme con la compra",
    "el soporte nunca respondió",
    "terrible product, do not buy",
    "i hate it, it broke after a day",
    "awful service and slow delivery",
    "very disappointed with the quality",
    "todo mal, una estafa",
    "malísimo, no cumple lo que promete",
    "la app se cuelga todo el tiempo",
    "atención pésima, muy groseros",
    "worst purchase ever",
    "useless support team",
    "no me sirvió para nada, un desastre",
    "calidad horrible y caro",
]


def build(version: str) -> Pipeline:
    if version == "v1":
        features = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
    elif version == "v2":
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ]
        )
    else:
        raise ValueError(f"unknown version {version}")
    return Pipeline([("features", features), ("clf", LogisticRegression(C=5.0, max_iter=1000))])


def train(out_dir: Path, versions: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = POSITIVE + NEGATIVE
    labels = ["positive"] * len(POSITIVE) + ["negative"] * len(NEGATIVE)
    for version in versions:
        pipeline = build(version)
        pipeline.fit(texts, labels)
        joblib.dump(pipeline, out_dir / f"classifier-{version}.joblib")
        meta = {
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "classes": list(pipeline.classes_),
            "n_samples": len(texts),
        }
        (out_dir / f"classifier-{version}.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {version} -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models")
    parser.add_argument("--versions", default="v1,v2")
    args = parser.parse_args()
    train(Path(args.out), [v.strip() for v in args.versions.split(",")])
