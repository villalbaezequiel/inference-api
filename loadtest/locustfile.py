"""Carga mixta (bonus): ~105 RPS sostenidos con 150 usuarios a 0.7 req/s cada uno.

    make up && make load
    # o interactivo: locust -f loadtest/locustfile.py --host http://localhost:8000

Mezcla: 60% /v1/infer (prompts de un set "caliente" -> mezcla de hits y misses),
25% /v1/classify (v1/v2), 15% /v1/stream (lee el stream completo).

Nota: el rate limit es 60 req/min por API key. Para medir capacidad (no el
limitador) `make load` levanta la app con RATE_LIMIT_PER_MINUTE alto.
"""

import random

from locust import HttpUser, constant_throughput, task

API_KEY = "dev-key-123"
HOT_PROMPTS = [f"Resumí el ticket de soporte #{i}" for i in range(50)]


class InferenceUser(HttpUser):
    wait_time = constant_throughput(0.7)

    def on_start(self):
        self.client.headers.update({"X-API-Key": API_KEY})

    @task(12)
    def infer(self):
        # 70% prompts repetidos (cacheables), 30% únicos (fuerzan llamada al LLM)
        prompt = random.choice(HOT_PROMPTS) if random.random() < 0.7 else f"pregunta única {random.random()}"
        self.client.post(
            "/v1/infer",
            json={"model": "gpt-4o", "input": prompt, "config": {"temperature": 0.7}},
            name="/v1/infer",
        )

    @task(5)
    def classify(self):
        self.client.post(
            "/v1/classify",
            json={"input": random.choice(["excelente servicio", "pésima atención", "llegó tarde pero bien"])},
            headers={"X-Model-Version": random.choice(["v1", "v2"])},
            name="/v1/classify",
        )

    @task(3)
    def stream(self):
        with self.client.post(
            "/v1/stream",
            json={"model": "gpt-4o", "input": f"stream {random.randint(0, 20)}"},
            stream=True,
            catch_response=True,
            name="/v1/stream",
        ) as response:
            body = b"".join(response.iter_content(chunk_size=None))
            if b"[DONE]" not in body:
                response.failure("stream sin [DONE]")
