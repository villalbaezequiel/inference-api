.PHONY: up down logs install models test lint format load

up:            ## Levanta app + Redis
	docker compose up --build -d
	@echo "API en http://localhost:8000  (docs: /docs)"

down:
	docker compose down

logs:
	docker compose logs -f app

install:       ## Dependencias locales (venv recomendado)
	pip install -r requirements-dev.txt

models:        ## Entrena los artefactos v1/v2 en ./models (para correr fuera de Docker)
	python -m app.core.train --out models

test:          ## Corre la suite de tests
	pytest -q

lint:          ## Lint + chequeo de formato
	ruff check .
	ruff format --check .

format:        ## Aplica fixes de lint y formato
	ruff check --fix .
	ruff format .

load:          ## Bonus: locust headless 60 s (~105 RPS). Rate limit alto para medir capacidad, no el limitador.
	mkdir -p loadtest/results
	RATE_LIMIT_PER_MINUTE=1000000 docker compose up -d --build --wait app
	docker compose --profile load run --rm --no-deps locust
