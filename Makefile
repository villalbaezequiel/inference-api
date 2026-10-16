.PHONY: up down logs install test lint format

up:            ## Levanta app + Redis
	docker compose up --build -d
	@echo "API en http://localhost:8000  (docs: /docs)"

down:
	docker compose down

logs:
	docker compose logs -f app

install:       ## Dependencias locales (venv recomendado)
	pip install -r requirements-dev.txt

test:          ## Corre la suite de tests
	pytest -q

lint:          ## Lint + chequeo de formato
	ruff check .
	ruff format --check .

format:        ## Aplica fixes de lint y formato
	ruff check --fix .
	ruff format .
