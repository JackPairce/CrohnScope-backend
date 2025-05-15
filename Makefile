include .env
export

venv := .venv
python := $(venv)/bin/python
pip := $(venv)/bin/pip

.PHONY: install dev run freeze lint

install:
	@[ ! -e $(python) ] && python3 -m venv $(venv) || true
	@$(pip) install --upgrade pip
	@$(pip) install ipykernel
	@$(python) -m ipykernel install --user --name=my-venv --display-name "Python (.venv)"
	@$(pip) install -r requirements.txt

dev: format
	@ENV=dev BASE_URL="http://$(HOST):$(PORT)" $(python) -m uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

run:
	@$(python) -m uvicorn app.main:app --host $(HOST) --port $(PORT)

freeze:
	@$(pip) freeze > requirements.txt

format:
	@$(python) -m black app

lint: format
	@npx openapi-typescript "http://$(HOST):$(PORT)/openapi.json" -o ../frontend/src/api.d.ts
	@$(python) -m flake8 app

add:
	@$(pip) install $(package)
	@$(pip) freeze > requirements.txt
	@echo "✅ Package '$(package)' added and requirements.txt updated."

remove:
	@$(pip) uninstall -y $(package)
	@$(pip) freeze > requirements.txt
	@echo "🗑️ Package '$(package)' removed and requirements.txt updated."

upgrade:
	@$(pip) list --outdated --format=freeze | cut -d = -f 1 | xargs -n1 $(pip) install -U
	@$(pip) freeze > requirements.txt
	@echo "⬆️ All packages upgraded and requirements.txt updated."
