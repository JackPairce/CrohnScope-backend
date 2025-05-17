include .env
export

# Check if we are in a Docker container
DOCKER ?= false

ifeq ($(DOCKER),true)
# Use system Python when in Docker
venv :=
python := python
pip := pip
else
# Use virtual environment in local development
venv := .venv
python := $(venv)/bin/python
pip := $(venv)/bin/pip
endif

.PHONY: install dev run freeze lint

install:
ifeq ($(DOCKER),true)
	@echo "Installing packages in Docker environment..."
	@$(pip) install --no-cache-dir -r requirements.txt
else
	@echo "Setting up virtual environment..."
	@[ ! -e $(venv)/bin/python ] && python3 -m venv $(venv) || true
	@$(pip) install --upgrade pip
	@$(pip) install ipykernel
	@$(python) -m ipykernel install --user --name=my-venv --display-name "Python (.venv)"
	@$(pip) install -r requirements.txt
endif

dev: format
	@ENV=dev BASE_URL="http://$(HOST):$(PORT_DEV)" $(python) -m uvicorn app.main:app --reload --host $(HOST) --port $(PORT_DEV)

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
