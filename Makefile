include .env
export

API_venv := API/.venv
API_python := $(API_venv)/bin/python
API_pip := $(API_venv)/bin/pip

AI_venv := AI/.venv
AI_python := $(AI_venv)/bin/python
AI_pip := $(AI_venv)/bin/pip

.PHONY: install install_api install_ai dev run freeze lint build

install_api:
	@[ ! -e $(API_venv)/bin/python ] && echo '[API] Setting up virtual environment...' || true
	@[ ! -e $(API_venv)/bin/python ] && python3 -m venv $(API_venv) || true
	@echo "[API] Upgrading PIP"
	@$(API_pip) install --upgrade pip
	@echo "[API] Installing Requirements"
	@$(API_pip) install -r API/requirements.txt

install_ai:
	@[ ! -e $(AI_venv)/bin/python ] && echo '[AI] Setting up virtual environment...' || true
	@[ ! -e $(AI_venv)/bin/python ] && python3 -m venv $(AI_venv) || true
	@echo "[AI] Upgrading PIP"
	@$(AI_pip) install --upgrade pip
	@echo "[AI] Installing Requirements"
	@$(AI_pip) install -r AI/requirements.txt

install: install_api install_ai


dev: format
	@ENV=dev BASE_URL="http://$(HOST):$(PORT_DEV)" $(API_python) -m uvicorn API.main:app --reload --host $(HOST) --port $(PORT_DEV)

run:
	@$(API_python) -m uvicorn API.main:app --host $(HOST) --port $(PORT)

freeze:
	@$(API_pip) freeze > API/requirements.txt

format:
	@$(API_python) -m black API AI

lint: format
	@npx openapi-typescript "http://$(HOST):$(PORT_DEV)/openapi.json" -o ../frontend/src/lib/api/types.d.ts
	@$(API_python) -m flake8 app

add:
	@$(API_pip) install $(package)
	@$(API_pip) freeze > API/requirements.txt
	@echo "✅ Package '$(package)' added and requirements.txt updated."

remove:
	@$(API_pip) uninstall -y $(package)
	@$(API_pip) freeze > API/requirements.txt
	@echo "🗑️ Package '$(package)' removed and requirements.txt updated."

upgrade:
	@$(API_pip) list --outdated --format=freeze | cut -d = -f 1 | xargs -n1 $(API_pip) install -U
	@$(API_pip) freeze > API/requirements.txt
	@echo "⬆️ All packages upgraded and requirements.txt updated."

build:
	@echo "Building binary with Nuitka..."
	@$(API_pip) install nuitka
	@mkdir -p build
	@$(API_python) -m nuitka --standalone --onefile --output-dir=build --output-filename=app app/main.py
	@echo "Building Docker image..."
	docker build -t crohnscope-backend .
