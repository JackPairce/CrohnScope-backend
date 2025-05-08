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

dev:
	@$(python) -m uvicorn app.main:app --reload

run:
	@$(python) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

freeze:
	@$(pip) freeze > requirements.txt

lint:
	@$(python) -m black app
	@$(python) -m flake8 app
