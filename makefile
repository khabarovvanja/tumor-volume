PYTHON_VERSION := 3.10.19

PYENV_ROOT := $(HOME)/.pyenv
PYENV_BIN := $(PYENV_ROOT)/bin/pyenv
PYTHON_BIN := $(PYENV_ROOT)/versions/$(PYTHON_VERSION)/bin/python
POETRY_REAL := $(HOME)/.local/bin/poetry
POETRY_WRAPPER := ./poetry

.PHONY: setup setup-local install-pyenv install-python install-poetry install-deps train infer

setup: install-pyenv install-python install-poetry create-poetry-wrapper install-deps
	@echo "✅ Environment setup completed"

install-pyenv:
	@if [ ! -x "$(PYENV_BIN)" ]; then \
		echo "🔧 Installing pyenv..."; \
		curl https://pyenv.run | bash; \
	else \
		echo "✅ pyenv already installed"; \
	fi

install-python:
	@if [ ! -x "$(PYTHON_BIN)" ]; then \
		echo "🐍 Installing Python $(PYTHON_VERSION)..."; \
		$(PYENV_BIN) install -s $(PYTHON_VERSION); \
	else \
		echo "✅ Python $(PYTHON_VERSION) already installed"; \
	fi; \
	$(PYENV_BIN) local $(PYTHON_VERSION)

install-poetry:
	@if [ ! -x "$(POETRY_REAL)" ]; then \
		echo "📦 Installing Poetry..."; \
		$(PYTHON_BIN) -m pip install --user poetry; \
	else \
		echo "✅ Poetry already installed"; \
	fi

create-poetry-wrapper:
	@if [ ! -f "$(POETRY_WRAPPER)" ]; then \
		echo "🔗 Creating local poetry wrapper"; \
		echo '#!/usr/bin/env bash' > $(POETRY_WRAPPER); \
		echo 'exec "$(POETRY_REAL)" "$$@"' >> $(POETRY_WRAPPER); \
		chmod +x $(POETRY_WRAPPER); \
	else \
		echo "✅ Poetry wrapper already exists"; \
	fi

install-deps:
	@echo "📚 Installing dependencies via Poetry"; \
	$(POETRY_WRAPPER) install
