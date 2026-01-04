PYTHON_VERSION := 3.10.19
PYENV_ROOT := $(HOME)/.pyenv

.PHONY: setup
setup:
	@echo ">>> Installing system dependencies"
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y \
		build-essential curl git \
		libssl-dev zlib1g-dev libbz2-dev \
		libreadline-dev libsqlite3-dev llvm \
		libncurses5-dev libncursesw5-dev \
		xz-utils tk-dev libffi-dev liblzma-dev; \
	elif command -v brew >/dev/null 2>&1; then \
		brew update && brew install \
		openssl readline sqlite3 xz zlib pyenv; \
	fi

	@echo ">>> Installing pyenv (if not exists)"
	@if [ ! -d "$(PYENV_ROOT)" ]; then \
		curl https://pyenv.run | bash; \
	fi

	@echo ">>> Initializing pyenv"
	@export PYENV_ROOT="$(PYENV_ROOT)" && \
	export PATH="$$PYENV_ROOT/bin:$$PATH" && \
	eval "$$(pyenv init -)" && \
	pyenv install -s $(PYTHON_VERSION) && \
	pyenv local $(PYTHON_VERSION)

	@echo ">>> Installing Poetry (if not installed)"
	@command -v poetry >/dev/null 2>&1 || \
		curl -sSL https://install.python-poetry.org | python$(PYTHON_VERSION) -

	@echo ">>> Configuring Poetry environment"
	@export PYENV_ROOT="$(PYENV_ROOT)" && \
	export PATH="$$PYENV_ROOT/bin:$$PATH" && \
	eval "$$(pyenv init -)" && \
	poetry env use $$(pyenv which python)

	@echo ">>> Installing project dependencies"
	@poetry install

	@echo ">>> Setup completed successfully"