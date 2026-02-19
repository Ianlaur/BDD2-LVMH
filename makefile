# Makefile
# Docker-first reproducible workflow + optional local venv dev workflow

IMAGE_NAME ?= lvmh-client-intelligence
CONTAINER_NAME ?= lvmh-client-intelligence

# System python (for local runs if you want)
PYTHON ?= python3

# Host directories (mounted into container)
DATA_RAW ?= $(PWD)/data/raw
DATA_PROCESSED ?= $(PWD)/data/processed
DATA_OUTPUTS ?= $(PWD)/data/outputs
TAXONOMY ?= $(PWD)/taxonomy
ACTIVATIONS ?= $(PWD)/activations
DEMO ?= $(PWD)/demo
MODELS ?= $(PWD)/models

# Inside-container paths
C_DATA_RAW := /app/data/raw
C_DATA_PROCESSED := /app/data/processed
C_DATA_OUTPUTS := /app/data/outputs
C_TAXONOMY := /app/taxonomy
C_ACTIVATIONS := /app/activations
C_DEMO := /app/demo
C_MODELS := /app/models

# Local venv configuration
VENV_DIR ?= .venv
PYTHON_BIN ?= python3
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make build                  Build docker image"
	@echo "  make run                    Run full pipeline in docker (with mounted folders)"
	@echo "  make shell                  Shell into container"
	@echo "  make pipeline               Run full pipeline locally (uses PYTHON=$(PYTHON))"
	@echo "  make setup-models           Download embedding model locally into ./models (uses PYTHON=$(PYTHON))"
	@echo "  make venv                   Create local venv and install deps (uses PYTHON_BIN=$(PYTHON_BIN))"
	@echo "  make dev                    Run full pipeline locally using venv"
	@echo "  make setup-models-local     Download embedding model using venv into ./models"
	@echo "  make clean                  Remove generated outputs"

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: run
run:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-v "$(DATA_RAW):$(C_DATA_RAW)" \
		-v "$(DATA_PROCESSED):$(C_DATA_PROCESSED)" \
		-v "$(DATA_OUTPUTS):$(C_DATA_OUTPUTS)" \
		-v "$(TAXONOMY):$(C_TAXONOMY)" \
		-v "$(ACTIVATIONS):$(C_ACTIVATIONS)" \
		-v "$(DEMO):$(C_DEMO)" \
		-v "$(MODELS):$(C_MODELS)" \
		$(IMAGE_NAME) make pipeline

.PHONY: shell
shell:
	docker run --rm -it \
		--name $(CONTAINER_NAME)-shell \
		-v "$(DATA_RAW):$(C_DATA_RAW)" \
		-v "$(DATA_PROCESSED):$(C_DATA_PROCESSED)" \
		-v "$(DATA_OUTPUTS):$(C_DATA_OUTPUTS)" \
		-v "$(TAXONOMY):$(C_TAXONOMY)" \
		-v "$(ACTIVATIONS):$(C_ACTIVATIONS)" \
		-v "$(DEMO):$(C_DEMO)" \
		-v "$(MODELS):$(C_MODELS)" \
		$(IMAGE_NAME) bash

# -------- Local (non-docker) pipeline --------

.PHONY: pipeline
pipeline:
	$(PYTHON) -m src.run_all

.PHONY: setup-models
setup-models:
	$(PYTHON) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models/sentence_transformers')"

# -------- Local venv --------

.PHONY: venv
venv:
	$(PYTHON_BIN) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

.PHONY: dev
dev:
	$(VENV_PY) -m src.run_all

.PHONY: setup-models-local
setup-models-local:
	$(VENV_PY) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models/sentence_transformers')"

.PHONY: clean
clean:
	rm -rf data/processed/* data/outputs/* taxonomy/* demo/*