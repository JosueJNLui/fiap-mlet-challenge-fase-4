.DEFAULT_GOAL := help

.PHONY: help install install-hooks uninstall-hooks validate validate-env validate-branch validate-commits validate-tags \
        lint test dvc-setup dvc-remote data-download data-push data-pull \
        reference train bad-batch demo-contract

UV      := uv --cache-dir /tmp/uv-cache
DVC     := $(UV) run dvc
BRANCH  ?= $(shell git branch --show-current)
COMMITS_RANGE ?= origin/main..HEAD
TAGS    ?=
DAGSHUB_USER  ?=
DAGSHUB_TOKEN ?=

# ── Geral ─────────────────────────────────────────────────────────────────────

help: ## Mostra esta mensagem de ajuda.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Instala as dependencias do projeto, incluindo ferramentas de dev.
	$(UV) sync --all-groups

install-hooks: ## Habilita os hooks locais de Git em .githooks.
	chmod +x .githooks/*
	git config core.hooksPath .githooks
	@echo "Git hooks installed from .githooks"

uninstall-hooks: ## Desabilita os hooks locais de Git do repositorio.
	git config --unset core.hooksPath || true
	@echo "Git hooks disabled"

# ── Validação ─────────────────────────────────────────────────────────────────

validate: lint test validate-branch validate-commits validate-tags ## Roda todas as validacoes.

lint: ## Roda o lint de src/, scripts/ e tests/ com ruff.
	$(UV) run ruff check src scripts tests

test: ## Roda a suite de testes.
	$(UV) run pytest -q

validate-branch: ## Valida a branch atual ou BRANCH=<nome>.
	$(UV) run python scripts/validate_branch.py "$(BRANCH)"

validate-commits: ## Valida os commits em COMMITS_RANGE, padrao origin/main..HEAD.
	$(UV) run python scripts/validate_commits.py --range "$(COMMITS_RANGE)"

validate-tags: ## Valida todas as tags do repositorio ou TAGS="1.2.3 2.0.0".
	$(UV) run python scripts/validate_tags.py $(TAGS)

validate-env: ## Verifica Python, deps criticas, .env e token DagsHub.
	$(UV) run python scripts/validate_env.py

# ── DagsHub / DVC ─────────────────────────────────────────────────────────────

dvc-setup: ## Credencializa o remote do DVC (requer DAGSHUB_USER e DAGSHUB_TOKEN no ambiente).
	@test -n "${DAGSHUB_USER}" || (echo "ERROR: DAGSHUB_USER is required.  Use: make dvc-setup DAGSHUB_USER=<user> DAGSHUB_TOKEN=<token>" && exit 1)
	@test -n "${DAGSHUB_TOKEN}" || (echo "ERROR: DAGSHUB_TOKEN is required.  Use: make dvc-setup DAGSHUB_USER=<user> DAGSHUB_TOKEN=<token>" && exit 1)
	$(DVC) remote modify --local origin auth basic
	$(DVC) remote modify --local origin user ${DAGSHUB_USER}
	$(DVC) remote modify --local origin password ${DAGSHUB_TOKEN}
	@echo "DVC remote configured successfully!"

dvc-remote: ## Mostra o remote do DVC configurado.
	$(DVC) remote list

data-download: ## Baixa o dataset da UCI e converte o .xls para CSV em data/raw/.
	$(UV) run python scripts/download_dataset.py

data-push: ## Versiona data/raw com DVC e envia para o remote DagsHub.
	$(DVC) add data/raw
	$(DVC) push -r origin

data-pull: ## Baixa data/raw do remote DagsHub (para reproducao).
	$(DVC) pull -r origin

# ── Etapa 1: contrato de dados e baseline ──────────────────────────────────────

reference: ## Gera data/reference/ (referencia congelada) e data/reference/holdout.csv.
	PYTHONPATH=src $(UV) run python -m creditmon.data

train: ## Treina o baseline XGBoost, loga no MLflow e grava os artefatos locais.
	PYTHONPATH=src $(UV) run python -m creditmon.train

bad-batch: ## Gera um lote sujo para demonstrar que o contrato bloqueia a ingestao.
	PYTHONPATH=src $(UV) run python scripts/create_bad_batch.py

demo-contract: bad-batch ## Demonstra o contrato: lote limpo passa, lote sujo e bloqueado.
	@echo "\n=== lote sujo (deve ser BLOQUEADO) ==="
	@PYTHONPATH=src $(UV) run python scripts/validate_batch.py data/bad_batch/bad_batch.csv || true
	@echo "\n=== lote de referencia (deve PASSAR) ==="
	@PYTHONPATH=src $(UV) run python scripts/validate_batch.py data/reference/reference.csv
