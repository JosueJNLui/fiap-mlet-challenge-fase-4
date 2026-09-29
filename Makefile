DEFAULT_GOAL := help

.PHONY: help install install-hooks uninstall-hooks validate validate-branch validate-commits validate-tags \
        lint test

UV      := uv --cache-dir /tmp/uv-cache
BRANCH  ?= $(shell git branch --show-current)
COMMITS_RANGE ?= origin/main..HEAD
TAGS    ?=

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

lint: ## Roda o lint de scripts/ e tests/ com ruff.
	$(UV) run ruff check scripts tests

test: ## Roda a suite de testes.
	$(UV) run pytest -q

validate-branch: ## Valida a branch atual ou BRANCH=<nome>.
	$(UV) run python scripts/validate_branch.py "$(BRANCH)"

validate-commits: ## Valida os commits em COMMITS_RANGE, padrao origin/main..HEAD.
	$(UV) run python scripts/validate_commits.py --range "$(COMMITS_RANGE)"

validate-tags: ## Valida todas as tags do repositorio ou TAGS="1.2.3 2.0.0".
	$(UV) run python scripts/validate_tags.py $(TAGS)
