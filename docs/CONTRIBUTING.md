# Guia de contribuição

Como preparar o ambiente, seguir as convenções e validar o trabalho antes de commitar.

## Pré-requisitos

- Python 3.12 ou 3.13 (o projeto exige `>=3.12,<3.14`; o `.python-version` fixa 3.13).
- [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências.

## Setup do ambiente

```bash
make install            # uv sync --all-groups (deps de prod + dev)
cp .env.example .env    # preencha DAGSHUB_TOKEN / DAGSHUB_USER
make validate-env       # confere Python, deps criticas, .env e token DagsHub
```

O token sai de https://dagshub.com/settings/tokens. Sem ele o `make dvc-setup` (que
grava a credencial do remote em `.dvc/config.local`, gitignored) e o tracking MLflow
falham — ambos exigem o DagsHub.

## Convenções de Git

O projeto valida branches, commits e tags de forma automatizada. As regras completas
estão em [AGENTS.md](../AGENTS.md); em resumo:

- **Branches**: Conventional Branch 1.0.0, no formato `<tipo>/<descricao>` (tipos:
  `feature`, `feat`, `bugfix`, `fix`, `hotfix`, `release`, `chore`), além das trunk
  `main`/`master`/`develop`.
- **Commits**: Conventional Commits 1.0.0, no formato `<tipo>[escopo]: <descricao>`.
- **Tags**: SemVer estrito `MAJOR.MINOR.PATCH` (por exemplo `1.2.3`).

Os mesmos validadores rodam no GitHub Actions
(`.github/workflows/validate-conventions.yml`) e bloqueiam o merge do pull request.

### Validação obrigatória

Antes de considerar qualquer trabalho concluído, rode:

```bash
make validate           # lint + testes + validação de branch, commits e tags
```

`make validate` deve passar sem erros. Validações individuais e overrides:

```bash
make validate-branch BRANCH=feat/adicionar-pipeline
make validate-commits COMMITS_RANGE=origin/main..HEAD
make validate-tags TAGS="1.0.0 1.1.0"
```

### Hooks locais (opcionais)

```bash
make install-hooks      # habilita os hooks versionados em .githooks/
make uninstall-hooks    # desabilita
```

Hooks disponíveis: `pre-commit` (nome da branch), `commit-msg` (mensagem do commit),
`pre-push` (branch, commits e tags enviados).

## Padrão de código e idioma

- `ruff check` sem erros e `pytest` verde.
- Comentários, docstrings e documentação em pt-BR; termos técnicos permanecem em inglês.
  Mensagens/logs/echos podem ficar em inglês (não devem alterar comportamento).
