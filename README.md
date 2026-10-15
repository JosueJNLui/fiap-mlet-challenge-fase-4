# FIAP MLET — Tech Challenge Fase 4

Observabilidade contínua, detecção de drift, qualidade de dados e governança para um
modelo de **credit scoring** em produção.

O problema: uma fintech tem um modelo de risco de crédito rodando, a economia muda e o
modelo degrada **em silêncio**. Este repositório constrói a camada de sustentação e
confiabilidade — contrato de dados, baseline rastreável, simulação/detecção de drift,
monitoramento e documentação de compliance (LGPD).

- Dataset: [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (30.000 clientes, 25 colunas, alvo binário `default payment next month`).
- Tracking + storage: [DagsHub](https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-4) (MLflow + DVC).

## Sumário

- [Arquitetura](#arquitetura)
- [Setup](#setup)
- [Etapa 1 — Contrato de dados e baseline](#etapa-1--contrato-de-dados-e-baseline)
- [Comandos](#comandos)
- [Governança e LGPD](#governança-e-lgpd)
- [Convenções de Git](#convenções-de-git)

## Arquitetura

```text
data/
  raw/         # dataset cru (UCI) versionado via DVC
  reference/   # referência congelada (24k) + holdout limpo (6k), via DVC
  bad_batch/   # lote sujo gerado sob demanda para a demo do contrato
models/        # baseline.pkl + baseline_metrics.json (derivados, fora do Git)
reports/       # evidências (HTML de drift, etc.) — versionadas de propósito
src/creditmon/
  config.py    # Settings (Pydantic) — paths, seed e credenciais
  schema.py    # contrato de dados (Pandera) — bloqueia a ingestão
  data.py      # carga, validação e congelamento de referência/holdout
  train.py     # baseline XGBoost + MLflow (log, registro e alias `production`)
  tracking.py  # configuração do MLflow no DagsHub
scripts/       # download do dataset, validação de lote, gates de Git
tests/         # pytest
```

Separação central: **referência** (usada no treino) vs. **produção** (onde o drift
ocorre, na Etapa 2). O contrato responde "este dado é válido?"; a detecção de drift
responde "esta distribuição mudou?". Dado de produção pode ser válido e ainda assim
sofrer drift — por isso o contrato é intencionalmente generoso nos limites e rígido
apenas no que é impossível.

## Setup

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.12/3.13.

```bash
make install                     # uv sync --all-groups
cp .env.example .env             # preencha DAGSHUB_USER e DAGSHUB_TOKEN
make validate-env                # checa Python, deps, .env e token
make install-hooks               # opcional: habilita os hooks de Git
```

Credenciais: crie um token em `https://dagshub.com/user/settings/tokens` e preencha
`.env` (que é gitignored):

```dotenv
DAGSHUB_USER=seu-usuario
DAGSHUB_TOKEN=seu-token
```

## Etapa 1 — Contrato de dados e baseline

**Modelo baseline.** XGBoost (`n_estimators=200`, `max_depth=4`, `learning_rate=0.1`,
`scale_pos_weight` corrigindo o desbalanceamento ~78/22), treinado na referência e
avaliado no holdout. Cada execução loga parâmetros e métricas no MLflow, registra o
modelo como `credit_scoring_baseline` e promove a versão mais recente para o alias
`production`.

**Contrato de dados.** `src/creditmon/schema.py` (Pandera) bloqueia a ingestão antes do
treino/predição. Regras rígidas:

| Regra | Coluna(s) |
| :-- | :-- |
| Sem nulos | todas |
| `ID` único e positivo | `ID` |
| Idade plausível | `AGE ∈ [18, 100]` |
| Domínios fechados | `SEX ∈ {1,2}`, `EDUCATION ∈ {0..6}`, `MARRIAGE ∈ {0..3}` |
| Alvo binário | `default payment next month ∈ {0,1}` |
| Limites de sanidade | `LIMIT_BAL ∈ [0, 2M]`, `BILL_AMT ∈ [-1M, 2M]`, `PAY_AMT ≥ 0`, `PAY_* ∈ [-2, 8]` |
| Schema estrito | nenhuma coluna extra além das 25 esperadas |

`BILL_AMT` pode ser negativo de propósito (saldo credor é um estado válido); o contrato
não barra valores economicamente possíveis, apenas impossíveis.

**Demonstração do bloqueio.** `scripts/create_bad_batch.py` injeta violações (nulo,
idade de menor, domínio inválido, alvo inválido, `ID` duplicado) e
`scripts/validate_batch.py` rejeita o lote com exit code `1`:

```bash
make demo-contract
```

## Comandos

```bash
make help            # lista todos os alvos
make install         # instala as dependências
make data-download   # baixa o dataset da UCI e converte para CSV
make reference       # congela data/reference/ (referência + holdout)
make train           # treina o baseline e loga no MLflow/DagsHub
make bad-batch       # gera o lote sujo da demo
make demo-contract   # demonstra o contrato aceitando o limpo e bloqueando o sujo
make validate        # lint + testes + gates de branch/commit/tag
make test            # pytest
make lint            # ruff
```

Fluxo completo do zero:

```bash
make install && make data-download && make reference && make train && make demo-contract
```

## Governança e LGPD

Plano de proteção de dados do projeto (detalhamento na Etapa 4):

- **Dados e PII.** O dataset é público e **anonimizado pela UCI**: sem nome, CPF,
  endereço ou qualquer identificador direto. O campo `ID` é um índice sintético do
  arquivo, não uma identidade. Ainda assim é tratado como identificador (não é
  feature do modelo) e nunca é exposto em relatórios.
- **Base legal.** A decisão de crédito se apoia em *legítimo interesse* e *execução de
  contrato* (LGPD, art. 7º, V e IX), com o titular ciente do processo de análise.
- **Minimização.** Só entram no modelo as 23 features preditivas; `ID` e o alvo são
  excluídos das features. Nenhum dado sensível é inferido ou criado.
- **Retenção.** Dados de referência e lotes de produção ficam versionados via DVC no
  DagsHub e são reprocessáveis; logs de experimento no MLflow guardam métricas e
  parâmetros, não o dado bruto.
- **Vieses.** `SEX`, `EDUCATION` e `MARRIAGE` são variáveis historicamente sensíveis.
  A Etapa 2 monitora drift nessas distribuições para dar visibilidade a possíveis
  disparidades, e o modelo é auditável (parâmetros e versão rastreados no MLflow).
- **Direito do titular.** O contrato de dados e a separação referência/produção
  permitem reconstruir e auditar o que influenciou cada decisão.

## Convenções de Git

O repositório aplica gates automatizados (CI + hooks opcionais):

- **Branches** — Conventional Branch 1.0.0: `feat/`, `fix/`, `chore/`, `release/…`.
- **Commits** — Conventional Commits 1.0.0: `feat:`, `fix(scope):`, `docs:`, etc.
- **Tags** — SemVer estrito `MAJOR.MINOR.PATCH`.

```bash
make validate          # roda todos os gates
make install-hooks     # ativa as validações no pre-commit/commit-msg/pre-push
```

## Roadmap

- [x] **Etapa 1** — contrato de dados + baseline treinado.
- [ ] **Etapa 2** — simulação e detecção de drift (Evidently AI, PSI/KS).
- [ ] **Etapa 3** — observabilidade de pipeline e modelo (MLflow/DagsHub, métricas).
- [ ] **Etapa 4** — governança (LGPD), análise causal e vídeo STAR.
