# Instruções para agentes de IA

Este repositório usa gates automatizados para validar nomes de branches,
mensagens de commits e tags. Ao gerar qualquer alteração, branch, commit ou tag,
siga obrigatoriamente os padrões abaixo.

## Branches

Use Conventional Branch 1.0.0.

Formato:

```text
<type>/<description>
```

Tipos aceitos:

- `feature`
- `feat`
- `bugfix`
- `fix`
- `hotfix`
- `release`
- `chore`

Branches de tronco também são aceitas:

- `main`
- `master`
- `develop`

Regras:

- use apenas letras minúsculas, números, hífens e pontos;
- não use espaços, underscores ou letras maiúsculas;
- não use hífen ou ponto no início ou no fim da descrição;
- não use separadores consecutivos, como `--` ou `..`;
- para releases, prefira `release/v1.2.0`.

Exemplos válidos:

```text
feat/adicionar-pipeline
feature/issue-123-new-login
fix/corrigir-validacao
hotfix/security-patch
release/v1.2.0
chore/update-dependencies
```

## Commits

Use Conventional Commits 1.0.0.

Formato:

```text
<type>[optional scope]: <description>
```

Tipos recomendados:

- `feat`: nova funcionalidade;
- `fix`: correção de bug;
- `docs`: documentação;
- `style`: formatação sem alteração de comportamento;
- `refactor`: refatoração sem correção de bug ou nova funcionalidade;
- `perf`: melhoria de performance;
- `test`: testes;
- `build`: build ou dependências;
- `ci`: integração contínua;
- `chore`: tarefas auxiliares;
- `revert`: reversão.

Exemplos válidos:

```text
feat: add validation pipeline
fix(api): handle empty payload
docs: update setup instructions
ci: add github actions gates
chore: update development dependencies
```

Breaking changes devem usar `!` ou footer `BREAKING CHANGE:`.

Exemplos:

```text
feat!: remove legacy endpoint
feat(api)!: change prediction response format
```

```text
feat: change prediction response format

BREAKING CHANGE: response payload now returns probabilities by class.
```

## Tags

Use versionamento semântico estrito no formato:

```text
MAJOR.MINOR.PATCH
```

Exemplos válidos:

```text
0.1.0
1.0.0
1.2.3
2.0.0
```

Exemplos inválidos:

```text
v1.0.0
1.0
1.0.0-beta
release-1.0.0
```

## Validação local

Antes de finalizar alterações, execute:

```bash
make validate
```

Para validar itens específicos:

```bash
make validate-branch
make validate-commits
make validate-tags
```

Também é possível passar valores explicitamente:

```bash
make validate-branch BRANCH=feat/adicionar-pipeline
make validate-commits COMMITS_RANGE=origin/main..HEAD
make validate-tags TAGS="1.0.0 1.1.0"
```

## Hooks locais

Os hooks de Git deste repositório ficam em `.githooks/`, mas são opcionais.
Eles só devem ser considerados ativos quando o usuário executar:

```bash
make install-hooks
```

Esse comando configura `core.hooksPath` para `.githooks`.

Para remover a configuração local:

```bash
make uninstall-hooks
```

Os hooks executam:

- `pre-commit`: valida o nome da branch atual;
- `commit-msg`: valida a mensagem do commit;
- `pre-push`: valida branches, commits e tags enviados ao remoto.

## Regra para agentes

Sempre que um agente de IA sugerir ou executar comandos Git neste repositório,
ele deve:

1. criar branches usando Conventional Branch;
2. escrever commits usando Conventional Commits;
3. criar tags somente no formato `MAJOR.MINOR.PATCH`;
4. executar `make validate` antes de considerar a tarefa concluída, quando
   houver alterações relacionadas a Git conventions, CI ou validações.
