# Relatório de equidade do modelo de crédito

Razão de aprovação = aprovação do grupo menos aprovado / do mais aprovado, entre grupos com n ≥ 30. Abaixo de 0.8 (em negrito) a regra dos 4/5 indica impacto desproporcional.

## Variantes do baseline

| Variante | ROC AUC holdout | Aprovação holdout | Razão SEX holdout | Razão EDUCATION holdout | Razão MARRIAGE holdout | Razão AGE holdout | ROC AUC 2026-06 | Aprovação 2026-06 | Razão SEX 2026-06 | Razão EDUCATION 2026-06 | Razão MARRIAGE 2026-06 | Razão AGE 2026-06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| todas as features | 0.7798 | 70.3% | 0.929 | **0.638** | 0.873 | 0.874 | 0.6416 | 57.6% | 0.906 | 0.871 | **0.761** | **0.703** |
| sem SEX e MARRIAGE (adotada) | 0.7804 | 70.6% | 0.950 | **0.624** | **0.789** | 0.860 | 0.6402 | 60.1% | 0.989 | 0.840 | **0.780** | **0.746** |
| sem SEX, MARRIAGE e EDUCATION | 0.7809 | 70.6% | 0.954 | **0.687** | 0.876 | 0.872 | 0.6371 | 61.9% | 0.975 | 0.850 | **0.797** | **0.720** |

## Modelo em produção por grupo: holdout

### SEX (razão de aprovação 0.950)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| feminino | 3594 | 72.1% | 61.9% | 18.9% | 20.9% |
| masculino | 2406 | 68.5% | 62.3% | 22.2% | 23.3% |

### EDUCATION (razão de aprovação **0.624**)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| ensino médio | 958 | 62.4% | 69.1% | 26.9% | 25.4% |
| outros | 94 | 100.0% | 0.0% | 0.0% | 5.3% |
| pós-graduação | 2104 | 74.7% | 56.3% | 17.6% | 19.9% |
| universidade | 2844 | 69.4% | 63.6% | 20.9% | 22.7% |

### MARRIAGE (razão de aprovação **0.789**)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| casado | 2717 | 68.2% | 63.0% | 22.3% | 23.3% |
| outros | 66 | 57.6% | 75.0% | 32.0% | 24.2% |
| solteiro | 3217 | 72.9% | 60.8% | 18.3% | 20.6% |

### AGE (razão de aprovação 0.860)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| 18-25 | 744 | 67.1% | 65.2% | 20.7% | 27.4% |
| 26-35 | 2643 | 73.6% | 60.8% | 17.8% | 19.9% |
| 36-45 | 1700 | 71.2% | 59.7% | 20.7% | 20.7% |
| 46-60 | 864 | 63.3% | 65.8% | 26.6% | 25.7% |
| 61+ | 49 | 69.4% | 66.7% | 22.5% | 18.4% |


## Modelo em produção por grupo: 2026-06

### SEX (razão de aprovação 0.989)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| feminino | 598 | 60.4% | 52.3% | 34.4% | 29.4% |
| masculino | 402 | 59.7% | 57.3% | 32.7% | 30.9% |

### EDUCATION (razão de aprovação 0.840)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| ensino médio | 183 | 53.0% | 61.0% | 40.3% | 32.2% |
| outros | 23 | 100.0% | 0.0% | 0.0% | 4.3% |
| pós-graduação | 331 | 63.1% | 49.0% | 31.9% | 29.0% |
| universidade | 463 | 58.8% | 55.6% | 34.8% | 31.1% |

### MARRIAGE (razão de aprovação **0.780**)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| casado | 455 | 52.3% | 58.9% | 42.4% | 32.1% |
| outros | 10 | 40.0% | 50.0% | 62.5% | 20.0% |
| solteiro | 535 | 67.1% | 50.0% | 26.1% | 28.4% |

### AGE (razão de aprovação **0.746**)

| Grupo | n | Aprovação | Recall | FPR | Default real |
|---|---|---|---|---|---|
| 18-25 | 402 | 68.9% | 48.4% | 23.6% | 30.3% |
| 26-35 | 325 | 56.9% | 58.4% | 36.2% | 31.1% |
| 36-45 | 175 | 51.4% | 56.8% | 45.8% | 25.1% |
| 46-60 | 92 | 52.2% | 57.1% | 43.8% | 30.4% |
| 61+ | 6 | 16.7% | 80.0% | 100.0% | 83.3% |
