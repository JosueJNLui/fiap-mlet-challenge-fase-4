# Relatório de drift: produção vs. referência

Baseline (holdout): ROC AUC = **0.780**. Critério: PSI ≥ 0.25 ou (PSI ≥ 0.1 e KS p < 0.05).

| Mês | Features com drift | Drift share | Score médio | Aprovação | PSI do score | ROC AUC | Default real |
|---|---|---|---|---|---|---|---|
| 2026-01 | nenhuma | 0% | 0.409 | 69.5% | 0.005 | 0.801 | 20.9% |
| 2026-02 | PAY_2 | 4% | 0.444 | 66.5% | 0.029 | 0.778 | 24.4% |
| 2026-03 | PAY_2, AGE | 9% | 0.441 | 65.9% | 0.024 | 0.732 | 24.9% |
| 2026-04 | PAY_2, AGE, PAY_0, PAY_AMT1, PAY_AMT4, PAY_AMT6 | 26% | 0.467 | 60.6% | 0.068 | 0.701 | 26.7% |
| 2026-05 | PAY_2, AGE, PAY_0, PAY_AMT2, PAY_AMT1 | 22% | 0.462 | 62.3% | 0.056 | 0.608 | 28.0% |
| 2026-06 | PAY_2, AGE, PAY_AMT1, PAY_AMT2, PAY_0, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, LIMIT_BAL, BILL_AMT1 | 48% | 0.490 | 57.6% | 0.120 | 0.642 | 30.0% |

## Variáveis degradadas no último mês (2026-06)

| Feature | PSI | PSI (Evidently) | KS p-valor | Média ref. | Média atual | Severidade |
|---|---|---|---|---|---|---|
| PAY_2 | 0.754 | 0.755 | 9.1e-46 | -0.13 | 0.41 | critical |
| AGE | 0.509 | 0.517 | 2.7e-67 | 35.49 | 30.75 | critical |
| PAY_AMT1 | 0.287 | 0.015 | 2.3e-36 | 5,671 | 3,216 | critical |
| PAY_AMT2 | 0.269 | 0.020 | 4.1e-44 | 5,961 | 2,941 | critical |
| PAY_0 | 0.238 | 0.238 | 1.3e-35 | -0.02 | 0.49 | warning |
| PAY_AMT3 | 0.190 | 0.019 | 2.0e-26 | 5,258 | 3,534 | warning |
| PAY_AMT4 | 0.167 | 0.031 | 1.6e-26 | 4,881 | 2,432 | warning |
| PAY_AMT5 | 0.154 | 0.012 | 8.9e-29 | 4,819 | 2,899 | warning |
| PAY_AMT6 | 0.148 | 0.011 | 8.7e-23 | 5,159 | 3,311 | warning |
| LIMIT_BAL | 0.132 | 0.122 | 2.8e-19 | 167,227 | 129,030 | warning |
| BILL_AMT1 | 0.101 | 0.240 | 8.0e-16 | 50,927 | 88,460 | warning |
