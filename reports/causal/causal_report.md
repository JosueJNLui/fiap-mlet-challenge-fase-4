# Análise causal da degradação (ablação dos mecanismos)

Holdout inteiro (6.000 clientes), intensidade 1,0, mesma seed em todos os cenários. Δ AUC = ROC AUC do cenário menos a do cenário sem choque. Na remoção, "AUC recuperada" é a fração da queda do choque completo que some ao desligar o mecanismo.

| Cenário | Tipo | Drift share | PSI do score | ROC AUC | Δ AUC | AUC recuperada | Recall | Aprovação | Default real |
|---|---|---|---|---|---|---|---|---|---|
| sem choque | controle | 0% | 0.002 | 0.7800 | +0.0000 |  | 62.5% | 70.3% | 21.9% |
| choque completo | controle | 67% | 0.087 | 0.6416 | -0.1384 |  | 56.7% | 58.8% | 30.3% |
| só D1 inflação | isolado | 48% | 0.078 | 0.7718 | -0.0082 |  | 65.8% | 65.8% | 21.9% |
| só D2 desemprego | isolado | 10% | 0.071 | 0.7419 | -0.0381 |  | 68.8% | 59.6% | 21.9% |
| só D3 novo perfil | isolado | 10% | 0.021 | 0.7618 | -0.0182 |  | 57.0% | 73.9% | 21.9% |
| só C1 inadimplência silenciosa | isolado | 0% | 0.002 | 0.6783 | -0.1017 |  | 48.3% | 70.3% | 30.3% |
| sem D1 inflação | remoção | 19% | 0.045 | 0.6482 | -0.1318 | 5% | 53.4% | 62.5% | 30.3% |
| sem D2 desemprego | remoção | 57% | 0.013 | 0.6611 | -0.1189 | 14% | 47.2% | 70.2% | 30.3% |
| sem D3 novo perfil | remoção | 57% | 0.194 | 0.6514 | -0.1286 | 7% | 60.4% | 54.9% | 30.3% |
| sem C1 inadimplência silenciosa | remoção | 67% | 0.087 | 0.7169 | -0.0631 | 54% | 67.3% | 58.8% | 21.9% |

## Dose-resposta do choque completo

Mesmas intensidades dos lotes mensais, mas sobre o holdout inteiro: separa o efeito do choque da variação amostral dos lotes de 1000 clientes. "IC 95% num lote" é o intervalo bootstrap (300 sorteios) da ROC AUC de um lote mensal de 1000 clientes na mesma intensidade.

| Mês | Intensidade | Drift share | PSI do score | ROC AUC | IC 95% num lote | Aprovação | Default real |
|---|---|---|---|---|---|---|---|
| 2026-01 | 0% | 0% | 0.001 | 0.7804 | 0.745 a 0.813 | 70.6% | 21.9% |
| 2026-02 | 20% | 0% | 0.007 | 0.7422 | 0.706 a 0.777 | 68.5% | 23.5% |
| 2026-03 | 40% | 5% | 0.019 | 0.7106 | 0.675 a 0.746 | 66.2% | 25.2% |
| 2026-04 | 60% | 24% | 0.035 | 0.6843 | 0.649 a 0.717 | 63.9% | 26.8% |
| 2026-05 | 80% | 29% | 0.060 | 0.6605 | 0.624 a 0.693 | 61.2% | 28.5% |
| 2026-06 | 100% | 67% | 0.087 | 0.6416 | 0.608 a 0.675 | 58.8% | 30.3% |
