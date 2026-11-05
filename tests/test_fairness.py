from __future__ import annotations

import pandas as pd

from creditmon import fairness
from creditmon.schema import TARGET


def test_group_metrics_and_approval_ratio_by_hand() -> None:
    # Mulheres: 4 de 5 aprovadas; homens: 2 de 5. Razão = 0,4 / 0,8 = 0,5.
    frame = pd.DataFrame({
        "SEX":       [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        "EDUCATION": [1] * 10,
        "MARRIAGE":  [1] * 10,
        "AGE":       [30] * 10,
        TARGET:      [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    })  # fmt: skip
    denied = pd.Series([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])

    table = fairness.group_metrics(frame[TARGET], denied, fairness.protected_groups(frame)["SEX"])
    rows = table.set_index("group")
    assert rows.loc["feminino", "approval_rate"] == 0.8
    assert rows.loc["masculino", "approval_rate"] == 0.4
    assert rows.loc["feminino", "recall"] == 0.5  # 1 de 2 inadimplentes negado
    assert rows.loc["masculino", "fpr"] == round(2 / 3, 4)  # 2 de 3 bons pagadores negados

    assert fairness.approval_ratio(table, min_n=5) == 0.5
    assert fairness.approval_ratio(table, min_n=6) is None  # grupos pequenos ficam fora

    result = fairness.audit(frame, denied)
    assert result["EDUCATION"]["approval_ratio"] is None  # um grupo só, sem comparação
