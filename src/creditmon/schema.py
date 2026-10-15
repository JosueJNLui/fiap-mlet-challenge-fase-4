"""Contrato de dados do dataset de credit scoring (Pandera).

Regras rígidas que bloqueiam a ingestão de lixo antes do treino ou da predição. As
regras são deliberadamente generosas no que é válido (ex.: `BILL_AMT` pode ser negativo,
porque saldo credor existe) e rígidas no que é impossível (nulos, domínios categóricos,
idade fora de faixa, alvo fora de {0, 1}).

A separação importa: o contrato responde "este dado é válido?", a detecção de drift da
Etapa 2 responde "esta distribuição mudou?". Dado de produção pode ser perfeitamente
válido e ainda assim sofrer drift.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

TARGET = "default payment next month"
ID_COLUMN = "ID"

CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Limites de sanidade (não os observados no dataset): folgados o bastante para que
# dados de produção com drift econômico continuem válidos, e apertados o bastante para
# barrar valores impossíveis. `AGE_GT` casa com a regra "idade > 18" do brief.
AGE_MIN, AGE_MAX = 18, 100
LIMIT_BAL_MIN, LIMIT_BAL_MAX = 0, 2_000_000
BILL_AMT_MIN, BILL_AMT_MAX = -1_000_000, 2_000_000
PAY_AMT_MIN, PAY_AMT_MAX = 0, 2_000_000
PAY_STATUS_MIN, PAY_STATUS_MAX = -2, 8

# Domínios observados na UCI. EDUCATION/MARRIAGE têm códigos 0/5/6 e 0/3 que a
# documentação não lista, mas aparecem no dataset e são válidos (categorias "other").
SEX_DOMAIN = [1, 2]
EDUCATION_DOMAIN = [0, 1, 2, 3, 4, 5, 6]
MARRIAGE_DOMAIN = [0, 1, 2, 3]
TARGET_DOMAIN = [0, 1]


def _numeric_column(name: str, low: int, high: int) -> pa.Column:
    """Coluna inteira obrigatória, sem nulos, dentro de `[low, high]`."""
    return pa.Column(
        int,
        name=name,
        nullable=False,
        checks=[pa.Check.ge(low), pa.Check.le(high)],
    )


def _categorical_column(name: str, domain: list[int]) -> pa.Column:
    """Coluna inteira obrigatória, sem nulos, restrita a um domínio fechado."""
    return pa.Column(int, name=name, nullable=False, checks=[pa.Check.isin(domain)])


CREDIT_SCHEMA = pa.DataFrameSchema(
    columns={
        ID_COLUMN: pa.Column(
            int,
            name=ID_COLUMN,
            nullable=False,
            unique=True,
            checks=[pa.Check.gt(0)],
        ),
        "LIMIT_BAL": _numeric_column("LIMIT_BAL", LIMIT_BAL_MIN, LIMIT_BAL_MAX),
        **{c: _categorical_column(c, SEX_DOMAIN) for c in ["SEX"]},
        "EDUCATION": _categorical_column("EDUCATION", EDUCATION_DOMAIN),
        "MARRIAGE": _categorical_column("MARRIAGE", MARRIAGE_DOMAIN),
        "AGE": _numeric_column("AGE", AGE_MIN, AGE_MAX),
        **{c: _numeric_column(c, PAY_STATUS_MIN, PAY_STATUS_MAX) for c in PAY_STATUS_COLUMNS},
        **{c: _numeric_column(c, BILL_AMT_MIN, BILL_AMT_MAX) for c in BILL_AMOUNT_COLUMNS},
        **{c: _numeric_column(c, PAY_AMT_MIN, PAY_AMT_MAX) for c in PAY_AMOUNT_COLUMNS},
        TARGET: _categorical_column(TARGET, TARGET_DOMAIN),
    },
    strict=True,
    coerce=True,
    name="credit_scoring_contract",
)


class DataContractError(Exception):
    """Lote rejeitado pelo contrato de dados.

    Carrega um resumo legível das regras violadas, para log e para o relatório da demo.
    """

    def __init__(self, batch_name: str, summary: str, failure_count: int) -> None:
        self.batch_name = batch_name
        self.summary = summary
        self.failure_count = failure_count
        super().__init__(
            f"Contrato de dados rejeitou o lote '{batch_name}': "
            f"{failure_count} violação(ões).\n{summary}"
        )


def format_failures(error: SchemaErrors, max_rows: int = 10) -> str:
    """Resume as violações de um `SchemaErrors` como texto tabular."""
    cases = error.failure_cases
    if cases is None or cases.empty:
        return "(sem detalhes)"

    label_column = "column" if "column" in cases.columns else "check"
    grouped = (
        cases.groupby(label_column)["check"]
        .agg(lambda checks: ", ".join(sorted(set(checks))))
        .sort_index()
    )

    lines = [f"  - {column}: {checks}" for column, checks in grouped.items()]
    if len(grouped) > max_rows:
        lines.append(f"  ... e mais {len(grouped) - max_rows} coluna(s).")
    return "\n".join(lines)


def validate_batch(frame: pd.DataFrame, *, name: str = "batch") -> pd.DataFrame:
    """Valida um lote contra o contrato de dados.

    Args:
        frame: DataFrame com as 25 colunas do dataset de crédito.
        name: rótulo do lote, usado na mensagem de erro e no log.

    Returns:
        O DataFrame validado (e coagido aos dtypes do contrato).

    Raises:
        DataContractError: se alguma regra for violada. A ingestão deve ser bloqueada.
    """
    try:
        return CREDIT_SCHEMA.validate(frame, lazy=True)
    except SchemaErrors as error:
        raise DataContractError(
            name,
            format_failures(error),
            failure_count=len(error.failure_cases),
        ) from error


def main() -> int:
    """Self-check: o schema aceita um lote limpo e rejeita cada tipo de violação."""
    clean = pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3],
            "LIMIT_BAL": [20_000, 120_000, 90_000],
            "SEX": [1, 2, 2],
            "EDUCATION": [2, 2, 1],
            "MARRIAGE": [1, 2, 2],
            "AGE": [24, 26, 34],
            **{c: [0, 1, -1] for c in PAY_STATUS_COLUMNS},
            **{c: [3913, 3102, 689] for c in BILL_AMOUNT_COLUMNS},
            **{c: [0, 689, 0] for c in PAY_AMOUNT_COLUMNS},
            TARGET: [1, 1, 0],
        }
    )
    validate_batch(clean, name="self-check-clean")
    print("clean batch aceito")

    for label, mutate in {
        "nulo em LIMIT_BAL": lambda d: d.assign(LIMIT_BAL=[None, 1, 1]),
        "idade de menor": lambda d: d.assign(AGE=[15, 26, 34]),
        "SEX fora do dominio": lambda d: d.assign(SEX=[3, 2, 2]),
        "alvo fora de {0,1}": lambda d: d.assign(**{TARGET: [2, 1, 0]}),
    }.items():
        try:
            validate_batch(mutate(clean), name=label)
        except DataContractError as error:
            print(f"rejeitado: {label} ({error.failure_count} violacoes)")
        else:
            raise AssertionError(f"contrato deveria ter rejeitado: {label}")

    print("contrato OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
