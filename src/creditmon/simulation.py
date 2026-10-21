"""Simulação do dataset de produção: a passagem do tempo sob um choque econômico.

O ponto de partida é o **holdout** congelado na Etapa 1: dados que o modelo nunca viu
no treino. Ele é fatiado em lotes mensais e cada mês recebe uma intensidade ``s`` de
choque, de 0 (mês de controle, sem drift) até 1 (choque pleno). Os mecanismos são
intencionais e documentados, para que a detecção (``drift.py``) e a análise causal
(Etapa 4) possam ser conferidas contra um gabarito conhecido.

DATA DRIFT, muda P(X):
    D1 Inflação       ``BILL_AMT*`` sobe até +60% e ``PAY_AMT*`` cai até -40%
                      (fatura maior, pagamento menor: renda real comprimida)
    D2 Desemprego     até 30% dos clientes pioram 1-2 níveis em ``PAY_0``/``PAY_2``
                      (atrasos recentes)
    D3 Novo perfil    até 55% dos clientes são 8-15 anos mais jovens e têm limite 40%
                      menor (expansão comercial para público de entrada)

CONCEPT DRIFT, muda P(y|X):
    C1 Inadimplência  clientes com histórico limpo (``PAY_0 <= 0``) passam a inadimplir
       silenciosa     com probabilidade extra de até 12%: perderam renda, mas o atraso
                      ainda não aparece nas features.

Os valores simulados respeitam os limites de sanidade do contrato (``schema.py``):
dado com drift continua **válido**. É exatamente a distinção que o projeto quer
mostrar, já que o contrato não pega drift e a detecção de drift não pega dado inválido.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from creditmon.config import Settings
from creditmon.schema import (
    AGE_MIN,
    BILL_AMOUNT_COLUMNS,
    BILL_AMT_MAX,
    BILL_AMT_MIN,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_MAX,
    TARGET,
)

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
MIN_LIMIT_BAL = 10_000


@dataclass(frozen=True)
class Scenario:
    """Intensidade máxima (s = 1) de cada mecanismo de drift."""

    inflation_bill: float = 0.60
    income_squeeze_pay: float = 0.40
    unemployment_share: float = 0.30
    new_profile_share: float = 0.55
    silent_default_prob: float = 0.12


def severity(month_idx: int, n_months: int = len(MONTHS)) -> float:
    """Intensidade do choque no mês: 0 no primeiro mês, 1 no último, linear."""
    return month_idx / (n_months - 1)


def apply_drift(
    frame: pd.DataFrame,
    s: float,
    rng: np.random.Generator,
    scenario: Scenario | None = None,
) -> pd.DataFrame:
    """Aplica os mecanismos D1-D3 e C1 com intensidade ``s`` em [0, 1]."""
    sc = scenario or Scenario()
    out = frame.copy()
    n = len(out)
    if s <= 0:
        return out

    # D1 inflação: faturas maiores, pagamentos menores (com ruído individual)
    bill_factor = (1 + sc.inflation_bill * s) * rng.lognormal(0, 0.05 * s, size=n)
    pay_factor = (1 - sc.income_squeeze_pay * s) * rng.lognormal(0, 0.05 * s, size=n)
    for col in BILL_AMOUNT_COLUMNS:
        out[col] = (out[col] * bill_factor).round().clip(BILL_AMT_MIN, BILL_AMT_MAX)
    for col in PAY_AMOUNT_COLUMNS:
        out[col] = (out[col] * pay_factor).round().clip(lower=0)

    # D2 desemprego: piora do status de pagamento recente
    hit = rng.random(n) < sc.unemployment_share * s
    bump = rng.integers(1, 3, size=n)
    for col in ("PAY_0", "PAY_2"):
        worse = (out[col].clip(lower=0) + bump).clip(upper=PAY_STATUS_MAX)
        out[col] = np.where(hit, worse, out[col])

    # D3 novo perfil de cliente: mais jovem e com limite menor
    young = rng.random(n) < sc.new_profile_share * s
    younger = (out["AGE"] - rng.integers(8, 16, size=n)).clip(lower=max(AGE_MIN, 21))
    out["AGE"] = np.where(young, younger, out["AGE"])
    lower_limit = (out["LIMIT_BAL"] * 0.6).round(-4).clip(lower=MIN_LIMIT_BAL)
    out["LIMIT_BAL"] = np.where(young, lower_limit, out["LIMIT_BAL"])

    # C1 concept drift: inadimplência silenciosa em quem parece bom pagador
    looks_good = (frame["PAY_0"] <= 0).to_numpy() & (out[TARGET] == 0).to_numpy()
    flip = looks_good & (rng.random(n) < sc.silent_default_prob * s)
    out[TARGET] = np.where(flip, 1, out[TARGET])

    return out.astype(frame.dtypes.to_dict())


def simulate_production(
    holdout: pd.DataFrame, seed: int, months: list[str] = MONTHS
) -> dict[str, pd.DataFrame]:
    """Fatia o holdout em lotes mensais e aplica drift crescente."""
    rng = np.random.default_rng(seed)
    shuffled = holdout.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunks = np.array_split(np.arange(len(shuffled)), len(months))
    return {
        month: apply_drift(shuffled.iloc[idx].reset_index(drop=True), severity(i, len(months)), rng)
        for i, (month, idx) in enumerate(zip(months, chunks, strict=True))
    }


def production_dir(settings: Settings) -> Path:
    return settings.paths.processed / "production"


def batch_path(settings: Settings, month: str) -> Path:
    return production_dir(settings) / f"production_{month}.csv"


def save_batches(settings: Settings, batches: dict[str, pd.DataFrame]) -> list[Path]:
    paths = []
    for month, frame in batches.items():
        path = batch_path(settings, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def load_batch(settings: Settings, month: str) -> pd.DataFrame:
    path = batch_path(settings, month)
    if not path.exists():
        raise FileNotFoundError(f"Lote {path} ausente. Rode `make simulate`.")
    return pd.read_csv(path)


def main() -> int:
    """Gera os lotes mensais de produção em data/processed/production/."""
    from creditmon.config import load_settings
    from creditmon.data import load_reference

    settings = load_settings()
    _, holdout = load_reference(settings)
    batches = simulate_production(holdout, settings.seed)
    for path, (month, frame) in zip(save_batches(settings, batches), batches.items(), strict=True):
        print(
            f"{path} ({len(frame):,} linhas, choque {severity(MONTHS.index(month)):.0%}, "
            f"default {frame[TARGET].mean():.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
