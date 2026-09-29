"""Configuração central via Pydantic Settings, sobreposta pelo `.env`.

Fonte única de verdade para caminhos, seed e credenciais. O `.env` guarda o que é
segredo (token DagsHub, usuário); o resto tem default aqui. Precedência:
init > env > `.env`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseModel):
    """Diretórios de dados, modelos e relatórios."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    models: Path = Path("models")
    reports: Path = Path("reports")


class MlflowCfg(BaseModel):
    """Configuração de tracking e registro no DagsHub."""

    experiment_name: str = "credit-scoring-drift-fase-4"


class Settings(BaseSettings):
    """Configuração do projeto: `.env` e variáveis de ambiente sobrepõem os defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    mlflow: MlflowCfg = Field(default_factory=MlflowCfg)

    # Credenciais DagsHub — obrigatórias para tracking e remote do DVC.
    dagshub_repo_owner: str = "JosueJNLui"
    dagshub_repo_name: str = "fiap-mlet-challenge-fase-4"
    dagshub_user: str | None = None
    dagshub_token: str | None = None

    @property
    def dagshub_repo_slug(self) -> str:
        """Slug `owner/nome` do repositório no DagsHub."""
        return f"{self.dagshub_repo_owner}/{self.dagshub_repo_name}"

    @property
    def mlflow_tracking_uri(self) -> str:
        """URI do servidor MLflow hospedado no DagsHub."""
        return f"https://dagshub.com/{self.dagshub_repo_slug}.mlflow"

    @property
    def dvc_remote_url(self) -> str:
        """URL do remote DVC hospedado no DagsHub."""
        return f"https://dagshub.com/{self.dagshub_repo_slug}.dvc"


@lru_cache
def load_settings() -> Settings:
    """Carrega (e memoiza) a configuração do projeto."""
    return Settings()
