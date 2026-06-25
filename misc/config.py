"""
Простой загрузчик конфигурации из .env для giga.
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Ошибка конфигурации (отсутствуют обязательные переменные)."""


@dataclass
class GigaChatConfig:
    basic_auth: str
    scope: str
    token_url: str
    api_base: str
    model: str
    embedding_model: str
    verify: bool

    def __post_init__(self):
        # GigaChat теперь опционален: основной провайдер — GitHub Models.
        # Не падаем при отсутствии кредов, чтобы пайплайн работал без GigaChat.
        if not self.basic_auth:
            logger.info(
                "GIGACHAT_BASIC_AUTH не задан — GigaChat отключён (используется GitHub Models)."
            )


@lru_cache(maxsize=1)
def load_config() -> GigaChatConfig:
    project_env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=project_env, override=False)
    load_dotenv(find_dotenv(), override=False)

    cfg = GigaChatConfig(
        basic_auth=os.getenv("GIGACHAT_BASIC_AUTH", "").strip(),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
        token_url=os.getenv(
            "GIGACHAT_TOKEN_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        ).strip(),
        api_base=os.getenv(
            "GIGACHAT_API_BASE", "https://gigachat.devices.sberbank.ru/api/v1"
        ).strip(),
        model=os.getenv("GIGACHAT_MODEL", "GigaChat").strip(),
        embedding_model=os.getenv("GIGACHAT_EMBEDDING_MODEL", "Embeddings").strip(),
        verify=os.getenv("GIGACHAT_VERIFY", "false").lower().strip() == "true",
    )

    logger.info("Конфигурация Gigachat загружена.")
    return cfg


def healthcheck() -> None:
    cfg = load_config()
    masked = f"{cfg.basic_auth[:6]}...{cfg.basic_auth[-6:]}" if cfg.basic_auth else "missing"
    print("GIGACHAT_BASIC_AUTH:", masked)
    print("GIGACHAT_SCOPE:", cfg.scope)
    print("GIGACHAT_TOKEN_URL:", cfg.token_url)
    print("GIGACHAT_API_BASE:", cfg.api_base)
    print("GIGACHAT_MODEL:", cfg.model)
    print("GIGACHAT_EMBEDDING_MODEL:", cfg.embedding_model)
    print("GIGACHAT_VERIFY:", cfg.verify)


if __name__ == "__main__":
    healthcheck()
