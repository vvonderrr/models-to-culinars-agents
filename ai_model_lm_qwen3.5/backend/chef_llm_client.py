"""
Клиент к языковой модели через OpenAI-совместимый API.

Провайдер выбирается переменной окружения LLM_PROVIDER:
    lmstudio   — локальный LM Studio (разработка, бесплатно)
    openrouter — OpenRouter (работает 24/7, нужен ключ)
    together   — Together AI (работает 24/7, нужен ключ)

Адрес, имя модели и названия параметров для каждого провайдера подставляются
автоматически. При необходимости их можно переопределить через LLM_BASE_URL
и LLM_MODEL.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

logger = logging.getLogger(__name__)


# ---------- Ошибки, которые понимает остальной бэкенд ----------

class LlmError(Exception):
    """Базовая ошибка работы с моделью."""


class LlmConfigError(LlmError):
    """Неверная настройка: неизвестный провайдер или нет ключа. Ловится при старте сервера."""


class LlmUnavailableError(LlmError):
    """Сервер модели не запущен или недоступен по сети."""


class LlmTimeoutError(LlmError):
    """Модель не ответила за отведённое время."""


class LlmBadResponseError(LlmError):
    """Сервер ответил ошибкой или пустым/некорректным ответом."""


# ---------- Провайдеры ----------

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    default_model: str
    needs_api_key: bool
    repeat_penalty_field: str       # как провайдер называет штраф за повторы
    default_timeout_seconds: float
    default_max_retries: int
    extra_headers: tuple[tuple[str, str], ...] = ()


PROVIDERS: dict[str, ProviderConfig] = {
    "lmstudio": ProviderConfig(
        name="lmstudio",
        base_url="http://localhost:1234/v1",
        default_model="qwen/qwen3.5-9b",
        needs_api_key=False,
        repeat_penalty_field="repeat_penalty",
        default_timeout_seconds=120.0,  # локальная GPU отвечает 30–45 с
        default_max_retries=0,          # повтор на медленной модели только удвоит ожидание
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="qwen/qwen3.5-9b",
        needs_api_key=True,
        repeat_penalty_field="repetition_penalty",
        default_timeout_seconds=60.0,
        default_max_retries=2,          # в облаке сбои короткие, повтор обычно помогает
        extra_headers=(("X-Title", "Nyam Chef"),),  # имя приложения в статистике OpenRouter
    ),
    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        default_model="Qwen/Qwen3.5-9B",
        needs_api_key=True,
        repeat_penalty_field="repetition_penalty",
        default_timeout_seconds=60.0,
        default_max_retries=2,
    ),
}


# ---------- Настройки диалога ----------

# Системный промпт кулинарного консультанта. Правит только ML (Катя), после правки — проверка через test_lm.py
SYSTEM_PROMPT = """Ты — дружелюбный кулинарный консультант в приложении «Ням Шеф». Ты помогаешь пользователю готовить: отвечаешь на вопросы по рецептам, подсказываешь замены ингредиентов, объясняешь термины и поддерживаешь живой диалог.

## Как ты работаешь
- Если в диалоге уже есть рецепт, отвечай по нему: уточняй детали, но не переписывай рецепт целиком.
- Если рецепта нет и пользователь спрашивает, что приготовить, предложи 2–3 подходящих блюда, по одной фразе о каждом, и спроси, какое расписать подробнее.
- Если для хорошего ответа не хватает данных, задай 1–2 коротких встречных вопроса по делу, например: «Лазанья с мясом или овощная?», «Готовить будете в духовке или в мультиварке?», «На сколько человек?». Не задавай анкетных вопросов вроде «Какой у вас опыт?».

## С чем ты помогаешь
- Ингредиенты: сколько класть и чем заменить («Можно заменить сметану на йогурт?»).
- Процесс: время, огонь, признаки готовности («Как понять, что курица готова?»).
- Термины: объясняй простыми словами («Пассеровать — значит слегка обжарить на масле до мягкости»).
- Проверка понимания: подтверждай или мягко поправляй («Да, сначала лук, а соль — в конце»).
- Адаптация: менее жирно, постно, по-вегански, на другое число порций.

## Стиль ответа
- Дружелюбно и терпеливо, как друг, который учит готовить.
- Длина ответа зависит от вопроса. На уточняющий вопрос — 2–5 предложений. Если пользователь просит рецепт или пошаговую инструкцию, дай её полностью: ингредиенты с количествами списком через дефис и 4–8 пронумерованных шагов. Если рецепт уже есть в диалоге, не повторяй его целиком, а отвечай на вопрос.
- Любой кулинарный термин сразу объясняй.
- Количества указывай в граммах, миллилитрах, ложках и штуках.
- Пиши только по-русски, без английских слов: не "cheese", а «сыр»; не "lasagna sheets", а «листы для лазаньи». Названия блюд пиши кириллицей («паста», «карбонара»).
- Пиши обычным текстом, без разметки: без символов #, ** и таблиц. Дефисы в списках и нумерация шагов (1., 2., 3.) допустимы.
- Не больше 1–2 эмодзи за ответ, никаких флагов.

## Безопасность
- Несъедобное или опасное (клей, мыло, стекло, гвозди): не обвиняй пользователя, по-доброму объясни, что это опасно для здоровья, и предложи готовить из настоящих продуктов. Пример: «Из гвоздей и клея еду не приготовить — это опасно для здоровья! А что найдётся в холодильнике? С настоящими продуктами я с радостью помогу».
- Пищевая безопасность: если вопрос о сыром мясе, сырых яйцах, хранении или подозрительных продуктах, советуй только безопасный вариант (например, курицу всегда доводить до полной готовности, продукты с плесенью выбрасывать).
- Аллергии и диеты по медицинским причинам: дай общий совет и порекомендуй уточнить у врача.
- Вопросы не про кулинарию: вежливо скажи, что помогаешь только с готовкой, и предложи вернуться к рецепту.

## Тон пользователя
Различай три случая:

1. Обычное сообщение — отвечай как обычно.

2. Требовательный или приказной тон без оскорблений («быстро ответь», «сделай сейчас же», «давай живее»). Сначала полностью ответь на вопрос по существу. Затем добавь в конце одну короткую доброжелательную фразу о вежливости, например: «И давайте общаться чуть вежливее — так готовить приятнее 😊». Не упрекай пользователя, не повторяй просьбу о вежливости дважды и не добавляй её, если тон обычный.

3. Мат, прямые оскорбления («ты тупой», «дурак»), угрозы или явное хамство. Весь твой ответ — ровно одна фраза, без каких-либо добавлений до или после неё:
Извините, я не могу помочь с этим запросом. Пожалуйста, будьте вежливы.

## Примеры
Пользователь: А сколько соли класть в жаркое?
Ты: Для жаркого на 500 г курицы и 6 картофелин хватит 1–1,5 чайной ложки соли. Положите половину в середине готовки, а в конце попробуйте и досолите. Рассказать, какие специи сюда ещё подойдут?

Пользователь: Можно вместо курицы взять грибы?
Ты: Конечно! Грибы жарятся быстрее, примерно 5–7 минут, а блюдо станет постным и более лёгким. Подсказать, какие грибы лучше выбрать? 😊

Пользователь: Помоги с лазаньей
Ты: С удовольствием! Лазанья будет с мясом или овощная? И листы теста у вас обычные или без варки?

Пользователь: Что приготовить из яиц и сыра?
Ты: Есть несколько вкусных вариантов: омлет с сыром — самый быстрый завтрак, запеканка из яиц, молока и тёртого сыра — сытно и без хлопот, а горячие тосты с сыром и яйцом — если хочется чего-то хрустящего. Какой рецепт расписать подробнее?

Пользователь: Быстро ответь, сколько варить яйцо всмятку
Ты: 3–4 минуты после закипания воды: белок схватится, а желток останется жидким. И давайте общаться чуть вежливее — так готовить приятнее 😊

Пользователь: Быстро дай рецепт, тупой
Ты: Извините, я не могу помочь с этим запросом. Пожалуйста, будьте вежливы."""

HISTORY_LIMIT = 10  # последних сообщений истории (5 пар «вопрос–ответ»)
ALLOWED_ROLES = frozenset({"user", "assistant"})

# Qwen3.5 по умолчанию «думает» вслух в блоке <think>...</think>
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class GenerationParams:
    """Параметры генерации. Сюда переносятся значения из правой панели LM Studio."""

    max_tokens: int = 1024        # Response Length Limit
    temperature: float = 0.7      # Temperature
    top_p: float = 0.8            # Top P
    top_k: int = 20               # Top K
    min_p: float = 0.0            # Min P
    repeat_penalty: float = 1.1   # Repeat Penalty


def _resolve_provider(name: str | None) -> ProviderConfig:
    key = (name or os.getenv("LLM_PROVIDER") or "lmstudio").strip().lower()
    provider = PROVIDERS.get(key)
    if provider is None:
        raise LlmConfigError(
            f"Неизвестный LLM_PROVIDER='{key}'. Допустимые значения: {', '.join(PROVIDERS)}"
        )
    return provider


class ChefLlmClient:
    """Один экземпляр на всё приложение: внутри переиспользуется HTTP-пул соединений."""

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        params: GenerationParams | None = None,
    ) -> None:
        self._provider = _resolve_provider(provider)
        self._params = params or GenerationParams()
        self._model = model or os.getenv("LLM_MODEL") or self._provider.default_model

        key = api_key or os.getenv("LLM_API_KEY")
        if self._provider.needs_api_key and not key:
            raise LlmConfigError(
                f"Для провайдера '{self._provider.name}' нужен ключ: задайте LLM_API_KEY"
            )

        try:
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", self._provider.default_timeout_seconds))
            retries = int(os.getenv("LLM_MAX_RETRIES", self._provider.default_max_retries))
        except ValueError as exc:
            raise LlmConfigError("LLM_TIMEOUT_SECONDS и LLM_MAX_RETRIES должны быть числами") from exc

        self._client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL") or self._provider.base_url,
            # LM Studio ключ не проверяет, но SDK требует непустую строку
            api_key=key or "lm-studio",
            timeout=timeout,
            max_retries=retries,
            default_headers=dict(self._provider.extra_headers),
        )
        logger.info("LLM: провайдер=%s, модель=%s", self._provider.name, self._model)

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._model

    async def reply(self, user_message: str, history: Iterable[Mapping[str, str]] = ()) -> str:
        """Возвращает финальный текст ответа модели без блока рассуждений."""
        text = (user_message or "").strip()
        if not text:
            raise ValueError("user_message не может быть пустым")

        messages = self._build_messages(text, history)
        p = self._params

        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=p.max_tokens,
                temperature=p.temperature,
                top_p=p.top_p,
                # Нестандартные для OpenAI параметры — под именами, которые понимает провайдер
                extra_body={
                    "top_k": p.top_k,
                    "min_p": p.min_p,
                    self._provider.repeat_penalty_field: p.repeat_penalty,
                },
            )
        except APITimeoutError as exc:
            raise LlmTimeoutError(f"{self._provider.name}: модель не ответила вовремя") from exc
        except APIConnectionError as exc:
            raise LlmUnavailableError(f"{self._provider.name}: сервер недоступен: {exc}") from exc
        except APIStatusError as exc:
            # 401 — неверный ключ, 402 — кончились деньги, 404 — неверный id модели, 429 — лимит запросов
            raise LlmBadResponseError(
                f"{self._provider.name} вернул HTTP {exc.status_code}: {str(exc.message)[:300]}"
            ) from exc

        if not completion.choices or completion.choices[0].message.content is None:
            raise LlmBadResponseError(f"{self._provider.name}: в ответе нет текста")

        cleaned = _THINK_RE.sub("", completion.choices[0].message.content).strip()
        if not cleaned:
            raise LlmBadResponseError(f"{self._provider.name}: модель вернула пустой ответ")

        if completion.usage:
            logger.info(
                "LLM %s: %d токенов на вход, %d на выход",
                self._provider.name,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
        return cleaned

    async def is_available(self) -> bool:
        """Проверка для health-эндпоинта: отвечает ли сервер модели."""
        try:
            await self._client.models.list()
            return True
        except (APIConnectionError, APITimeoutError, APIStatusError):
            return False

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _build_messages(user_message: str, history: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
        cleaned_history = [
            {"role": str(m["role"]), "content": m["content"]}
            for m in history
            if m.get("role") in ALLOWED_ROLES and isinstance(m.get("content"), str)
        ]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *cleaned_history[-HISTORY_LIMIT:],
            {"role": "user", "content": user_message},
        ]
