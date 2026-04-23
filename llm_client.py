import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


class LLMClientError(Exception):
    pass


def _read_from_secrets(key: str) -> Optional[str]:
    if st is None:
        return None
    try:
        value = st.secrets.get(key)
    except Exception:
        return None
    if value is None:
        return None
    return str(value)


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    env_value = os.getenv(key)
    if env_value is not None and str(env_value).strip():
        return env_value

    secret_value = _read_from_secrets(key)
    if secret_value is not None and secret_value.strip():
        return secret_value

    return default


def get_provider_name(provider: Optional[str] = None) -> str:
    return (provider or get_setting("LLM_PROVIDER", "mock") or "mock").strip().lower()


def get_llm_client(provider: Optional[str] = None) -> Any:
    provider_name = get_provider_name(provider)

    if provider_name in {"github", "cerebras"} and OpenAI is None:
        raise LLMClientError("Пакет openai не установлен. Установите зависимости: pip install -r requirements.txt")

    if provider_name == "github":
        token = get_setting("GITHUB_TOKEN")
        if not token:
            raise LLMClientError("Не найден GITHUB_TOKEN в переменных окружения или secrets")
        return OpenAI(
            api_key=token,
            base_url="https://models.inference.ai.azure.com",
            default_query={"api-version": "2024-05-01-preview"},
            timeout=30,
        )

    if provider_name == "cerebras":
        api_key = get_setting("CEREBRAS_API_KEY")
        if not api_key:
            raise LLMClientError("Не найден CEREBRAS_API_KEY в переменных окружения или secrets")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            timeout=30,
        )

    if provider_name == "mock":
        return None

    raise LLMClientError(f"Неподдерживаемый провайдер: {provider_name}")


def get_model_name(provider: Optional[str] = None) -> str:
    provider_name = get_provider_name(provider)
    if provider_name == "github":
        return get_setting("GITHUB_MODEL", "gpt-4o") or "gpt-4o"
    if provider_name == "cerebras":
        return get_setting("CEREBRAS_MODEL", "llama-3.3-70b") or "llama-3.3-70b"
    return "mock-model"


def build_prompt(user_query: str, schema_info: str, semantic_context: str, examples: str) -> str:
    return f"""
Ты — эксперт по генерации SQL для DuckDB.

Правила:
0) Сначала кратко объсяни как понял запрос.
1) Потом верни - только SQL без пояснений.
2) Разрешены только SELECT или WITH ... SELECT.
3) Не изменяй данные (никаких INSERT/UPDATE/DELETE/DDL).
4) Используй только таблицу incity_orders.
5) Используй одинарные кавычки в строковых литералах.
6) Добавь LIMIT 1000, если его нет.
7) Учитывай семантический слой и синонимы.

Схема таблицы incity_orders:
{schema_info}

Семантический слой:
{semantic_context}

Примеры:
{examples}

Запрос пользователя:
{user_query}
""".strip()


def _extract_sql(raw: str) -> str:
    if not raw:
        return ""

    text = raw.strip()

    fenced = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        text = text.replace("```", "").strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _mock_sql(user_query: str) -> str:
    q = user_query.lower()
    if "вчера" in q and "выполн" in q:
        return (
            "SELECT COUNT(*) AS done_orders "
            "FROM incity_orders "
            "WHERE status_order = 'done' "
            "AND DATE(order_timestamp) = CURRENT_DATE - INTERVAL 1 DAY "
            "LIMIT 1000"
        )
    if "отмен" in q and "7" in q:
        return (
            "SELECT DATE(order_timestamp) AS day, COUNT(*) AS cancels "
            "FROM incity_orders "
            "WHERE status_order = 'cancel' "
            "AND DATE(order_timestamp) >= CURRENT_DATE - INTERVAL 7 DAY "
            "GROUP BY 1 ORDER BY 1 LIMIT 1000"
        )
    if "средн" in q and ("стоим" in q or "чек" in q):
        return (
            "SELECT DATE(order_timestamp) AS day, AVG(price_order_local) AS avg_price "
            "FROM incity_orders "
            "WHERE status_order = 'done' "
            "AND DATE(order_timestamp) >= CURRENT_DATE - INTERVAL 7 DAY "
            "GROUP BY 1 ORDER BY 1 LIMIT 1000"
        )
    if "топ" in q and "выруч" in q:
        return (
            "SELECT STRFTIME(order_timestamp, '%Y-%m') AS month, "
            "SUM(price_order_local) AS revenue "
            "FROM incity_orders "
            "WHERE status_order = 'done' "
            "GROUP BY 1 ORDER BY revenue DESC LIMIT 3"
        )

    return "SELECT * FROM incity_orders LIMIT 1000"


def generate_sql(
    user_query: str,
    schema_info: str,
    semantic_context: str,
    examples: str,
    provider: Optional[str] = None,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> str:
    provider_name = get_provider_name(provider)

    if provider_name == "mock":
        return _mock_sql(user_query)

    active_client = client or get_llm_client(provider_name)
    active_model = model or get_model_name(provider_name)
    prompt = build_prompt(user_query, schema_info, semantic_context, examples)

    try:
        response = active_client.chat.completions.create(
            model=active_model,
            temperature=0.1,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "Ты генерируешь только валидный SQL для DuckDB."},
                {"role": "user", "content": prompt},
            ],
            timeout=30,
        )
    except Exception as e:
        raise LLMClientError(f"Ошибка обращения к LLM: {e}") from e

    content = ""
    if response and response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""

    sql = _extract_sql(content)
    if not sql:
        raise LLMClientError("LLM вернула пустой SQL")

    return sql
