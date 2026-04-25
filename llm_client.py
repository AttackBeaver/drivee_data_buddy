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
    if provider and str(provider).strip():
        return str(provider).strip().lower()

    env_provider = get_setting("LLM_PROVIDER")
    if env_provider and env_provider.strip():
        return env_provider.strip().lower()

    default_provider = get_setting("DEFAULT_PROVIDER", "github")
    return (default_provider or "github").strip().lower()


def _get_provider_base_url(provider_name: str) -> str:
    if provider_name == "github":
        return (
            get_setting("GITHUB_BASE_URL")
            or get_setting("GITHUB_MODELS_BASE_URL")
            or "https://models.inference.ai.azure.com"
        )

    if provider_name == "deepseek":
        return (
            get_setting("DEEPSEEK_BASE_URL")
            or get_setting("DEEPSEEK_MODELS_BASE_URL")
            or "https://api.deepseek.com"
        )

    return ""


def get_llm_client(provider: Optional[str] = None) -> Any:
    provider_name = get_provider_name(provider)

    if provider_name in {"github", "deepseek"} and OpenAI is None:
        raise LLMClientError("Пакет openai не установлен. Установите зависимости: pip install -r requirements.txt")

    if provider_name == "github":
        token = get_setting("GITHUB_TOKEN")
        if not token:
            raise LLMClientError("Не найден GITHUB_TOKEN в переменных окружения или secrets")

        base_url = _get_provider_base_url(provider_name)
        api_version = get_setting("GITHUB_API_VERSION", "2024-05-01-preview")

        return OpenAI(
            api_key=token,
            base_url=base_url,
            default_query={"api-version": api_version},
            timeout=30,
        )

    if provider_name == "deepseek":
        api_key = get_setting("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMClientError("Не найден DEEPSEEK_API_KEY в переменных окружения или secrets")

        base_url = _get_provider_base_url(provider_name)

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30,
        )

    if provider_name == "mock":
        return None

    raise LLMClientError(f"Неподдерживаемый провайдер: {provider_name}")


def get_model_name(provider: Optional[str] = None) -> str:
    provider_name = get_provider_name(provider)
    if provider_name == "github":
        return get_setting("GITHUB_MODEL", "gpt-4o") or "gpt-4o"
    if provider_name == "deepseek":
        return get_setting("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat"
    return "mock-model"


def build_prompt(
    user_query: str,
    schema_info: str,
    semantic_context: str,
    examples: str,
    table_name: str,
    time_column: Optional[str],
    status_column: Optional[str],
) -> str:
    time_hint = time_column or "(не определено)"
    status_hint = status_column or "(не определено)"
    return f"""
Ты — эксперт по генерации SQL для DuckDB.

Правила:
0) Подумай - как ты можешь объяснить этот запрос.
1) Потом верни - только SQL без пояснений.
2) Разрешены только SELECT или WITH ... SELECT.
3) Не изменяй данные (никаких INSERT/UPDATE/DELETE/DDL).
4) Используй только таблицу {table_name}.
5) Используй одинарные кавычки в строковых литералах.
6) Добавь LIMIT 1000, если его нет.
7) Учитывай семантический слой и синонимы.
8) Для времени используй поле {time_hint}.
9) Для статусов используй поле {status_hint}.

Схема таблицы {table_name}:
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


def _mock_sql(
    user_query: str,
    table_name: str = "incity",
    time_column: Optional[str] = None,
    status_column: Optional[str] = None,
) -> str:
    q = user_query.lower()
    date_expr = f"DATE(CAST({time_column} AS TIMESTAMP))" if time_column else "CURRENT_DATE"

    if "вчера" in q and status_column and "выполн" in q:
        return (
            f"SELECT COUNT(*) AS done_items "
            f"FROM {table_name} "
            f"WHERE {status_column} = 'done' "
            f"AND {date_expr} = CURRENT_DATE - INTERVAL 1 DAY "
            f"LIMIT 1000"
        )

    if ("отмен" in q and "7" in q and status_column and time_column):
        return (
            f"SELECT {date_expr} AS day, COUNT(*) AS cancels "
            f"FROM {table_name} "
            f"WHERE {status_column} = 'cancel' "
            f"AND {date_expr} >= CURRENT_DATE - INTERVAL 7 DAY "
            f"GROUP BY 1 ORDER BY 1 LIMIT 1000"
        )

    if "7" in q and time_column:
        return (
            f"SELECT {date_expr} AS day, COUNT(*) AS items "
            f"FROM {table_name} "
            f"WHERE {date_expr} >= CURRENT_DATE - INTERVAL 7 DAY "
            f"GROUP BY 1 ORDER BY 1 LIMIT 1000"
        )

    return f"SELECT * FROM {table_name} LIMIT 1000"


def generate_sql(
    user_query: str,
    schema_info: str,
    semantic_context: str,
    examples: str,
    provider: Optional[str] = None,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    table_name: str = "incity",
    time_column: Optional[str] = None,
    status_column: Optional[str] = None,
) -> str:
    provider_name = get_provider_name(provider)

    if provider_name == "mock":
        return _mock_sql(
            user_query=user_query,
            table_name=table_name,
            time_column=time_column,
            status_column=status_column,
        )

    active_client = client or get_llm_client(provider_name)
    active_model = model or get_model_name(provider_name)
    prompt = build_prompt(
        user_query=user_query,
        schema_info=schema_info,
        semantic_context=semantic_context,
        examples=examples,
        table_name=table_name,
        time_column=time_column,
        status_column=status_column,
    )

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
