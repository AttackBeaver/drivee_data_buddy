import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from guardrails import validate_sql
from llm_client import (
    LLMClientError,
    generate_sql,
    get_llm_client,
    get_model_name,
    get_provider_name,
)
try:
    import duckdb
except ImportError:
    duckdb = None
st.set_page_config(page_title="Drivee Data Buddy",
                   page_icon="📊", layout="wide")
st.title("Drivee Data Buddy")

CSV_PATH = Path("data/incity_orders.csv")
DB_PATH = Path("data/drivee.duckdb")
SEMANTIC_PATH = Path("semantic/semantic_layer.json")
TABLE_NAME = "incity_orders"


@st.cache_resource
def init_database(csv_path: str):
    """Инициализация DuckDB и загрузка выбранного CSV в таблицу."""
    if duckdb is None:
        raise RuntimeError(
            "Модуль duckdb не установлен. Установите зависимости из requirements.txt")

    source = Path(csv_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Файл {source.as_posix()} не найден. Укажите корректный CSV."
        )

    con = duckdb.connect(DB_PATH.as_posix())
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT *
        FROM read_csv_auto('{source.as_posix()}')
        """
    )
    return con


@st.cache_data
def load_semantic_layer(path: str = SEMANTIC_PATH.as_posix()):
    """Загрузка семантического слоя из JSON."""
    semantic_file = Path(path)
    if not semantic_file.exists():
        raise FileNotFoundError(
            f"Файл {semantic_file.as_posix()} не найден. Создайте semantic/semantic_layer.json"
        )

    with semantic_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_data_dirs() -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)


def save_uploaded_csv(uploaded_file) -> Path:
    content = uploaded_file.getvalue()
    file_hash = hashlib.md5(content).hexdigest()[:12]
    safe_name = uploaded_file.name.replace(" ", "_")
    target = Path("data/uploads") / f"{file_hash}_{safe_name}"
    target.write_bytes(content)
    return target


def get_dataset_requirements() -> Dict[str, List[str]]:
    return {
        "required": ["order_timestamp", "status_order"],
        "recommended": [
            "price_order_local",
            "distance_in_meters",
            "duration_in_seconds",
            "user_id",
            "driver_id",
            "offset_hours",
        ],
    }


def validate_dataset_columns(columns: List[str]) -> Tuple[bool, List[str], List[str]]:
    reqs = get_dataset_requirements()
    required = reqs["required"]
    recommended = reqs["recommended"]

    missing_required = [c for c in required if c not in columns]
    missing_recommended = [c for c in recommended if c not in columns]
    return len(missing_required) == 0, missing_required, missing_recommended


def filter_semantic_layer_by_columns(semantic_layer: dict, columns: List[str]) -> dict:
    filtered = copy.deepcopy(semantic_layer)
    available = set(columns)

    def is_available(item: dict) -> bool:
        deps = item.get("depends_on", [])
        return all(dep in available for dep in deps)

    filtered["metrics"] = {
        name: cfg
        for name, cfg in semantic_layer.get("metrics", {}).items()
        if is_available(cfg)
    }
    filtered["dimensions"] = {
        name: cfg
        for name, cfg in semantic_layer.get("dimensions", {}).items()
        if is_available(cfg)
    }
    filtered["filters"] = {
        name: cfg
        for name, cfg in semantic_layer.get("filters", {}).items()
        if is_available(cfg)
    }

    available_terms = set(filtered["metrics"].keys()) | set(
        filtered["dimensions"].keys())
    filtered["synonyms"] = {
        src: target
        for src, target in semantic_layer.get("synonyms", {}).items()
        if target in available_terms
    }

    return filtered


def init_chat_if_needed() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "text": (
                    "Привет! Напиши мне свой запрос на естественном языке.\n\n"
                    "Примеры запросов:\n"
                    "- Сколько выполненных заказов за вчера?\n"
                    "- Отмены по дням за последние 7 дней\n"
                    "- Средний чек по месяцам"
                ),
            }
        ]


def get_metric_sql(semantic_layer: dict, metric_name: str) -> Tuple[Optional[str], Optional[str]]:
    metric = semantic_layer.get("metrics", {}).get(metric_name)
    if not metric:
        return None, None
    return metric.get("sql"), metric.get("requires")


def get_dimension_field(semantic_layer: dict, dimension_name: str) -> Optional[str]:
    dimension = semantic_layer.get("dimensions", {}).get(dimension_name)
    if not dimension:
        return None
    return dimension.get("field")


def resolve_time_expression(semantic_layer: dict, user_text: str) -> Optional[str]:
    lowered = user_text.lower()
    for phrase, expr in semantic_layer.get("time_expressions", {}).items():
        if phrase in lowered:
            return expr
    return None


def resolve_metric_and_dimension(semantic_layer: dict, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Простой резолвер: ищет метрику и измерение по фразе и синонимам."""
    text = user_text.lower()
    synonyms = semantic_layer.get("synonyms", {})

    normalized_text = text
    for src, target in synonyms.items():
        normalized_text = normalized_text.replace(src, target)

    metric_name = None
    for name in semantic_layer.get("metrics", {}).keys():
        if name in normalized_text:
            metric_name = name
            break

    dimension_name = None
    for name in semantic_layer.get("dimensions", {}).keys():
        if name in normalized_text:
            dimension_name = name
            break

    return metric_name, dimension_name


def query_needs_dimension(user_text: str) -> bool:
    text = user_text.lower()
    markers = ["по ", "в разрезе", "групп",
               "динамик", "по дням", "по недел", "по месяц"]
    return any(marker in text for marker in markers)


def build_sql_from_semantics(
    semantic_layer: dict,
    metric_name: str,
    dimension_name: Optional[str] = None,
    user_text: str = "",
) -> str:
    metric_sql, metric_requires = get_metric_sql(semantic_layer, metric_name)
    if not metric_sql:
        raise ValueError(f"Неизвестная метрика: {metric_name}")

    select_parts = [f"{metric_sql} AS value"]
    group_by = ""
    order_by = " ORDER BY value DESC"

    if dimension_name:
        dim_field = get_dimension_field(semantic_layer, dimension_name)
        if not dim_field:
            raise ValueError(f"Неизвестное измерение: {dimension_name}")
        select_parts.insert(0, f"{dim_field} AS dimension")
        group_by = " GROUP BY 1"
        order_by = " ORDER BY 1"

    where_clauses = []
    if metric_requires:
        where_clauses.append(metric_requires)

    time_expr = resolve_time_expression(semantic_layer, user_text)
    if time_expr:
        where_clauses.append(f"DATE(order_timestamp) >= {time_expr}")

    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    return (
        f"SELECT {', '.join(select_parts)} "
        f"FROM {TABLE_NAME}"
        f"{where_sql}"
        f"{group_by}"
        f"{order_by}"
    )


def run_query(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def format_schema_info(schema_df: pd.DataFrame) -> str:
    lines = []
    for _, row in schema_df.iterrows():
        lines.append(f"- {row['column_name']}: {row['data_type']}")
    return "\n".join(lines)


def format_semantic_context(semantic_layer: dict) -> str:
    parts = ["Метрики:"]
    for name, cfg in semantic_layer.get("metrics", {}).items():
        requires = cfg.get("requires")
        req_text = f" | requires: {requires}" if requires else ""
        parts.append(f"- {name}: {cfg.get('sql', '')}{req_text}")

    parts.append("\nИзмерения:")
    for name, cfg in semantic_layer.get("dimensions", {}).items():
        parts.append(f"- {name}: {cfg.get('field', '')}")

    parts.append("\nВременные выражения:")
    for name, expr in semantic_layer.get("time_expressions", {}).items():
        parts.append(f"- {name}: {expr}")

    parts.append("\nФильтры:")
    for name, cfg in semantic_layer.get("filters", {}).items():
        if isinstance(cfg, dict):
            parts.append(f"- {name}: {cfg.get('sql', '')}")
        else:
            parts.append(f"- {name}: {cfg}")

    return "\n".join(parts)


def get_few_shot_examples() -> str:
    return """
Вопрос: Сколько всего заказов было выполнено за вчера?
SQL:
SELECT COUNT(*) AS done_orders
FROM incity_orders
WHERE status_order = 'done'
  AND DATE(order_timestamp) = CURRENT_DATE - INTERVAL 1 DAY
LIMIT 1000;

Вопрос: Покажи отмены по дням за последние 7 дней
SQL:
SELECT DATE(order_timestamp) AS day, COUNT(*) AS cancels
FROM incity_orders
WHERE status_order = 'cancel'
  AND DATE(order_timestamp) >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY 1
ORDER BY 1
LIMIT 1000;

Вопрос: Какая средняя стоимость поездки по дням за прошлую неделю?
SQL:
SELECT DATE(order_timestamp) AS day, AVG(price_order_local) AS avg_price
FROM incity_orders
WHERE status_order = 'done'
  AND DATE(order_timestamp) >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY 1
ORDER BY 1
LIMIT 1000;
""".strip()


@st.cache_resource
def init_llm_runtime(provider: str):
    model = get_model_name(provider)
    client = get_llm_client(provider)
    return client, model


def build_auto_chart(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return None

    chart_df = df.copy()
    datetime_cols = [
        c for c in chart_df.columns if pd.api.types.is_datetime64_any_dtype(chart_df[c])
    ]
    numeric_cols = [
        c for c in chart_df.columns if pd.api.types.is_numeric_dtype(chart_df[c])
    ]

    if not datetime_cols:
        for c in chart_df.columns:
            name = c.lower()
            if any(k in name for k in ["date", "day", "time", "month", "week", "hour", "день", "дата", "месяц", "недел", "час"]):
                parsed = pd.to_datetime(chart_df[c], errors="coerce")
                if parsed.notna().mean() >= 0.6:
                    chart_df[c] = parsed
                    datetime_cols.append(c)
                    break

    if datetime_cols and numeric_cols:
        return px.line(chart_df, x=datetime_cols[0], y=numeric_cols[0], title="Динамика")

    if len(numeric_cols) >= 2:
        return px.scatter(chart_df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter")

    categorical_cols = [
        c
        for c in chart_df.columns
        if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(chart_df[c])
    ]
    if categorical_cols and numeric_cols:
        return px.bar(chart_df, x=categorical_cols[0], y=numeric_cols[0], title="Категории")

    if numeric_cols:
        return px.bar(chart_df.reset_index(), x="index", y=numeric_cols[0], title="Результат")

    return None


if duckdb is None:
    st.error(
        "Модуль `duckdb` не установлен. Установите зависимости: `pip install -r requirements.txt`."
    )
    st.stop()

if not SEMANTIC_PATH.exists():
    st.error(
        "Файл `semantic/semantic_layer.json` не найден. "
        "Создайте семантический слой и перезапустите приложение."
    )
    st.stop()

ensure_data_dirs()

st.sidebar.header("Источник данных")
st.sidebar.caption(
    "Можно использовать дефолтный CSV или загрузить свой датасет.")

uploaded_file = st.sidebar.file_uploader(
    "Загрузить CSV датасет",
    type=["csv"],
    help="Минимально нужны колонки: order_timestamp, status_order",
)

if uploaded_file is not None:
    uploaded_path = save_uploaded_csv(uploaded_file)
    st.sidebar.success(f"Файл сохранён: `{uploaded_path.name}`")
    st.session_state["selected_dataset"] = uploaded_path.as_posix()

available_datasets = []
if CSV_PATH.exists():
    available_datasets.append(CSV_PATH.as_posix())
available_datasets.extend(
    sorted([p.as_posix() for p in Path("data/uploads").glob("*.csv")])
)

if not available_datasets:
    st.error(
        "Не найдено ни одного CSV. Добавьте файл в `data/` или загрузите через сайдбар.")
    st.stop()

default_dataset = st.session_state.get(
    "selected_dataset", available_datasets[0])
if default_dataset not in available_datasets:
    default_dataset = available_datasets[0]

selected_dataset = st.sidebar.selectbox(
    "Активный датасет",
    options=available_datasets,
    index=available_datasets.index(default_dataset),
)
st.session_state["selected_dataset"] = selected_dataset

provider_default = get_provider_name()
provider_options = ["mock", "github", "cerebras"]
if provider_default not in provider_options:
    provider_default = "github"

requirements = get_dataset_requirements()
with st.sidebar.expander("Требования к датасету"):
    st.markdown("**Обязательные колонки:**")
    for col in requirements["required"]:
        st.markdown(f"- `{col}`")
    st.markdown("**Рекомендуемые колонки:**")
    for col in requirements["recommended"]:
        st.markdown(f"- `{col}`")

provider = st.sidebar.selectbox(
    "LLM провайдер",
    options=provider_options,
    index=provider_options.index(provider_default),
)

try:
    conn = init_database(selected_dataset)
except Exception as e:
    st.error(f"Ошибка инициализации базы: {e}")
    st.stop()

schema_df = conn.execute(
    f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{TABLE_NAME}'
    ORDER BY ordinal_position
    """
).fetchdf()

columns = schema_df["column_name"].tolist()
is_valid, missing_required, missing_recommended = validate_dataset_columns(
    columns)
if not is_valid:
    st.error(
        "Выбранный датасет не подходит: отсутствуют обязательные колонки "
        f"{', '.join(missing_required)}"
    )
    st.stop()

if missing_recommended:
    st.warning(
        "Часть метрик будет недоступна. Нет колонок: "
        f"{', '.join(missing_recommended)}"
    )

try:
    base_semantic_layer = load_semantic_layer()
    semantic_layer = filter_semantic_layer_by_columns(
        base_semantic_layer, columns)
except Exception as e:
    st.error(f"Ошибка загрузки semantic layer: {e}")
    st.stop()

if not semantic_layer.get("metrics"):
    st.error("После фильтрации семантического слоя не осталось доступных метрик.")
    st.stop()

row_count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
st.success(
    f"Активный датасет: `{Path(selected_dataset).name}` · строк: {row_count:,}"
)

with st.expander("Показать первые 5 строк"):
    preview_df = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5").fetchdf()
    st.dataframe(preview_df, use_container_width=True)

st.sidebar.header("Семантический слой")
st.sidebar.subheader("Метрики")
for metric_name, metric_def in semantic_layer.get("metrics", {}).items():
    st.sidebar.markdown(
        f"- **{metric_name}**: {metric_def.get('description', '')}")

st.sidebar.subheader("Измерения")
for dim_name, dim_def in semantic_layer.get("dimensions", {}).items():
    st.sidebar.markdown(f"- **{dim_name}** ({dim_def.get('field')})")

st.sidebar.subheader("Временные выражения")
for time_key in semantic_layer.get("time_expressions", {}).keys():
    st.sidebar.markdown(f"- {time_key}")

with st.expander("Анализ структуры данных (DESCRIBE / NULL / DISTINCT)"):
    describe_df = conn.execute(f"DESCRIBE {TABLE_NAME}").fetchdf()
    st.write("**DESCRIBE таблицы**")
    st.dataframe(describe_df, use_container_width=True)

    st.write("**information_schema.columns**")
    st.dataframe(schema_df, use_container_width=True)

    key_fields = [
        "order_timestamp",
        "status_order",
        "price_order_local",
        "duration_in_seconds",
        "distance_in_meters",
        "driver_id",
        "user_id",
        "offset_hours",
    ]
    existing_fields = [f for f in key_fields if f in columns]

    if existing_fields:
        union_parts = []
        for col in existing_fields:
            union_parts.append(
                f"""
                SELECT
                    '{col}' AS column_name,
                    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                    COUNT(DISTINCT {col}) AS distinct_count
                FROM {TABLE_NAME}
                """
            )
        profile_sql = " UNION ALL ".join(union_parts)
        profile_df = conn.execute(profile_sql).fetchdf()
        st.write("**NULL / DISTINCT по ключевым полям**")
        st.dataframe(profile_df, use_container_width=True)

    if "status_order" in columns:
        status_df = conn.execute(
            f"""
            SELECT status_order, COUNT(*) AS cnt
            FROM {TABLE_NAME}
            GROUP BY 1
            ORDER BY cnt DESC
            """
        ).fetchdf()
        st.write("**Возможные значения status_order**")
        st.dataframe(status_df, use_container_width=True)

st.subheader("Чат аналитики")

llm_client = None
llm_model = get_model_name(provider)
if provider != "mock":
    try:
        llm_client, llm_model = init_llm_runtime(provider)
        st.sidebar.success(f"LLM: {provider} / {llm_model}")
    except Exception as e:
        st.sidebar.error(f"LLM не инициализирован: {e}")

schema_info = format_schema_info(schema_df)
semantic_context = format_semantic_context(semantic_layer)
few_shot_examples = get_few_shot_examples()

init_chat_if_needed()
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("sql"):
            with st.expander("Сгенерированный SQL"):
                st.code(msg["sql"], language="sql")
        if isinstance(msg.get("data"), pd.DataFrame):
            st.dataframe(msg["data"], use_container_width=True)
            fig = build_auto_chart(msg["data"])
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True,
                                key=f"hist-fig-{idx}")

user_query = st.chat_input("Задайте вопрос по данным...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "text": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        generated_sql = ""
        route_used = ""

        metric_name, dimension_name = resolve_metric_and_dimension(
            semantic_layer, user_query)
        need_dimension = query_needs_dimension(user_query)
        can_use_semantic = metric_name is not None and (
            dimension_name is not None or not need_dimension)

        if can_use_semantic:
            route_used = "semantic"
            try:
                generated_sql = build_sql_from_semantics(
                    semantic_layer=semantic_layer,
                    metric_name=metric_name,
                    dimension_name=dimension_name,
                    user_text=user_query,
                )
                st.info("Семантический слой")
            except Exception:
                generated_sql = ""

        if not generated_sql:
            route_used = "llm"
            try:
                with st.spinner("Генерируем SQL через LLM..."):
                    generated_sql = generate_sql(
                        user_query=user_query,
                        schema_info=schema_info,
                        semantic_context=semantic_context,
                        examples=few_shot_examples,
                        provider=provider,
                        client=llm_client,
                        model=llm_model,
                    )
                st.info("Используем LLM")
            except (LLMClientError, Exception):
                msg = "Не удалось сгенерировать SQL. Попробуйте переформулировать."
                st.error(msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "text": msg})

        if generated_sql:
            with st.expander("Сгенерированный SQL", expanded=True):
                st.code(generated_sql, language="sql")

            is_safe, guardrail_result = validate_sql(generated_sql)
            if not is_safe:
                msg = f"Guardrails: {guardrail_result}"
                st.error(msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "text": msg, "sql": generated_sql})
            else:
                safe_sql = guardrail_result
                try:
                    result_df = run_query(conn, safe_sql)
                except Exception as e:
                    err = f"Ошибка выполнения SQL: {e}"
                    st.error(err)
                    st.info("Попробуйте уточнить вопрос.")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": err, "sql": safe_sql})
                else:
                    if result_df.empty:
                        st.warning("Нет данных")
                        st.session_state.chat_history.append(
                            {"role": "assistant", "text": "Нет данных", "sql": safe_sql}
                        )
                    else:
                        st.dataframe(result_df, use_container_width=True)
                        fig = build_auto_chart(result_df)
                        if fig is not None:
                            st.plotly_chart(
                                fig, use_container_width=True, key=f"live-fig-{len(st.session_state.chat_history)}")
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "text": f"Готово: SQL выполнен, результат построен ({route_used}).",
                                "sql": safe_sql,
                                "data": result_df,
                            }
                        )
