import copy
import hashlib
import json
from datetime import time
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
from intent_resolver import resolve_intent_with_confidence
from report_manager import (
    add_subscription,
    delete_report,
    load_reports,
    load_subscriptions,
    remove_subscription,
    save_report,
    simulate_email,
)
try:
    import duckdb
except ImportError:
    duckdb = None
st.set_page_config(page_title="Drivee Data Buddy",
                   page_icon="📊", layout="wide")

logo_left_path = Path("assets/logo_case_giver.png")
logo_right_path = Path("assets/logo_competition.png")

col1, col2, col3 = st.columns([3, 0.5, 0.5])
with col1:
    st.title("Drivee Data Buddy")
    st.caption(
        'Разработано командой "404: Имя не найдено", в рамках ГРАНД-ФИНАЛА проекта "Моя профессия ИТ 2025/26"')
with col2:
    if logo_right_path.exists():
        st.image(str(logo_right_path))
    else:
        st.caption("Лого конкурса")
with col3:
    if logo_left_path.exists():
        st.image(str(logo_left_path))
    else:
        st.caption("Лого кейсодателя")

DB_PATH = Path("data/drivee.duckdb")
SEMANTIC_PATH = Path("semantic/semantic_layer.json")
DEFAULT_TABLE = "incity"
TABLE_NAME = DEFAULT_TABLE

TABLE_SOURCE_CANDIDATES = {
    "incity": [Path("data/incity.csv")],
    "incity_orders": [Path("data/incity_orders.csv")],
    "pass_detail": [Path("data/pass_detail.csv")],
    "driver_detail": [Path("data/driver_detail.csv")],
}

def build_sources_signature() -> str:
    parts = []
    for table_name, candidates in TABLE_SOURCE_CANDIDATES.items():
        for source in candidates:
            if source.exists():
                stat = source.stat()
                parts.append(f"{table_name}:{source.as_posix()}:{stat.st_mtime_ns}:{stat.st_size}")
                break
    return "|".join(parts)


@st.cache_resource
def init_database(sources_signature: str):
    """Инициализация DuckDB и загрузка доступных CSV в таблицы."""
    _ = sources_signature

    if duckdb is None:
        raise RuntimeError(
            "Модуль duckdb не установлен. Установите зависимости из requirements.txt")

    available_sources = {}
    for table_name, candidates in TABLE_SOURCE_CANDIDATES.items():
        for source in candidates:
            if source.exists():
                available_sources[table_name] = source
                break

    if not available_sources:
        raise FileNotFoundError(
            "Не найдено ни одного датасета в data/. Ожидались incity.csv, incity_orders.csv, pass_detail.csv, driver_detail.csv"
        )

    con = duckdb.connect(DB_PATH.as_posix())
    for table_name, source in available_sources.items():
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{source.as_posix()}', sample_size=-1)
            """
        )

    return con, {name: path.as_posix() for name, path in available_sources.items()}


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


def get_dataset_requirements(table_name: str) -> Dict[str, List[str]]:
    requirements_by_table = {
        "incity": {
            "required": ["order_timestamp", "status_order"],
            "recommended": [
                "price_order_local",
                "distance_in_meters",
                "duration_in_seconds",
                "user_id",
                "driver_id",
                "offset_hours",
            ],
        },
        "incity_orders": {
            "required": ["order_timestamp", "status_order"],
            "recommended": [
                "price_order_local",
                "distance_in_meters",
                "duration_in_seconds",
                "user_id",
                "driver_id",
                "offset_hours",
            ],
        },        "pass_detail": {
            "required": ["order_date_part", "user_id"],
            "recommended": [
                "orders_count",
                "orders_cnt_with_tenders",
                "orders_cnt_accepted",
                "rides_count",
                "rides_time_sum_seconds",
                "online_time_sum_seconds",
            ],
        },
        "driver_detail": {
            "required": ["tender_date_part", "driver_id"],
            "recommended": [
                "orders",
                "orders_cnt_with_tenders",
                "orders_cnt_accepted",
                "rides_count",
                "rides_time_sum_seconds",
                "online_time_sum_seconds",
            ],
        },
    }
    return requirements_by_table.get(table_name, {"required": [], "recommended": []})


def validate_dataset_columns(columns: List[str], table_name: str) -> Tuple[bool, List[str], List[str]]:
    reqs = get_dataset_requirements(table_name)
    required = reqs["required"]
    recommended = reqs["recommended"]

    missing_required = [c for c in required if c not in columns]
    missing_recommended = [c for c in recommended if c not in columns]
    return len(missing_required) == 0, missing_required, missing_recommended


def detect_time_column(columns: List[str]) -> Optional[str]:
    priority = ["order_timestamp", "tender_timestamp", "order_date_part", "tender_date_part"]
    for name in priority:
        if name in columns:
            return name
    return None


def adapt_semantic_layer_to_time_column(semantic_layer: dict, time_column: Optional[str]) -> dict:
    updated = copy.deepcopy(semantic_layer)
    if not time_column:
        return updated

    dimensions = updated.setdefault("dimensions", {})
    dimensions["день"] = {
        "field": f"DATE(CAST({time_column} AS TIMESTAMP))",
        "type": "time",
        "description": f"Дата по полю {time_column}",
        "depends_on": [time_column],
    }
    dimensions["неделя"] = {
        "field": f"STRFTIME(CAST({time_column} AS TIMESTAMP), '%Y-%W')",
        "type": "time",
        "description": f"Год-неделя по полю {time_column}",
        "depends_on": [time_column],
    }
    dimensions["месяц"] = {
        "field": f"STRFTIME(CAST({time_column} AS TIMESTAMP), '%Y-%m')",
        "type": "time",
        "description": f"Год-месяц по полю {time_column}",
        "depends_on": [time_column],
    }
    if time_column.endswith("timestamp"):
        dimensions["час"] = {
            "field": f"STRFTIME(CAST({time_column} AS TIMESTAMP), '%Y-%m-%d %H:00')",
            "type": "time",
            "description": f"Час по полю {time_column}",
            "depends_on": [time_column],
        }

    return updated


def build_fallback_semantic_layer(table_name: str, columns: List[str], time_column: Optional[str]) -> dict:
    if table_name == "pass_detail":
        count_metric_sql = "SUM(orders_count)"
        entity_field = "user_id"
    else:
        count_metric_sql = "SUM(orders)" if "orders" in columns else "COUNT(*)"
        entity_field = "driver_id"

    metrics = {
        "количество заказов": {
            "sql": count_metric_sql,
            "description": "Количество заказов",
            "depends_on": ["orders_count"] if table_name == "pass_detail" else (["orders"] if "orders" in columns else []),
            "aliases": ["заказы", "число заказов"],
        },
        "количество поездок": {
            "sql": "SUM(rides_count)",
            "description": "Количество завершённых поездок",
            "depends_on": ["rides_count"],
            "aliases": ["поездки", "райды"],
        },
        "время поездок": {
            "sql": "SUM(rides_time_sum_seconds)",
            "description": "Суммарное время поездок в секундах",
            "depends_on": ["rides_time_sum_seconds"],
            "aliases": ["длительность поездок"],
        },
        "время онлайн": {
            "sql": "SUM(online_time_sum_seconds)",
            "description": "Суммарное время онлайн в секундах",
            "depends_on": ["online_time_sum_seconds"],
            "aliases": ["онлайн время", "онлайн"],
        },
        "количество пользователей": {
            "sql": f"COUNT(DISTINCT {entity_field})",
            "description": "Уникальные пользователи",
            "depends_on": [entity_field],
            "aliases": ["пользователи", "уники"],
        },
    }

    dimensions = {
        "город": {
            "field": "city_id",
            "type": "categorical",
            "description": "Идентификатор города",
            "depends_on": ["city_id"],
        },
        "пользователь": {
            "field": entity_field,
            "type": "categorical",
            "description": "Идентификатор пользователя",
            "depends_on": [entity_field],
        },
    }

    if time_column:
        dimensions.update(
            {
                "день": {
                    "field": f"DATE(CAST({time_column} AS TIMESTAMP))",
                    "type": "time",
                    "description": f"Дата по полю {time_column}",
                    "depends_on": [time_column],
                },
                "неделя": {
                    "field": f"STRFTIME(CAST({time_column} AS TIMESTAMP), '%Y-%W')",
                    "type": "time",
                    "description": f"Неделя по полю {time_column}",
                    "depends_on": [time_column],
                },
                "месяц": {
                    "field": f"STRFTIME(CAST({time_column} AS TIMESTAMP), '%Y-%m')",
                    "type": "time",
                    "description": f"Месяц по полю {time_column}",
                    "depends_on": [time_column],
                },
            }
        )

    return {
        "metrics": metrics,
        "dimensions": dimensions,
        "time_expressions": {
            "вчера": "CURRENT_DATE - INTERVAL 1 DAY",
            "сегодня": "CURRENT_DATE",
            "последние 7 дней": "CURRENT_DATE - INTERVAL 7 DAY",
            "последние 30 дней": "CURRENT_DATE - INTERVAL 30 DAY",
        },
        "filters": {},
        "synonyms": {
            "заказы": "количество заказов",
            "поездки": "количество поездок",
            "онлайн": "время онлайн",
            "по дням": "день",
            "по неделям": "неделя",
            "по месяцам": "месяц",
        },
    }


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

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "resolved_intent" not in st.session_state:
        st.session_state.resolved_intent = None
    if "intent_edit_mode" not in st.session_state:
        st.session_state.intent_edit_mode = False


def init_report_state_if_needed() -> None:
    if "saved_reports" not in st.session_state:
        st.session_state.saved_reports = load_reports()
    if "report_subscriptions" not in st.session_state:
        st.session_state.report_subscriptions = load_subscriptions()
    if "email_preview" not in st.session_state:
        st.session_state.email_preview = None
    if "report_open_feedback" not in st.session_state:
        st.session_state.report_open_feedback = None


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
    table_name: str,
    time_column: Optional[str],
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
    if time_expr and time_column:
        where_clauses.append(f"DATE(CAST({time_column} AS TIMESTAMP)) >= {time_expr}")

    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    return (
        f"SELECT {', '.join(select_parts)} "
        f"FROM {table_name}"
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


def get_few_shot_examples(
    table_name: str,
    time_column: Optional[str],
    status_column: Optional[str],
) -> str:
    date_expr = f"DATE(CAST({time_column} AS TIMESTAMP))" if time_column else "CURRENT_DATE"

    if status_column:
        return f"""
Вопрос: Сколько всего записей со статусом done за вчера?
SQL:
SELECT COUNT(*) AS done_items
FROM {table_name}
WHERE {status_column} = 'done'
  AND {date_expr} = CURRENT_DATE - INTERVAL 1 DAY
LIMIT 1000;

Вопрос: Покажи записи по дням за последние 7 дней
SQL:
SELECT {date_expr} AS day, COUNT(*) AS items
FROM {table_name}
WHERE {date_expr} >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY 1
ORDER BY 1
LIMIT 1000;
""".strip()

    return f"""
Вопрос: Сколько записей за вчера?
SQL:
SELECT COUNT(*) AS items
FROM {table_name}
WHERE {date_expr} = CURRENT_DATE - INTERVAL 1 DAY
LIMIT 1000;

Вопрос: Покажи динамику по дням за последние 7 дней
SQL:
SELECT {date_expr} AS day, COUNT(*) AS items
FROM {table_name}
WHERE {date_expr} >= CURRENT_DATE - INTERVAL 7 DAY
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


def generate_report_html(sql: str, df: pd.DataFrame, fig=None) -> str:
    """Создаёт HTML-отчёт с таблицей, SQL-запросом и интерактивным графиком Plotly."""
    table_html = df.to_html(classes="dataframe", border=0, index=False)

    chart_html = ""
    if fig is not None:
        # Встраиваем Plotly-график как интерактивный HTML
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Drivee Data Buddy – Отчёт</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #2c3e50; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .dataframe {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            .dataframe th {{ background-color: #3498db; color: white; padding: 8px; text-align: left; }}
            .dataframe td {{ border: 1px solid #ddd; padding: 8px; }}
            .dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h2>Отчёт по запросу</h2>
        <h3>SQL‑запрос</h3>
        <pre>{sql}</pre>
        <h3>Результат (таблица)</h3>
        {table_html}
        <h3>Визуализация</h3>
        {chart_html if chart_html else "<p>График не доступен.</p>"}
    </body>
    </html>
    """
    return html


def render_reports_sidebar(conn) -> None:
    st.sidebar.header("Отчёты")
    st.sidebar.subheader("Мои отчёты")

    reports = st.session_state.get("saved_reports", [])
    if not reports:
        st.sidebar.caption("Пока нет сохранённых отчётов")
    else:
        for idx, report in enumerate(reports):
            report_id = report.get("id", f"report-{idx}")
            report_name = report.get("name", f"Отчёт {idx + 1}")
            report_sql = report.get("sql", "")

            col_open, col_del = st.sidebar.columns([4, 1])
            if col_open.button(report_name, key=f"saved-open-{report_id}"):
                try:
                    report_df = run_query(conn, report_sql)
                    st.session_state.chat_history.append(
                        {"role": "user", "text": f"Открыть сохранённый отчёт: {report_name}"}
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "text": f"Открыт сохранённый отчёт: {report_name}",
                            "sql": report_sql,
                            "data": report_df,
                        }
                    )
                    st.session_state.report_open_feedback = f"Открыт отчёт: {report_name}"
                except Exception as e:
                    st.session_state.report_open_feedback = f"Не удалось открыть отчёт: {e}"

            if col_del.button("🗑", key=f"saved-del-{report_id}"):
                st.session_state.saved_reports = delete_report(
                    st.session_state.saved_reports, report_id
                )
                st.session_state.report_subscriptions = [
                    s for s in st.session_state.report_subscriptions if s.get("report_id") != report_id
                ]
                st.sidebar.success("Отчёт удалён")
                st.rerun()

            with st.sidebar.expander(f"Настроить рассылку · {report_name}"):
                with st.form(key=f"sub-form-{report_id}"):
                    email = st.text_input("Email", key=f"sub-email-{report_id}")
                    frequency = st.selectbox(
                        "Периодичность",
                        options=["Ежедневно", "Еженедельно", "Ежемесячно"],
                        index=1,
                        key=f"sub-freq-{report_id}",
                    )
                    send_time = st.time_input(
                        "Время отправки",
                        value=time(hour=9, minute=0),
                        key=f"sub-time-{report_id}",
                    )
                    subscribe = st.form_submit_button("Подписаться")

                if subscribe:
                    if not email or "@" not in email:
                        st.warning("Введите корректный email")
                    else:
                        send_time_str = send_time.strftime("%H:%M")
                        st.session_state.report_subscriptions = add_subscription(
                            subscriptions=st.session_state.report_subscriptions,
                            report_id=report_id,
                            report_name=report_name,
                            email=email,
                            frequency=frequency,
                            send_time=send_time_str,
                        )

                        try:
                            report_df = run_query(conn, report_sql)
                        except Exception:
                            preview_rows = report.get("preview", [])
                            preview_cols = report.get("columns", [])
                            report_df = pd.DataFrame(preview_rows)
                            if preview_cols and not report_df.empty:
                                report_df = report_df.reindex(columns=preview_cols)

                        report_fig = build_auto_chart(report_df) if not report_df.empty else None
                        report_html = generate_report_html(report_sql, report_df, report_fig)

                        email_result = simulate_email(
                            report=report,
                            email=email,
                            schedule={"frequency": frequency, "send_time": send_time_str},
                            email_html=report_html,
                        )
                        st.session_state.email_preview = email_result
                        st.success(email_result["message"])

    feedback = st.session_state.get("report_open_feedback")
    if feedback:
        st.sidebar.info(feedback)

    st.sidebar.subheader("Подписки")
    subscriptions = st.session_state.get("report_subscriptions", [])
    if not subscriptions:
        st.sidebar.caption("Активных подписок нет")
    else:
        for sub in subscriptions:
            sub_id = sub.get("id", "")
            txt = (
                f"**{sub.get('report_name', 'Отчёт')}**\n"
                f"{sub.get('email', '-')} · {sub.get('frequency', '-')} · {sub.get('send_time', '-')}"
            )
            st.sidebar.markdown(txt)
            if st.sidebar.button("Отписаться", key=f"unsub-{sub_id}"):
                st.session_state.report_subscriptions = remove_subscription(
                    st.session_state.report_subscriptions, sub_id
                )
                st.sidebar.success("Подписка удалена")
                st.rerun()

    preview = st.session_state.get("email_preview")
    if isinstance(preview, dict):
        with st.sidebar.expander("Превью тестового письма"):
            st.markdown(f"**Тема:** {preview.get('subject', '-')}")
            st.markdown(preview.get("html", ""), unsafe_allow_html=True)


def execute_template(
    template,
    metric_name,
    dimension_name,
    semantic_layer,
    conn,
    table_name: str,
    time_column: Optional[str],
    allowed_tables: List[str],
):
    """Выполняет SQL по шаблону сценария, используя семантический слой."""
    try:
        sql = build_sql_from_semantics(
            semantic_layer=semantic_layer,
            metric_name=metric_name,
            table_name=table_name,
            time_column=time_column,
            dimension_name=dimension_name,
            user_text="",
        )
        is_safe, safe_sql = validate_sql(
            sql,
            allowed_tables=set(allowed_tables),
            required_table=table_name,
        )
        if not is_safe:
            msg = f"Guardrails: {safe_sql}"
            st.error(msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "text": msg, "sql": sql})
            return
        result_df = run_query(conn, safe_sql)
        if result_df.empty:
            st.warning("Нет данных")
            st.session_state.chat_history.append(
                {"role": "assistant", "text": "Нет данных", "sql": safe_sql})
            return
        user_msg = f"Сценарий: {metric_name} по {dimension_name}" if dimension_name else f"Сценарий: {metric_name}"
        st.session_state.chat_history.append(
            {"role": "user", "text": user_msg})
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": "Готово: SQL выполнен, результат построен.",
            "sql": safe_sql,
            "data": result_df,
        })
    except Exception as e:
        error_msg = f"Ошибка выполнения сценария: {e}"
        st.error(error_msg)
        st.session_state.chat_history.append(
            {"role": "assistant", "text": error_msg})


def find_matching_template(templates: List[dict], metric_name: str, dimension_name: Optional[str]) -> Optional[dict]:
    for tmpl in templates:
        if tmpl.get("metric") != metric_name:
            continue

        single_dim = tmpl.get("dimension")
        dims_in = tmpl.get("dimension_in", [])

        if dimension_name is None:
            if not single_dim and not dims_in:
                return tmpl
            continue

        if single_dim == dimension_name:
            return tmpl
        if isinstance(dims_in, list) and dimension_name in dims_in:
            return tmpl

    return None


def render_intent_explainability(intent: dict) -> None:
    confidence = float(intent.get("confidence", 0.0))
    metric_label = intent.get("metric") or "не определена"
    dimension_label = intent.get("dimension") or "без разреза"
    period_label = intent.get("time_range") or "не указан"
    filters = intent.get("filters") or []

    msg = (
        "Я понял ваш запрос как:\n"
        f"- Метрика: {metric_label}\n"
        f"- Разрез: {dimension_label}\n"
        f"- Период: {period_label}\n"
        f"- Фильтры: {', '.join(filters) if filters else 'нет'}\n"
        f"- Уверенность: {confidence * 100:.0f}%"
    )

    if confidence >= 0.7:
        st.success(msg)
    elif confidence >= 0.4:
        st.warning(msg)
    else:
        st.error(msg)


def execute_query_pipeline(
    user_query: str,
    intent: dict,
    semantic_layer: dict,
    templates: List[dict],
    schema_info: str,
    semantic_context: str,
    few_shot_examples: str,
    provider: str,
    llm_client,
    llm_model: str,
    conn,
    table_name: str,
    time_column: Optional[str],
    status_column: Optional[str],
    allowed_tables: List[str],
) -> None:
    generated_sql = ""
    route_used = ""

    metric_name = intent.get("metric")
    dimension_name = intent.get("dimension")
    confidence = float(intent.get("confidence", 0.0))
    force_llm = bool(intent.get("force_llm", False))
    need_dimension = query_needs_dimension(user_query)

    if not force_llm and metric_name is not None and (dimension_name is not None or not need_dimension):
        matched_template = None
        if confidence >= 0.7:
            matched_template = find_matching_template(
                templates=templates,
                metric_name=metric_name,
                dimension_name=dimension_name,
            )

        try:
            generated_sql = build_sql_from_semantics(
                semantic_layer=semantic_layer,
                metric_name=metric_name,
                table_name=table_name,
                time_column=time_column,
                dimension_name=dimension_name,
                user_text=user_query,
            )
            if matched_template:
                route_used = "template"
                st.info("Используем шаблон/семантический слой")
            else:
                route_used = "semantic"
                st.info("Используем семантический слой")
        except Exception:
            generated_sql = ""

    if not generated_sql:
        route_used = "llm"
        try:
            if confidence < 0.7 or force_llm:
                st.warning("Низкая уверенность интерпретации: используем fallback через LLM.")
            with st.spinner("Генерируем SQL через LLM..."):
                generated_sql = generate_sql(
                    user_query=user_query,
                    schema_info=schema_info,
                    semantic_context=semantic_context,
                    examples=few_shot_examples,
                    provider=provider,
                    client=llm_client,
                    model=llm_model,
                    table_name=table_name,
                    time_column=time_column,
                    status_column=status_column,
                )
            st.info("Используем LLM")
        except (LLMClientError, Exception):
            msg = "Не удалось сгенерировать SQL. Попробуйте переформулировать."
            st.error(msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "text": msg})
            return

    with st.expander("Сгенерированный SQL"):
        st.code(generated_sql, language="sql")

    is_safe, guardrail_result = validate_sql(
        generated_sql,
        allowed_tables=set(allowed_tables),
        required_table=table_name,
    )
    if not is_safe:
        msg = f"Guardrails: {guardrail_result}"
        st.error(msg)
        st.session_state.chat_history.append(
            {"role": "assistant", "text": msg, "sql": generated_sql})
        return

    safe_sql = guardrail_result
    try:
        result_df = run_query(conn, safe_sql)
    except Exception as e:
        err = f"Ошибка выполнения SQL: {e}"
        st.error(err)
        st.info("Попробуйте уточнить вопрос.")
        st.session_state.chat_history.append(
            {"role": "assistant", "text": err, "sql": safe_sql})
        return

    if result_df.empty:
        st.warning("Нет данных")
        st.session_state.chat_history.append(
            {"role": "assistant", "text": "Нет данных", "sql": safe_sql}
        )
        return

    st.dataframe(result_df, use_container_width=True)
    fig = build_auto_chart(result_df)
    if fig is not None:
        st.plotly_chart(
            fig, use_container_width=True, key=f"live-fig-{len(st.session_state.chat_history)}")

    report_html = None
    try:
        report_html = generate_report_html(
            safe_sql, result_df, fig)
        st.download_button(
            label="Скачать отчёт",
            data=report_html,
            file_name="drivee_report.html",
            mime="text/html",
            key=f"dl-{len(st.session_state.chat_history)}"
        )
    except Exception as e:
        st.warning(f"Не удалось создать отчёт: {e}")

    with st.form(key=f"save-report-form-{len(st.session_state.chat_history)}"):
        report_name = st.text_input("Название отчёта", value=f"Отчёт {route_used}")
        save_clicked = st.form_submit_button("💾 Сохранить отчёт")

    if save_clicked:
        chart_type = fig.__class__.__name__ if fig is not None else "none"
        st.session_state.saved_reports = save_report(
            reports=st.session_state.get("saved_reports", []),
            name=report_name,
            query=user_query,
            sql=safe_sql,
            data_df=result_df,
            chart_type=chart_type,
            config={"route_used": route_used},
        )
        st.success("Отчёт сохранён в разделе «Мои отчёты»")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "text": f"Готово: SQL выполнен, результат построен ({route_used}).",
            "sql": safe_sql,
            "data": result_df,
        }
    )


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

logo_team_path = Path("assets/logo_team.png")
if logo_team_path.exists():
    st.sidebar.image(str(logo_team_path))
else:
    st.sidebar.caption("Логотип команды не найден")

st.sidebar.header("Источник данных")
st.sidebar.caption("Работаем с таблицами из папки data (большие датасеты загружаются локально в DuckDB).")

uploaded_file = st.sidebar.file_uploader(
    "Загрузить дополнительный CSV (небольшой файл)",
    type=["csv"],
    help="Для больших файлов (>100MB) рекомендуется класть CSV сразу в папку data.",
)
if uploaded_file is not None:
    uploaded_path = save_uploaded_csv(uploaded_file)
    st.sidebar.success(f"Файл сохранён: `{uploaded_path.name}`")

provider_default = get_provider_name()
provider_options = ["mock", "github", "deepseek"]
if provider_default not in provider_options:
    provider_default = "github"

provider = st.sidebar.selectbox(
    "LLM провайдер",
    options=provider_options,
    index=provider_options.index(provider_default),
)

llm_client = None
llm_model = get_model_name(provider)
if provider != "mock":
    try:
        llm_client, llm_model = init_llm_runtime(provider)
        st.sidebar.success(f"LLM: {provider} / {llm_model}")
    except Exception as e:
        st.sidebar.error(f"LLM не инициализирован: {e}")
else:
    st.sidebar.info("Mock-режим (без LLM)")

sources_signature = build_sources_signature()
try:
    conn, available_sources = init_database(sources_signature)
except Exception as e:
    st.error(f"Ошибка инициализации базы: {e}")
    st.stop()

available_tables = sorted(available_sources.keys())
if not available_tables:
    st.error("Не найдены доступные таблицы в data/")
    st.stop()

default_table = st.session_state.get("selected_table", DEFAULT_TABLE)
if default_table not in available_tables:
    default_table = available_tables[0]

selected_table = st.sidebar.selectbox(
    "Активная таблица",
    options=available_tables,
    index=available_tables.index(default_table),
    format_func=lambda t: f"{t} ({Path(available_sources[t]).name})",
)
st.session_state["selected_table"] = selected_table

time_column = detect_time_column(
    conn.execute(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{selected_table}'
        ORDER BY ordinal_position
        """
    ).fetchdf()["column_name"].tolist()
)

schema_df = conn.execute(
    f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{selected_table}'
    ORDER BY ordinal_position
    """
).fetchdf()

columns = schema_df["column_name"].tolist()
status_column = "status_order" if "status_order" in columns else ("status_tender" if "status_tender" in columns else None)

requirements = get_dataset_requirements(selected_table)
with st.sidebar.expander("Требования к выбранной таблице"):
    st.markdown("**Обязательные колонки:**")
    for col in requirements["required"]:
        st.markdown(f"- `{col}`")
    st.markdown("**Рекомендуемые колонки:**")
    for col in requirements["recommended"]:
        st.markdown(f"- `{col}`")

is_valid, missing_required, missing_recommended = validate_dataset_columns(
    columns, selected_table)
if not is_valid:
    st.error(
        "Выбранная таблица не подходит: отсутствуют обязательные колонки "
        f"{', '.join(missing_required)}"
    )
    st.stop()

if missing_recommended:
    st.warning(
        "Часть метрик будет недоступна. Нет колонок: "
        f"{', '.join(missing_recommended)}"
    )

try:
    if selected_table in {"incity", "incity_orders"}:
        base_semantic_layer = load_semantic_layer()
        base_semantic_layer = adapt_semantic_layer_to_time_column(base_semantic_layer, time_column)
        semantic_layer = filter_semantic_layer_by_columns(base_semantic_layer, columns)
    else:
        semantic_layer = build_fallback_semantic_layer(selected_table, columns, time_column)
        semantic_layer = filter_semantic_layer_by_columns(semantic_layer, columns)
except Exception as e:
    st.error(f"Ошибка загрузки semantic layer: {e}")
    st.stop()

if not semantic_layer.get("metrics"):
    st.error("После фильтрации семантического слоя не осталось доступных метрик.")
    st.stop()

init_chat_if_needed()
init_report_state_if_needed()
render_reports_sidebar(conn)

if st.sidebar.button("Очистить чат"):
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

templates: List[dict] = []
templates_path = Path("semantic/templates.json")
if templates_path.exists():
    with templates_path.open("r", encoding="utf-8") as f:
        templates_data = json.load(f)
    templates = templates_data.get("templates", [])
    if templates:
        st.sidebar.subheader("Готовые сценарии")
        for tmpl in templates:
            metric = tmpl["metric"]
            dimension = tmpl.get("dimension")
            dimensions_list = tmpl.get("dimension_in", [])
            if dimensions_list:
                for dim in dimensions_list:
                    label = f"{metric} по {dim}"
                    if st.sidebar.button(label):
                        execute_template(
                            tmpl,
                            metric,
                            dim,
                            semantic_layer,
                            conn,
                            table_name=selected_table,
                            time_column=time_column,
                            allowed_tables=available_tables,
                        )
            else:
                if dimension:
                    label = f"{metric} по {dimension}"
                else:
                    label = metric
                if st.sidebar.button(label):
                    execute_template(
                        tmpl,
                        metric,
                        dimension,
                        semantic_layer,
                        conn,
                        table_name=selected_table,
                        time_column=time_column,
                        allowed_tables=available_tables,
                    )

row_count = conn.execute(f"SELECT COUNT(*) FROM {selected_table}").fetchone()[0]
st.success(
    f"Активная таблица: `{selected_table}` ({Path(available_sources[selected_table]).name}) · строк: {row_count:,}"
)

with st.expander("Показать первые 5 строк"):
    preview_df = conn.execute(f"SELECT * FROM {selected_table} LIMIT 5").fetchdf()
    st.dataframe(preview_df, use_container_width=True)

st.sidebar.header("Семантический слой")
st.sidebar.subheader("Метрики")
metrics_names = list(semantic_layer.get("metrics", {}).keys())
if metrics_names:
    for name in metrics_names:
        st.sidebar.markdown(f"- {name}")
else:
    st.sidebar.markdown("—")

st.sidebar.subheader("Измерения")
dims_names = list(semantic_layer.get("dimensions", {}).keys())
if dims_names:
    for name in dims_names:
        st.sidebar.markdown(f"- {name}")
else:
    st.sidebar.markdown("—")

st.sidebar.subheader("Временные выражения")
time_expr_names = list(semantic_layer.get("time_expressions", {}).keys())
if time_expr_names:
    for name in time_expr_names:
        st.sidebar.markdown(f"- {name}")
else:
    st.sidebar.markdown("—")

with st.expander("Анализ структуры данных (DESCRIBE / NULL / DISTINCT)"):
    describe_df = conn.execute(f"DESCRIBE {selected_table}").fetchdf()
    st.write("**DESCRIBE таблицы**")
    st.dataframe(describe_df, use_container_width=True)

    st.write("**information_schema.columns**")
    st.dataframe(schema_df, use_container_width=True)

    key_fields = [
        "order_timestamp",
        "tender_timestamp",
        "order_date_part",
        "tender_date_part",
        "status_order",
        "status_tender",
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
                FROM {selected_table}
                """
            )
        profile_sql = " UNION ALL ".join(union_parts)
        profile_df = conn.execute(profile_sql).fetchdf()
        st.write("**NULL / DISTINCT по ключевым полям**")
        st.dataframe(profile_df, use_container_width=True)

    if status_column:
        status_df = conn.execute(
            f"""
            SELECT {status_column} AS status_value, COUNT(*) AS cnt
            FROM {selected_table}
            GROUP BY 1
            ORDER BY cnt DESC
            """
        ).fetchdf()
        st.write(f"**Возможные значения {status_column}**")
        st.dataframe(status_df, use_container_width=True)


st.subheader("Чат аналитики")

schema_info = format_schema_info(schema_df)
semantic_context = format_semantic_context(semantic_layer)
few_shot_examples = get_few_shot_examples(
    table_name=selected_table,
    time_column=time_column,
    status_column=status_column,
)

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

    st.session_state.pending_query = user_query
    st.session_state.resolved_intent = resolve_intent_with_confidence(
        user_query=user_query,
        semantic_layer=semantic_layer,
    )
    st.session_state.intent_edit_mode = False

pending_query = st.session_state.get("pending_query")
intent = st.session_state.get("resolved_intent")

if pending_query and isinstance(intent, dict):
    request_key = hashlib.md5(pending_query.encode("utf-8")).hexdigest()[:8]

    with st.chat_message("assistant"):
        render_intent_explainability(intent)

        if st.button("Изменить", key=f"intent-edit-{request_key}"):
            st.session_state.intent_edit_mode = True

        ambiguities = intent.get("ambiguities") or []
        if ambiguities:
            st.write("Возможные варианты:")
            for idx, amb in enumerate(ambiguities):
                label = amb.get("description") or amb.get("value") or "Вариант"
                if st.button(label, key=f"intent-amb-{request_key}-{idx}"):
                    kind = amb.get("kind")
                    value = amb.get("value")
                    if kind in {"metric", "dimension"}:
                        intent[kind] = value
                        intent["confidence"] = 1.0
                        intent["ambiguities"] = []
                        st.session_state.resolved_intent = intent
                        st.session_state.intent_edit_mode = False

        confidence = float(intent.get("confidence", 0.0))
        show_manual = st.session_state.intent_edit_mode or confidence < 0.7

        if show_manual:
            metric_options = list(semantic_layer.get("metrics", {}).keys())
            dimension_options = ["(без разреза)"] + list(
                semantic_layer.get("dimensions", {}).keys())

            current_metric = intent.get("metric")
            current_dimension = intent.get("dimension")

            metric_index = metric_options.index(
                current_metric) if current_metric in metric_options else 0
            dimension_current_value = current_dimension if current_dimension else "(без разреза)"
            dimension_index = dimension_options.index(
                dimension_current_value) if dimension_current_value in dimension_options else 0

            selected_metric = st.selectbox(
                "Уточните метрику",
                options=metric_options,
                index=metric_index,
                key=f"intent-metric-{request_key}",
            )
            selected_dimension_raw = st.selectbox(
                "Уточните разрез",
                options=dimension_options,
                index=dimension_index,
                key=f"intent-dimension-{request_key}",
            )
            selected_dimension = None if selected_dimension_raw == "(без разреза)" else selected_dimension_raw

            if selected_metric != current_metric or selected_dimension != current_dimension:
                intent["metric"] = selected_metric
                intent["dimension"] = selected_dimension
                intent["confidence"] = 1.0
                intent["ambiguities"] = []
                confidence = 1.0
                st.session_state.resolved_intent = intent

        run_with_intent = False
        force_llm = False
        confidence = float(intent.get("confidence", 0.0))

        if confidence >= 0.7 and not st.session_state.intent_edit_mode:
            run_with_intent = True
        elif confidence >= 0.4:
            st.warning("Подтвердите интерпретацию или уточните метрику/разрез.")
            run_with_intent = st.button(
                "Подтвердить и получить данные",
                key=f"intent-confirm-{request_key}",
            )
        else:
            st.error("Низкая уверенность: выберите метрику и разрез вручную или используйте LLM fallback.")
            run_with_intent = st.button(
                "Получить данные",
                key=f"intent-run-{request_key}",
            )
            force_llm = st.button(
                "Попробовать через LLM без уточнений",
                key=f"intent-force-llm-{request_key}",
            )

        if force_llm:
            intent["force_llm"] = True
            st.session_state.resolved_intent = intent
            execute_query_pipeline(
                user_query=pending_query,
                intent=intent,
                semantic_layer=semantic_layer,
                templates=templates,
                schema_info=schema_info,
                semantic_context=semantic_context,
                few_shot_examples=few_shot_examples,
                provider=provider,
                llm_client=llm_client,
                llm_model=llm_model,
                conn=conn,
                table_name=selected_table,
                time_column=time_column,
                status_column=status_column,
                allowed_tables=available_tables,
            )
            st.session_state.pending_query = None
            st.session_state.resolved_intent = None
            st.session_state.intent_edit_mode = False
        elif run_with_intent:
            intent["force_llm"] = False
            st.session_state.resolved_intent = intent
            execute_query_pipeline(
                user_query=pending_query,
                intent=intent,
                semantic_layer=semantic_layer,
                templates=templates,
                schema_info=schema_info,
                semantic_context=semantic_context,
                few_shot_examples=few_shot_examples,
                provider=provider,
                llm_client=llm_client,
                llm_model=llm_model,
                conn=conn,
                table_name=selected_table,
                time_column=time_column,
                status_column=status_column,
                allowed_tables=available_tables,
            )
            st.session_state.pending_query = None
            st.session_state.resolved_intent = None
            st.session_state.intent_edit_mode = False
