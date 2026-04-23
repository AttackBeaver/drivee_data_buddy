import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

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
def init_database():
    """Инициализация DuckDB и загрузка CSV в таблицу."""
    if duckdb is None:
        raise RuntimeError(
            "Модуль duckdb не установлен. Установите зависимости из requirements.txt")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Файл {CSV_PATH.as_posix()} не найден. Поместите incity_orders.csv в папку data/."
        )

    con = duckdb.connect(DB_PATH.as_posix())
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT *
        FROM read_csv_auto('{CSV_PATH.as_posix()}')
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
    """Простой резолвер для отладки: ищет метрику и измерение по фразе и синонимам."""
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


if duckdb is None:
    st.error(
        "Модуль `duckdb` не установлен. Установите зависимости: `pip install -r requirements.txt`.")
    st.stop()

if not CSV_PATH.exists():
    st.error(
        "Файл `data/incity_orders.csv` не найден. "
        "Пожалуйста, добавьте CSV в папку `data/` и перезапустите приложение."
    )
    st.stop()

if not SEMANTIC_PATH.exists():
    st.error(
        "Файл `semantic/semantic_layer.json` не найден. "
        "Создайте семантический слой и перезапустите приложение."
    )
    st.stop()

try:
    conn = init_database()
except Exception as e:
    st.error(f"Ошибка инициализации базы: {e}")
    st.stop()

row_count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
st.success(
    f"Таблица `{TABLE_NAME}` загружена. Количество строк: {row_count:,}")

with st.expander("Показать первые 5 строк"):
    preview_df = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5").fetchdf()
    st.dataframe(preview_df, use_container_width=True)

try:
    semantic_layer = load_semantic_layer()
except Exception as e:
    st.error(f"Ошибка загрузки semantic layer: {e}")
    st.stop()

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
    st.write("**DESCRIBE incity_orders**")
    st.dataframe(describe_df, use_container_width=True)

    schema_df = conn.execute(
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{TABLE_NAME}'
        ORDER BY ordinal_position
        """
    ).fetchdf()
    st.write("**information_schema.columns**")
    st.dataframe(schema_df, use_container_width=True)

    key_fields = [
        "city_id",
        "order_timestamp",
        "status_order",
        "price_order_local",
        "duration_in_seconds",
        "distance_in_meters",
        "driver_id",
        "user_id",
        "offset_hours",
    ]
    existing_fields = [
        f for f in key_fields if f in schema_df["column_name"].tolist()]

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

    if "status_order" in schema_df["column_name"].tolist():
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
st.caption(
    "Пример: 'покажи отмены по дням за прошлую неделю' или 'среднее расстояние по месяцам'")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "ai",
            "text": "Привет! Я готов показать SQL и результат по метрикам из семантического слоя.",
        }
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

user_text = st.chat_input("Введите аналитический запрос...")

if user_text:
    st.session_state.chat_history.append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("ai"):
        metric_name, dimension_name = resolve_metric_and_dimension(
            semantic_layer, user_text)

        if not metric_name:
            answer = "Не удалось определить метрику. Попробуйте: 'отмены по дням' или 'среднее расстояние по месяцам'."
            st.warning(answer)
            st.session_state.chat_history.append(
                {"role": "ai", "text": answer})
        else:
            try:
                sql = build_sql_from_semantics(
                    semantic_layer=semantic_layer,
                    metric_name=metric_name,
                    dimension_name=dimension_name,
                    user_text=user_text,
                )
                st.markdown(f"**Интерпретация:** метрика `{metric_name}`" + (
                    f", измерение `{dimension_name}`" if dimension_name else ""))
                st.code(sql, language="sql")
                demo_df = run_query(conn, sql)
                st.dataframe(demo_df, use_container_width=True)

                if not demo_df.empty and {"dimension", "value"}.issubset(demo_df.columns):
                    if dimension_name in {"день", "неделя"}:
                        fig_demo = px.line(
                            demo_df, x="dimension", y="value", markers=True, title="Динамика метрики")
                    elif dimension_name == "месяц":
                        fig_demo = px.bar(
                            demo_df, x="dimension", y="value", title="Метрика по месяцам")
                    else:
                        fig_demo = px.bar(
                            demo_df, x="dimension", y="value", title="Результат запроса")
                    st.plotly_chart(fig_demo, use_container_width=True)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "text": f"Готово. Нашёл метрику: **{metric_name}**" + (f" и измерение: **{dimension_name}**." if dimension_name else "."),
                    }
                )
            except Exception as e:
                err = f"Ошибка выполнения SQL: {e}"
                st.error(err)
                st.session_state.chat_history.append(
                    {"role": "assistant", "text": err})

if st.checkbox("Тестирование семантического слоя"):
    st.markdown("### 1) Количество отмен по дням за последнюю неделю")
    q1 = f"""
        SELECT
            DATE(order_timestamp) AS day,
            COUNT(*) AS cancels
        FROM {TABLE_NAME}
        WHERE status_order = 'cancel'
          AND DATE(order_timestamp) >= CURRENT_DATE - INTERVAL 7 DAY
        GROUP BY 1
        ORDER BY 1
    """
    st.code(q1, language="sql")
    df1 = run_query(conn, q1)
    st.dataframe(df1, use_container_width=True)
    if not df1.empty:
        fig1 = px.area(df1, x="day", y="cancels",
                       title="Отмены по дням (area chart)")
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 2) Распределение заказов по статусам")
    q2 = f"""
        SELECT
            status_order,
            COUNT(*) AS cnt
        FROM {TABLE_NAME}
        GROUP BY 1
        ORDER BY cnt DESC
    """
    st.code(q2, language="sql")
    df2 = run_query(conn, q2)
    st.dataframe(df2, use_container_width=True)
    if not df2.empty:
        fig2 = px.pie(df2, names="status_order", values="cnt",
                      title="Структура статусов заказов")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 3) Среднее расстояние по месяцам")
    q3 = f"""
        SELECT
            STRFTIME(order_timestamp, '%Y-%m') AS month,
            AVG(distance_in_meters) AS avg_distance_m
        FROM {TABLE_NAME}
        GROUP BY 1
        ORDER BY 1
    """
    st.code(q3, language="sql")
    df3 = run_query(conn, q3)
    st.dataframe(df3, use_container_width=True)
    if not df3.empty:
        fig3 = px.bar(df3, x="month", y="avg_distance_m",
                      title="Среднее расстояние по месяцам (bar chart)")
        st.plotly_chart(fig3, use_container_width=True)
