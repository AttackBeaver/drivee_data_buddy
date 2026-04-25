import re
from typing import Set, Tuple

FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
]


def _strip_sql_comments(sql: str) -> str:
    no_line_comments = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    no_block_comments = re.sub(r"/\*.*?\*/", "", no_line_comments, flags=re.DOTALL)
    return no_block_comments.strip()


def _append_limit_if_missing(sql: str, limit_value: int = 1000) -> str:
    clean = sql.rstrip().rstrip(";")
    if re.search(r"\bLIMIT\s+\d+\b", clean, flags=re.IGNORECASE):
        return clean
    return f"{clean} LIMIT {limit_value}"


def _extract_cte_names(sql: str) -> Set[str]:
    cte_names: Set[str] = set()
    if not re.match(r"^\s*WITH\b", sql, flags=re.IGNORECASE):
        return cte_names

    for name in re.findall(r"\b([a-zA-Z_][\w]*)\s+AS\s*\(", sql, flags=re.IGNORECASE):
        cte_names.add(name.lower())
    return cte_names


def validate_sql(
    sql: str,
    allowed_tables: Set[str] = None,
    required_table: str = None,
) -> Tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "SQL пустой"

    clean_sql = _strip_sql_comments(sql)
    upper_sql = clean_sql.upper()

    if any(keyword in upper_sql for keyword in FORBIDDEN_KEYWORDS):
        return False, "Обнаружены запрещённые SQL-операции"

    if ";" in clean_sql.rstrip(";"):
        return False, "Разрешён только один SQL-запрос"

    if not re.match(r"^\s*(SELECT|WITH)\b", clean_sql, flags=re.IGNORECASE):
        return False, "Разрешены только SELECT или WITH ... SELECT"

    table_refs = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][\w\.]*)", clean_sql, flags=re.IGNORECASE)
    if not table_refs:
        return False, "В запросе должна использоваться хотя бы одна таблица"

    cte_names = _extract_cte_names(clean_sql)
    normalized_refs = {ref.split(".")[-1].lower() for ref in table_refs}

    active_required = (required_table or "incity_orders").lower()
    allowed_refs = set(t.lower() for t in (allowed_tables or {active_required})) | cte_names
    illegal_refs = {ref for ref in normalized_refs if ref not in allowed_refs}
    if illegal_refs:
        allowed_list = ", ".join(sorted(allowed_refs))
        return False, f"Запрос содержит недопустимые таблицы. Разрешены: {allowed_list}"

    if active_required not in normalized_refs:
        return False, f"В запросе должна использоваться таблица {active_required}"

    safe_sql = _append_limit_if_missing(clean_sql)
    return True, safe_sql
