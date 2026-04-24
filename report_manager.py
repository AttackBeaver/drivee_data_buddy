import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA_DIR = Path("data")
SAVED_REPORTS_PATH = DATA_DIR / "saved_reports.json"
SUBSCRIPTIONS_PATH = DATA_DIR / "subscriptions.json"


def _read_json(path: Path, default_value):
    if not path.exists():
        return default_value
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_reports() -> List[dict]:
    reports = _read_json(SAVED_REPORTS_PATH, [])
    if isinstance(reports, list):
        return reports
    return []


def persist_reports(reports: List[dict]) -> None:
    _write_json(SAVED_REPORTS_PATH, reports)


def save_report(
    reports: List[dict],
    name: str,
    query: str,
    sql: str,
    data_df: pd.DataFrame,
    chart_type: str,
    config: Optional[Dict] = None,
) -> List[dict]:
    report = {
        "id": str(uuid.uuid4()),
        "name": name.strip() or f"Отчёт {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "query": query,
        "sql": sql,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "chart_type": chart_type,
        "config": config or {},
        "preview": data_df.head(20).to_dict(orient="records"),
        "columns": data_df.columns.tolist(),
    }
    updated = [report] + list(reports)
    persist_reports(updated)
    return updated


def delete_report(reports: List[dict], report_id: str) -> List[dict]:
    updated = [r for r in reports if r.get("id") != report_id]
    persist_reports(updated)
    return updated


def load_subscriptions() -> List[dict]:
    subscriptions = _read_json(SUBSCRIPTIONS_PATH, [])
    if isinstance(subscriptions, list):
        return subscriptions
    return []


def persist_subscriptions(subscriptions: List[dict]) -> None:
    _write_json(SUBSCRIPTIONS_PATH, subscriptions)


def add_subscription(
    subscriptions: List[dict],
    report_id: str,
    report_name: str,
    email: str,
    frequency: str,
    send_time: str,
) -> List[dict]:
    clean_email = (email or "").strip()
    updated = [
        s
        for s in subscriptions
        if not (
            s.get("report_id") == report_id
            and s.get("email") == clean_email
            and s.get("frequency") == frequency
            and s.get("send_time") == send_time
        )
    ]
    updated.append(
        {
            "id": str(uuid.uuid4()),
            "report_id": report_id,
            "report_name": report_name,
            "email": clean_email,
            "frequency": frequency,
            "send_time": send_time,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    persist_subscriptions(updated)
    return updated


def remove_subscription(subscriptions: List[dict], subscription_id: str) -> List[dict]:
    updated = [s for s in subscriptions if s.get("id") != subscription_id]
    persist_subscriptions(updated)
    return updated


def simulate_email(report: dict, email: str, schedule: dict, email_html: str) -> dict:
    frequency = schedule.get("frequency", "Еженедельно")
    send_time = schedule.get("send_time", "09:00")
    message = (
        f"Отчёт будет отправлен на email {email} ({frequency}, {send_time}). "
        "Сегодня мы отправили тестовое письмо."
    )
    subject = f"Drivee Data Buddy: {report.get('name', 'Отчёт')}"
    return {
        "message": message,
        "subject": subject,
        "html": email_html,
    }
