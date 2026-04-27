"""
Логирование запросов в JSON-файл для последующей аналитики.

Минимальная реализация — без БД. Подходит для прототипа.
В роадмапе — миграция на SQLite + Plotly-дашборд.
"""
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs.json")


def log_query(
    question: str,
    response: str,
    context: str,
    level: str = "beginner",
) -> None:
    """Дописывает одну запись в logs.json."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "question": question,
        "response": response,
        "context_length": len(context),
    }

    if LOG_FILE.exists():
        try:
            with LOG_FILE.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, OSError):
            logs = []
    else:
        logs = []

    logs.append(entry)

    with LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
