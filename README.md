# Drivee Data Buddy

## Ссылка доступа: <https://drivee-data-buddy.streamlit.app/>

Drivee Data Buddy — веб-приложение для бизнеса: пользователь задаёт вопрос на естественном языке, приложение строит SQL, валидирует его, выполняет в DuckDB и отображает таблицу + график. Поддерживает готовые сценарии из `templates.json` и экспорт результатов в HTML‑отчёт.

### Разработано командой "404: Имя не найдено", в рамках ГРАНД-ФИНАЛА проекта "Моя профессия ИТ 2025/26"

## Что умеет

- Семантический маршрут: метрики/измерения из `semantic/semantic_layer.json`
- LLM: генерация SQL через GitHub Models или DeepSeek (OpenAI‑совместимый API)
- Guardrails для SQL-безопасности (`SELECT`/`WITH`, запрет DDL/DML, только `incity_orders`, авто-`LIMIT`)
- Загрузка собственного CSV через интерфейс
- Готовые сценарии из `templates.json` (активные кнопки в сайдбаре)
- Скачивание HTML‑отчёта с таблицей, SQL и интерактивным графиком

## Архитектура

- `app.py` — UI, роутинг запроса, выполнение SQL, визуализация, экспорт отчётов, обработка сценариев
- `llm_client.py` — инициализация LLM‑клиента (GitHub Models, DeepSeek, Mock), промпт, генерация SQL
- `guardrails.py` — валидация и нормализация SQL
- `semantic/semantic_layer.json` — семантический слой (метрики, измерения, синонимы, временные выражения)
- `templates.json` — готовые аналитические сценарии (метрика + измерение)

## Локальный запуск

1. Установите зависимости:

```text
pip install -r requirements.txt
```

2. (Если хотите использовать свои API) создайте `.env` для LLM, например:

```text
LLM_PROVIDER=github      
GITHUB_TOKEN=           
GITHUB_MODEL=gpt-4o      
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
```

3. Запустите приложение:

```text
streamlit run app.py
```

## Примечания по безопасности SQL

`guardrails.py` проверяет, что запрос:

- не пустой;
- содержит только один statement;
- начинается с `SELECT` или `WITH`;
- не содержит запрещённые операции (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`, `TRUNCATE`, `REPLACE`);
- использует только `incity_orders` (и CTE-алиасы);
- получает `LIMIT 1000`, если лимит не задан.
