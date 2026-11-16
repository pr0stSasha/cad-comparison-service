# sheetgen-agent

Агент, описывающий различия между 3D-чертежом и его текстовым описанием.

**Вход:** PNG (растр) → **обработка:** перевод в векторное представление + разбор 2D-проекций/областей → **выход:** словесный отчёт «что не так».

## Как это работает

1. **Предобработка** ([`app/pipeline/preprocess.py`](app/pipeline/preprocess.py)) — декодирование PNG, ресайз, бинаризация (Otsu), склейка штрихов.
2. **Векторизация** ([`app/pipeline/vectorize.py`](app/pipeline/vectorize.py)) — извлечение примитивов (отрезки, полилинии, окружности) через контуры + `HoughLinesP`, генерация SVG-предпросмотра.
3. **2D-разбор проекций** ([`app/pipeline/projections.py`](app/pipeline/projections.py)) — сегментация чертежа на области (виды/проекции/разрезы) через связные компоненты.
4. **Метрики** ([`app/pipeline/features.py`](app/pipeline/features.py)) — количество примитивов, симметрия, мелкие компоненты (вероятные размерные числа) и т.д.
5. **LLM-агент** ([`app/agent.py`](app/agent.py)) — модель **Qwen 27B**, поднятая на локальной машине, получает изображение + векторное резюме + описание детали и возвращает JSON-отчёт о расхождениях.
6. **API и UI** ([`app/api.py`](app/api.py), [`static/index.html`](static/index.html)) — FastAPI + простая веб-страница для загрузки чертежа и просмотра отчёта.

## Запуск LLM на локальной машине

Нужен любой **OpenAI-совместимый** сервер с мультимодальной Qwen.

### Вариант 1: Ollama (проще всего)

```bash
ollama pull qwen2.5vl:27b
ollama serve        # если ещё не запущен, слушает http://127.0.0.1:11434
```

### Вариант 2: LM Studio / vLLM / llama.cpp server

Поднимите модель с включённым OpenAI-совместимым API (например, `http://127.0.0.1:1234/v1`).

## Установка и запуск агента

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env   # при необходимости поправьте LLM_BASE_URL / LLM_MODEL

python run.py
```

Открыть <http://localhost:8000>, загрузить PNG чертежа, вставить описание — получить отчёт.

## API

| Метод | Путь | Описание |
|---|---|---|
| `GET`  | `/api/health`  | статус + какая модель подключена |
| `POST` | `/api/analyze` | multipart: `file` (PNG) + `description` (текст) → JSON-отчёт |

Пример:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@drawing.png" \
  -F "description=Фланец круглый, D=120 мм, центральное отверстие 40 мм, 6 отверстий под болты"
```

Формат ответа: `vector_summary` (что извлекла векторизация), `features`, `views`, `svg` (предпросмотр), `report` (`summary`, `differences[]`, `report_md`).

## Настройка (`.env`)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | эндпоинт локальной LLM |
| `LLM_MODEL` | `qwen2.5vl:27b` | имя модели («квина 27б») |
| `LLM_API_KEY` | `none` | ключ, если требуется |
| `LLM_SUPPORTS_VISION` | `true` | отправлять ли изображение в модель |
| `MAX_IMAGE_PX` | `2048` | максимальная сторона после ресайза |
| `LLM_TEMPERATURE` | `0.2` | температура генерации |

> Если модель без визуальной модальности — поставьте `LLM_SUPPORTS_VISION=false`, тогда агент анализирует чертёж по векторному резюме (работает хуже, но работает).

## Структура проекта

```
app/
  config.py            # настройки из .env
  agent.py             # оркестрация: векторизация + LLM-сравнение
  api.py               # FastAPI-эндпоинты
  llm/client.py        # клиент OpenAI-совместимого API
  pipeline/
    preprocess.py      # декодирование, бинаризация
    vectorize.py       # примитивы + SVG
    projections.py     # сегментация на виды/проекции
    features.py        # метрики для промпта
static/index.html      # веб-интерфейс
run.py                 # запуск uvicorn
```
