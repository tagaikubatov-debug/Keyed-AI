# Keyed-AI# KeyedVault AI Layer
### RAG-нотариус · AI-адвокат · CLIP-антиплагиат

Три AI-модуля поверх существующего Spring Boot + Python стека.  
Ничего в оригинальном коде менять не нужно — всё добавляется рядом.

---

## Что добавляется

| Файл | Назначение |
|------|-----------|
| `rag_notary_engine.py` | FastAPI AI-сервис (порт 8001) |
| `requirements_ai.txt` | Python зависимости |
| `legal_knowledge_base/*.txt` | База правовых знаний (RAG) |
| `NotaryService.java` | Spring-сервис: HTTP-клиент к Python |
| `NotaryController.java` | 4 новых REST-эндпоинта `/api/ai/...` |

---

## Установка

### Шаг 1 — Python зависимости
```bash
pip install -r requirements_ai.txt
```
*Первый раз скачает ~1.5 GB моделей (CLIP + sentence-transformers). Последующие запуски мгновенные.*

### Шаг 2 — API ключ Anthropic
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Linux / macOS
export ANTHROPIC_API_KEY=sk-ant-...
```
Получить ключ: https://console.anthropic.com

### Шаг 3 — Скопировать файлы в проект
```
Assets/
  src/main/java/com/example/Assets/
    controller/NotaryController.java   ← скопировать
    service/NotaryService.java         ← скопировать
  src/main/resources/
    application.properties             ← добавить строки из application_properties_additions.txt

rag_notary_engine.py                   ← скопировать рядом с watermark_engine.py
requirements_ai.txt                    ← рядом
legal_knowledge_base/                  ← папку целиком
```

### Шаг 4 — Запуск
```bash
# Терминал 1: Python AI-движок
python rag_notary_engine.py

# Терминал 2: Spring Boot (как обычно)
./mvnw spring-boot:run
```

---

## API Эндпоинты

### 1. Генерация нотариального документа
```http
POST http://localhost:8080/api/ai/notary/generate
Content-Type: application/json

{
  "description": "Доверенность на управление автомобилем Toyota Camry 2020",
  "authorId": "EMIR_2024",
  "assetHash": "ABC123"
}
```
**Ответ:**
```json
{
  "status": "SUCCESS",
  "document": "ДОВЕРЕННОСТЬ\n\nг. Бишкек, двадцатого мая две тысячи двадцать шестого года...",
  "law_refs": ["03_notary_documents_kg", "01_echr_ekpch"],
  "timestamp": 1748000000.0
}
```

---

### 2. AI-адвокат (международное право)
```http
POST http://localhost:8080/api/ai/legal/ask
Content-Type: application/json

{
  "question": "Как подать жалобу в ЕСПЧ против незаконного задержания?",
  "context": "Задержан без предъявления обвинения на 72 часа.",
  "language": "ru"
}
```
**Ответ:**
```json
{
  "status": "SUCCESS",
  "answer": "Для подачи жалобы в Европейский суд по правам человека...",
  "sources": ["01_echr_ekpch", "02_icc_icj_un"],
  "disclaimer": "Информация носит общеправовой характер..."
}
```

---

### 3. CLIP-антиплагиат
```http
POST http://localhost:8080/api/ai/plagiarism/check
Content-Type: multipart/form-data

file_a: [оригинальное изображение]
file_b: [проверяемое изображение]
```
**Ответ:**
```json
{
  "status": "SUCCESS",
  "similarity_pct": 94.7,
  "verdict": "SUSPECT",
  "details": "Высокое визуальное сходство — возможен плагиат (94.7%). Рекомендуется ручная проверка."
}
```
Вердикты: `ORIGINAL` (< 80%), `SUSPECT` (80–95%), `DUPLICATE` (≥ 95%)

---

### 4. Health Check
```http
GET http://localhost:8080/api/ai/health
```
```json
{
  "spring_backend": "UP",
  "python_ai_engine": "UP",
  "message": "Все системы работают"
}
```

---

## Расширение базы знаний

Добавьте любой `.txt` файл в папку `legal_knowledge_base/` — он автоматически  
индексируется при следующем запуске `rag_notary_engine.py`.

Примеры документов для добавления:
- Гражданский кодекс КР (выдержки)
- Конвенция ООН о правах ребёнка
- Трудовой кодекс КР
- GDPR (для IT-проектов)
- Ваши внутренние шаблоны документов

---

## Архитектура

```
Frontend / Postman
      │
      ▼
Spring Boot :8080
  NotaryController  (/api/ai/...)
      │  HTTP JSON
      ▼
Python FastAPI :8001  (rag_notary_engine.py)
  ├── RAG: SentenceTransformers + FAISS-like cosine search
  ├── LLM: Anthropic Claude (claude-opus-4-5)
  └── Vision: OpenAI CLIP (clip-vit-base-patch32)
      │
      ▼
legal_knowledge_base/*.txt  (векторный индекс)
```
