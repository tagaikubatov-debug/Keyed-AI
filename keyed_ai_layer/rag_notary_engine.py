"""
rag_notary_engine.py — KeyedVault AI Layer
===========================================================
Запуск:  py -3.11 rag_notary_engine.py
Порт:    8001

"""

import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ─── Автозагрузка .env ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Настройки ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-5"        # быстрый и умный
KNOWLEDGE_DIR     = Path(__file__).parent / "legal_knowledge_base"
EMBED_MODEL       = "paraphrase-multilingual-MiniLM-L12-v2"
CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"
PORT              = 8001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("keyed")


# ─── Состояние приложения ─────────────────────────────────────────────────
class AppState:
    claude:     Optional[anthropic.Anthropic]   = None
    embedder:   Optional[SentenceTransformer]   = None
    clip:       Optional[CLIPModel]             = None
    processor:  Optional[CLIPProcessor]         = None
    chunks:     list[str]                       = []
    embeddings: Optional[np.ndarray]            = None

state = AppState()


# ─── Запуск ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Загружаю модели, подождите ~30 сек при первом запуске...")

    # Claude
    if ANTHROPIC_API_KEY:
        state.claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        log.info("✅  Claude подключён  (%s)", CLAUDE_MODEL)
    else:
        log.warning("⚠️  ANTHROPIC_API_KEY не найден!")
        log.warning("    Создай .env файл: ANTHROPIC_API_KEY=sk-ant-...")

    # Текстовые эмбеддинги для RAG
    state.embedder = SentenceTransformer(EMBED_MODEL)
    log.info("✅  Текстовая модель загружена")

    # CLIP для антиплагиата
    state.clip      = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    state.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    state.clip.eval()
    log.info("✅  CLIP загружен")

    # Индекс базы знаний
    state.chunks, state.embeddings = _build_index()
    log.info("✅  База знаний: %d фрагментов", len(state.chunks))

    log.info("🚀  Сервер запущен → http://localhost:%d/docs", PORT)
    yield
    log.info("Остановка...")


app = FastAPI(
    title="KeyedVault AI Layer",
    description="RAG-нотариус | AI-адвокат | CLIP-антиплагиат (Claude)",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── База знаний ──────────────────────────────────────────────────────────
def _build_index() -> tuple[list[str], np.ndarray]:
    """Читает .txt файлы из legal_knowledge_base/ и строит векторный индекс."""
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True)
        log.warning("Папка legal_knowledge_base/ пустая — добавь .txt файлы с законами")
        return [], np.zeros((0, 384))

    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("**/*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        for i in range(0, len(text), 700):
            chunk = text[i : i + 800].strip()
            if chunk:
                chunks.append(f"[{path.stem}]\n{chunk}")

    if not chunks:
        return [], np.zeros((0, 384))

    vecs = state.embedder.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False
    )
    return chunks, np.array(vecs)


def _find_relevant(query: str, top_k: int = 4) -> list[str]:
    """Косинусный поиск — возвращает самые релевантные фрагменты."""
    if state.embeddings is None or len(state.chunks) == 0:
        return []

    q      = state.embedder.encode([query], normalize_embeddings=True)[0]
    scores = state.embeddings @ q
    top    = np.argsort(scores)[::-1][:top_k]
    return [state.chunks[i] for i in top if scores[i] > 0.20]


# ─── Вызов Claude ─────────────────────────────────────────────────────────
def _ask_claude(system: str, user: str, max_tokens: int = 1500) -> str:
    """Отправляет запрос в Claude и возвращает текст ответа."""
    if not state.claude:
        raise HTTPException(
            status_code=503,
            detail="Claude не подключён. Создай .env: ANTHROPIC_API_KEY=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"
        )

    message = state.claude.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = max_tokens,
        system     = system,
        messages   = [{"role": "user", "content": user}]
    )
    return message.content[0].text


# ─── Модели данных ────────────────────────────────────────────────────────
class NotaryRequest(BaseModel):
    description: str       # "Доверенность на управление машиной"
    author_id:   str       # ID пользователя из KeyedVault
    asset_hash:  str = ""  # хэш файла (необязательно)

class NotaryResponse(BaseModel):
    status:    str
    document:  str
    law_refs:  list[str]
    timestamp: float

class LegalRequest(BaseModel):
    question: str         # "Как подать жалобу в ЕСПЧ?"
    context:  str = ""    # дополнительный контекст
    language: str = "ru"  # ru / en / kz

class LegalResponse(BaseModel):
    status:     str
    answer:     str
    sources:    list[str]
    disclaimer: str

class PlagiarismResponse(BaseModel):
    status:         str
    similarity_pct: float   # 0.0 — 100.0
    verdict:        str     # ORIGINAL / SUSPECT / DUPLICATE
    details:        str


# ═══════════════════════════════════════════════════════════════════════════
#  ЭНДПОИНТ 1: RAG-НОТАРИУС
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/notary/generate", response_model=NotaryResponse, tags=["Нотариус"])
async def generate_document(req: NotaryRequest):
    """
    Генерирует нотариальный документ по описанию.
    RAG: сначала ищет нужные нормы → потом Claude составляет документ.
    """
    relevant = _find_relevant(req.description, top_k=4)
    law_text = "\n\n---\n\n".join(relevant) if relevant else "Используй общие нормы права."

    system = (
        "Ты опытный нотариус с 20-летней практикой. "
        "Составляй чёткие юридически грамотные документы: "
        "заголовок, стороны, суть, правовое основание, дата и место. "
        "Отвечай на том же языке, на котором задан вопрос."
    )

    user = (
        f'Составь нотариальный документ по запросу:\n"{req.description}"\n\n'
        f"Автор: {req.author_id}\n"
        + (f"Файл в Vault: {req.asset_hash}\n" if req.asset_hash else "")
        + f"\nПравовые нормы:\n{law_text}\n\n"
        "Верни только готовый документ без пояснений."
    )

    document = _ask_claude(system, user, max_tokens=1500)
    sources  = list({c.split("\n")[0].strip("[]") for c in relevant})

    return NotaryResponse(
        status    = "SUCCESS",
        document  = document,
        law_refs  = sources,
        timestamp = time.time()
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ЭНДПОИНТ 2: AI-АДВОКАТ
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/legal/ask", response_model=LegalResponse, tags=["Адвокат"])
async def legal_advisor(req: LegalRequest):
    """
    Отвечает на вопросы по международному праву:
    ЕКПЧ, МУС, ООН, права человека.
    """
    relevant = _find_relevant(req.question + " " + req.context, top_k=5)
    law_text = "\n\n---\n\n".join(relevant) if relevant else "Используй международное право."

    lang = {
        "ru": "Отвечай на русском.",
        "en": "Answer in English.",
        "kz": "Қазақша жауап бер."
    }.get(req.language, "Отвечай на русском.")

    system = (
        "Ты опытный международный юрист по правам человека. "
        "Специализация: ЕКПЧ, МУС, система ООН. "
        f"Давай практические советы с конкретными статьями и шагами. {lang}"
    )

    user = (
        f'Вопрос: "{req.question}"\n'
        + (f"Контекст: {req.context}\n" if req.context else "")
        + f"\nРелевантные нормы:\n{law_text}\n\n"
        "Дай развёрнутый практический ответ."
    )

    answer  = _ask_claude(system, user, max_tokens=2000)
    sources = list({c.split("\n")[0].strip("[]") for c in relevant})

    return LegalResponse(
        status     = "SUCCESS",
        answer     = answer,
        sources    = sources,
        disclaimer = (
            "Ответ носит общеправовой характер "
            "и не заменяет консультацию лицензированного адвоката."
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ЭНДПОИНТ 3: CLIP-АНТИПЛАГИАТ
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/plagiarism/check", response_model=PlagiarismResponse, tags=["Антиплагиат"])
async def check_plagiarism(
    file_a: UploadFile = File(..., description="Оригинальное изображение"),
    file_b: UploadFile = File(..., description="Проверяемое изображение"),
):
    """
    Сравнивает два изображения через CLIP.
    Результат: схожесть в % + вердикт ORIGINAL / SUSPECT / DUPLICATE.
    """
    if state.clip is None:
        raise HTTPException(503, "CLIP модель не загружена")

    try:
        img_a = Image.open(file_a.file).convert("RGB")
        img_b = Image.open(file_b.file).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Не удалось открыть изображение: {e}")

    with torch.no_grad():
        feat_a = state.clip.get_image_features(
            **state.processor(images=img_a, return_tensors="pt")
        )
        feat_b = state.clip.get_image_features(
            **state.processor(images=img_b, return_tensors="pt")
        )
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
        score  = float((feat_a * feat_b).sum())

    pct = round(score * 100, 2)

    if score >= 0.95:
        verdict = "DUPLICATE"
        details = f"Изображения идентичны ({pct}%)."
    elif score >= 0.80:
        verdict = "SUSPECT"
        details = f"Высокое сходство ({pct}%) — рекомендуется проверка."
    else:
        verdict = "ORIGINAL"
        details = f"Изображения различны ({pct}%). Плагиат маловероятен."

    return PlagiarismResponse(
        status         = "SUCCESS",
        similarity_pct = pct,
        verdict        = verdict,
        details        = details,
    )


# ─── Health check ─────────────────────────────────────────────────────────

@app.get("/health", tags=["Система"])
async def health():
    return {
        "status":    "ok",
        "claude":    state.claude is not None,
        "model":     CLAUDE_MODEL,
        "clip":      state.clip is not None,
        "embedder":  state.embedder is not None,
        "kb_chunks": len(state.chunks),
    }


if __name__ == "__main__":
    uvicorn.run("rag_notary_engine:app", host="0.0.0.0", port=PORT, reload=False)