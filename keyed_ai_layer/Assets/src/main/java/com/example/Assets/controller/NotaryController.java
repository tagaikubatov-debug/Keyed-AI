package com.example.Assets.controller;

import com.example.Assets.service.NotaryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * NotaryController — три новых AI-эндпоинта поверх существующего стека KeyedVault.
 *
 * POST /api/ai/notary/generate     — генерация нотариального документа (RAG)
 * POST /api/ai/legal/ask           — вопрос AI-адвокату по международному праву
 * POST /api/ai/plagiarism/check    — CLIP-антиплагиат для двух изображений
 * GET  /api/ai/health              — проверка доступности Python AI-движка
 *
 * Все эндпоинты независимы от существующего MainController —
 * ничего в оригинальном коде менять не нужно.
 */
@RestController
@RequestMapping("/api/ai")
@CrossOrigin(origins = "*")
public class NotaryController {

    private final NotaryService notaryService;

    @Autowired
    public NotaryController(NotaryService notaryService) {
        this.notaryService = notaryService;
    }

    // =========================================================================
    //  1. RAG-НОТАРИУС: генерация юридического документа
    // =========================================================================

    /**
     * Генерирует нотариальный документ по текстовому описанию.
     *
     * Пример запроса:
     * POST /api/ai/notary/generate
     * {
     *   "description": "Нужна доверенность на получение пенсии в банке",
     *   "authorId":    "EMIR_2024",
     *   "assetHash":   ""
     * }
     *
     * Ответ содержит: готовый документ, ссылки на законы, временну́ю метку.
     */
    @PostMapping("/notary/generate")
    public ResponseEntity<Map<String, Object>> generateNotaryDoc(
            @RequestBody Map<String, String> body) {

        String description = body.getOrDefault("description", "").trim();
        String authorId    = body.getOrDefault("authorId",    "ANONYMOUS");
        String assetHash   = body.getOrDefault("assetHash",   "");

        if (description.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "status",  "ERROR",
                    "message", "Поле 'description' обязательно"
            ));
        }

        Map<String, Object> result = notaryService.generateNotaryDocument(
                description, authorId, assetHash);

        return toResponse(result);
    }

    // =========================================================================
    //  2. AI-АДВОКАТ: вопросы по международному праву
    // =========================================================================

    /**
     * Отвечает на юридический вопрос с опорой на базу знаний и Claude.
     *
     * Пример запроса:
     * POST /api/ai/legal/ask
     * {
     *   "question":  "Как подать жалобу в ЕКПЧ против действий полиции?",
     *   "context":   "Задержание прошло без предъявления обвинения, 48 часов.",
     *   "language":  "ru"
     * }
     */
    @PostMapping("/legal/ask")
    public ResponseEntity<Map<String, Object>> askLegal(
            @RequestBody Map<String, String> body) {

        String question = body.getOrDefault("question", "").trim();
        String context  = body.getOrDefault("context",  "");
        String language = body.getOrDefault("language", "ru");

        if (question.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "status",  "ERROR",
                    "message", "Поле 'question' обязательно"
            ));
        }

        Map<String, Object> result = notaryService.askLegalAdvisor(question, context, language);
        return toResponse(result);
    }

    // =========================================================================
    //  3. CLIP-АНТИПЛАГИАТ: сравнение двух изображений
    // =========================================================================

    /**
     * Сравнивает два изображения через CLIP-эмбеддинги.
     * Возвращает схожесть в % и вердикт: ORIGINAL / SUSPECT / DUPLICATE.
     *
     * Пример запроса (multipart/form-data):
     * POST /api/ai/plagiarism/check
     *   file_a = [изображение из Vault]
     *   file_b = [проверяемое изображение]
     */
    @PostMapping("/plagiarism/check")
    public ResponseEntity<Map<String, Object>> checkPlagiarism(
            @RequestParam("file_a") MultipartFile fileA,
            @RequestParam("file_b") MultipartFile fileB) {

        if (fileA.isEmpty() || fileB.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "status",  "ERROR",
                    "message", "Оба файла (file_a, file_b) обязательны"
            ));
        }

        Map<String, Object> result = notaryService.checkPlagiarism(fileA, fileB);
        return toResponse(result);
    }

    // =========================================================================
    //  4. HEALTH CHECK: жив ли Python AI-движок?
    // =========================================================================

    /**
     * Быстрая проверка — запущен ли rag_notary_engine.py.
     * Удобно показывать зелёный/красный индикатор на UI.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        boolean alive = notaryService.isAiEngineAlive();
        return ResponseEntity.ok(Map.of(
                "spring_backend",   "UP",
                "python_ai_engine", alive ? "UP" : "DOWN",
                "message",          alive
                        ? "Все системы работают"
                        : "Python AI-движок недоступен — запустите rag_notary_engine.py"
        ));
    }

    // =========================================================================
    //  ВСПОМОГАТЕЛЬНЫЙ МЕТОД
    // =========================================================================

    /** Возвращает 200 если status=SUCCESS, иначе 502 Bad Gateway. */
    private ResponseEntity<Map<String, Object>> toResponse(Map<String, Object> result) {
        boolean ok = "SUCCESS".equals(result.get("status"));
        return ok
                ? ResponseEntity.ok(result)
                : ResponseEntity.status(502).body(result);
    }
}
