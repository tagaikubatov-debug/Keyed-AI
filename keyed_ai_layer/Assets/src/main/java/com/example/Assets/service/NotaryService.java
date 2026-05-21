package com.example.Assets.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;

/**
 * NotaryService — мост между Spring Boot и Python AI-движком (rag_notary_engine.py).
 *
 * Не содержит бизнес-логики AI — просто аккуратно общается с FastAPI
 * по HTTP и пробрасывает результаты наверх в контроллер.
 *
 * Интеграция в 3 шага:
 *   1. Запустить rag_notary_engine.py (python rag_notary_engine.py)
 *   2. Добавить в application.properties: ai.engine.url=http://localhost:8001
 *   3. Использовать NotaryController для вызовов
 */
@Service
public class NotaryService {

    private static final Logger log = LoggerFactory.getLogger(NotaryService.class);

    // URL Python AI-сервиса, задаётся в application.properties
    @Value("${ai.engine.url:http://localhost:8001}")
    private String aiEngineUrl;

    // Таймаут для AI-запросов (генерация документа может занять ~10 сек)
    private static final int TIMEOUT_SECONDS = 60;

    private final HttpClient httpClient;
    private final ObjectMapper json;

    public NotaryService() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.json = new ObjectMapper();
    }

    // =========================================================================
    //  ГЕНЕРАЦИЯ НОТАРИАЛЬНОГО ДОКУМЕНТА
    // =========================================================================

    /**
     * Отправляет описание пользователя в RAG-нотариус и возвращает готовый документ.
     *
     * @param description  Что нужно ("Доверенность на управление автомобилем")
     * @param authorId     ID автора из KeyedVault
     * @param assetHash    Опциональная привязка к файлу в Vault
     * @return Map с ключами: status, document, law_refs, timestamp
     */
    public Map<String, Object> generateNotaryDocument(
            String description, String authorId, String assetHash) {

        String body = toJson(Map.of(
                "description", description,
                "author_id",   authorId,
                "asset_hash",  assetHash != null ? assetHash : ""
        ));

        return postJson(aiEngineUrl + "/notary/generate", body);
    }

    // =========================================================================
    //  AI-АДВОКАТ (МЕЖДУНАРОДНОЕ ПРАВО)
    // =========================================================================

    /**
     * Задаёт вопрос AI-адвокату по международному праву.
     *
     * @param question  Вопрос пользователя
     * @param context   Дополнительный контекст (необязательно)
     * @param language  Язык ответа: "ru", "en", "kz"
     * @return Map с ключами: status, answer, sources, disclaimer
     */
    public Map<String, Object> askLegalAdvisor(
            String question, String context, String language) {

        String body = toJson(Map.of(
                "question",  question,
                "context",   context != null ? context : "",
                "language",  language != null ? language : "ru"
        ));

        return postJson(aiEngineUrl + "/legal/ask", body);
    }

    // =========================================================================
    //  CLIP-АНТИПЛАГИАТ
    // =========================================================================

    /**
     * Сравнивает два изображения через CLIP-эмбеддинги.
     * Возвращает схожесть 0–100% и вердикт ORIGINAL / SUSPECT / DUPLICATE.
     *
     * @param fileA  Исходное изображение (из KeyedVault)
     * @param fileB  Проверяемое изображение
     * @return Map с ключами: status, similarity_pct, verdict, details
     */
    public Map<String, Object> checkPlagiarism(MultipartFile fileA, MultipartFile fileB) {
        return postMultipart(
                aiEngineUrl + "/plagiarism/check",
                Map.of("file_a", fileA, "file_b", fileB)
        );
    }

    // =========================================================================
    //  ПРОВЕРКА ДОСТУПНОСТИ AI-ДВИЖКА
    // =========================================================================

    /**
     * Пингует Python-сервис. Удобно показывать статус на UI.
     */
    public boolean isAiEngineAlive() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(aiEngineUrl + "/health"))
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;

        } catch (Exception e) {
            log.warn("AI Engine недоступен: {}", e.getMessage());
            return false;
        }
    }

    // =========================================================================
    //  ПРИВАТНЫЕ УТИЛИТЫ
    // =========================================================================

    /** Отправляет POST с JSON-телом и разбирает ответ в Map. */
    private Map<String, Object> postJson(String url, String body) {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofSeconds(TIMEOUT_SECONDS))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());

            if (resp.statusCode() != 200) {
                log.error("AI Engine вернул {}: {}", resp.statusCode(), resp.body());
                return errorMap("AI Engine error " + resp.statusCode() + ": " + resp.body());
            }

            return jsonToMap(resp.body());

        } catch (Exception e) {
            log.error("Не удалось связаться с AI Engine ({}): {}", url, e.getMessage());
            return errorMap("AI Engine недоступен. Убедитесь, что rag_notary_engine.py запущен. " + e.getMessage());
        }
    }

    /** Отправляет multipart/form-data (для изображений в антиплагиате). */
    private Map<String, Object> postMultipart(String url, Map<String, MultipartFile> files) {
        try {
            String boundary = "----KeyedVaultBoundary" + System.currentTimeMillis();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();

            for (Map.Entry<String, MultipartFile> entry : files.entrySet()) {
                String fieldName = entry.getKey();
                MultipartFile file = entry.getValue();
                String filename   = file.getOriginalFilename() != null
                        ? file.getOriginalFilename() : "image.png";

                // Boundary start
                baos.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
                baos.write(("Content-Disposition: form-data; name=\"" + fieldName
                        + "\"; filename=\"" + filename + "\"\r\n").getBytes(StandardCharsets.UTF_8));
                baos.write(("Content-Type: " + file.getContentType() + "\r\n\r\n")
                        .getBytes(StandardCharsets.UTF_8));
                baos.write(file.getBytes());
                baos.write("\r\n".getBytes(StandardCharsets.UTF_8));
            }
            baos.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofSeconds(TIMEOUT_SECONDS))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray()))
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());

            if (resp.statusCode() != 200) {
                return errorMap("Plagiarism engine error " + resp.statusCode() + ": " + resp.body());
            }

            return jsonToMap(resp.body());

        } catch (Exception e) {
            log.error("Ошибка multipart запроса к {}: {}", url, e.getMessage());
            return errorMap("Ошибка антиплагиата: " + e.getMessage());
        }
    }

    /** Сериализует Map в JSON-строку без внешней зависимости. */
    private String toJson(Map<String, Object> data) {
        try {
            return json.writeValueAsString(data);
        } catch (Exception e) {
            throw new RuntimeException("JSON serialization failed", e);
        }
    }

    /** Десериализует JSON-строку в Map<String, Object>. */
    @SuppressWarnings("unchecked")
    private Map<String, Object> jsonToMap(String jsonStr) {
        try {
            return json.readValue(jsonStr, Map.class);
        } catch (Exception e) {
            log.error("Не удалось распарсить ответ AI: {}", jsonStr);
            return errorMap("Некорректный ответ от AI Engine");
        }
    }

    private Map<String, Object> errorMap(String message) {
        Map<String, Object> err = new HashMap<>();
        err.put("status",  "ERROR");
        err.put("message", message);
        return err;
    }
}
