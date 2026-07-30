/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.yolo;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import io.bioimage.modelrunner.model.special.yolo.YoloTrainingAttemptResult;

final class YoloTrainingResults {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private final OffsetDateTime startedAt = OffsetDateTime.now();
    private final List<Map<String, Object>> attempts = new ArrayList<Map<String, Object>>();
    private Map<String, Object> dataset = new LinkedHashMap<String, Object>();
    private String status = "running";
    private YoloTrainingAttemptResult completed;
    private String failure;

    void setDataset(Map<String, Object> summary) {
        dataset = summary == null
                ? new LinkedHashMap<String, Object>()
                : new LinkedHashMap<String, Object>(summary);
    }

    void addAttempt(int number, Object requestedBatch, Integer resolvedBatch, String attemptStatus) {
        Map<String, Object> attempt = new LinkedHashMap<String, Object>();
        attempt.put("attempt", Integer.valueOf(number));
        attempt.put("requested_batch", requestedBatch);
        attempt.put("resolved_batch", resolvedBatch);
        attempt.put("status", attemptStatus);
        attempts.add(attempt);
    }

    void complete(YoloTrainingAttemptResult result) {
        status = "completed";
        completed = result;
    }

    void retainPartial(YoloTrainingAttemptResult result) {
        if (result != null) {
            completed = result;
        }
    }

    void fail(String finalStatus, Throwable error) {
        status = finalStatus;
        failure = error == null ? null : errorMessage(error);
    }

    void cancel() {
        status = "cancelled";
    }

    void cancel(YoloTrainingAttemptResult result) {
        completed = result;
        cancel();
    }

    void write(YoloTrainingConfig config) throws IOException {
        OffsetDateTime finishedAt = OffsetDateTime.now();
        Map<String, Object> root = new LinkedHashMap<String, Object>();
        root.put("framework", "yolo");
        root.put("format_version", Integer.valueOf(0));
        root.put("status", status);
        root.put("started_at", startedAt.toString());
        root.put("finished_at", finishedAt.toString());
        root.put("duration_seconds", Long.valueOf(Duration.between(startedAt, finishedAt).getSeconds()));
        root.put("dataset", dataset);
        root.put("batch_attempts", attempts);
        root.put("training", trainingMap(config));
        root.put("artifacts", artifactMap(config));
        root.put("failure", failure);
        File file = config.getTrainingResultsFile();
        Files.createDirectories(file.getParentFile().toPath());
        File temporary = new File(file.getParentFile(), file.getName() + ".tmp");
        try (Writer writer = Files.newBufferedWriter(temporary.toPath(), StandardCharsets.UTF_8)) {
            GSON.toJson(root, writer);
        }
        replaceAtomically(temporary, file);
    }

    private Map<String, Object> trainingMap(YoloTrainingConfig config) {
        Map<String, Object> training = new LinkedHashMap<String, Object>();
        training.put("requested_epochs", Integer.valueOf(config.getEpochs()));
        training.put("completed_epochs", Integer.valueOf(completed == null ? 0 : completed.getCompletedEpochs()));
        training.put("best_epoch", Integer.valueOf(completed == null ? 0 : completed.getBestEpoch()));
        training.put("best_metrics", completed == null ? null : completed.getBestMetrics());
        training.put("final_metrics", completed == null ? null : completed.getFinalMetrics());
        return training;
    }

    private Map<String, Object> artifactMap(YoloTrainingConfig config) {
        Map<String, Object> artifacts = new LinkedHashMap<String, Object>();
        File outputDir = config.getOutputModelDir();
        artifacts.put("best_checkpoint", relativeIfFile(outputDir, new File(outputDir, "weights/best.pt")));
        artifacts.put("last_checkpoint", relativeIfFile(outputDir, new File(outputDir, "weights/last.pt")));
        artifacts.put("model_file", relativeIfFile(outputDir, new File(config.getOutputWeightsPath())));
        artifacts.put("ui_log", relativeIfFile(outputDir, new File(outputDir, "training-ui.log")));
        artifacts.put("backend_log", relativeIfFile(outputDir, new File(outputDir, "training.log")));
        return artifacts;
    }

    private static String relativeIfFile(File root, File file) {
        return file.isFile() ? root.toPath().relativize(file.toPath()).toString().replace('\\', '/') : null;
    }

    private static void replaceAtomically(File source, File target) throws IOException {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String errorMessage(Throwable error) {
        Throwable current = error;
        String message = null;
        while (current != null) {
            if (current.getMessage() != null && !current.getMessage().trim().isEmpty()) {
                message = current.getMessage().trim();
            }
            current = current.getCause();
        }
        return message == null ? error.getClass().getSimpleName() : message;
    }
}
