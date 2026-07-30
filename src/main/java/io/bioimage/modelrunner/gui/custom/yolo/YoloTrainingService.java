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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.stream.Stream;

import org.apposed.appose.BuildException;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.model.special.yolo.YoloTrainingAttemptResult;
import io.bioimage.modelrunner.model.special.yolo.YoloTrainingProgress;
import io.bioimage.modelrunner.model.special.yolo.YoloValidationPreview;

public class YoloTrainingService {

    private static final long CANCEL_FALLBACK_TIMEOUT_MS = 250L;

    private final ModelInstaller installer;
    private File cancelSignalFile;
    private Service runningPython;
    private volatile boolean cancelRequested;

    /**
     * Creates a new YoloTrainingService instance.
     *
     * @param installer the installer.
     */
    public YoloTrainingService(ModelInstaller installer) {
        this.installer = installer;
    }

    /**
     * Runs model training.
     *
     * @param config the config.
     * @param progressConsumer the progress consumer callback.
     * @param previewConsumer the preview consumer callback.
     * @param logConsumer the log consumer callback.
     * @throws IOException if an I/O error occurs.
     * @throws ExecutionException if an asynchronous operation fails.
     * @throws InterruptedException if the current thread is interrupted.
     * @throws BuildException if the Python environment or service cannot be built.
     * @throws TaskException if task occurs.
     */
    public void train(YoloTrainingConfig config,
            Consumer<YoloTrainingProgress> progressConsumer,
            Consumer<YoloValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        validate(config);
        cancelRequested = false;
        Files.createDirectories(config.getOutputModelDir().toPath());
        cleanupStaleAttempts(config.getOutputModelDir());
        YoloTrainingResults results = new YoloTrainingResults();
        YoloDatasetPreparationResult dataset = null;
        File cancelFile = null;
        try {
            dataset = YoloDatasetPreparer.prepareDetailed(config.getDatasetYamlPath(), config.getModelName(),
                    config.getModelsDir(), config.getImageSize(), logConsumer);
            results.setDataset(dataset.getSummary());
            File configFile = config.writeConfig(dataset.getYaml());
            log(logConsumer, "Config file saved at: " + configFile.getAbsolutePath());
            log(logConsumer, "YOLO training parameters:");
            for (String line : config.toLogLines(dataset.getYaml())) {
                log(logConsumer, "  " + line);
            }
            if (!installer.isEnvironmentInstalled()) {
                installer.installEnvironment(logConsumer);
            }
            if (config.isFineTune() && !installer.isModelInstalled(config.getBaseModelPath())) {
                installer.installModelWeights(config.getBaseModelPath(), logConsumer);
            }
            cancelFile = beginCancelSignal();
            final File resolvedYaml = dataset.getYaml();
            Object batch = config.getRequestedBatch();
            int attemptNumber = 0;
            while (true) {
                attemptNumber++;
                File attemptDir = attemptDirectory(config, attemptNumber);
                Files.createDirectories(attemptDir.toPath());
                File attemptOutput = new File(attemptDir, attemptDir.getName()
                        + YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
                AtomicReference<YoloTrainingAttemptResult> latest =
                        new AtomicReference<YoloTrainingAttemptResult>();
                Consumer<String> attemptLog = message -> log(logConsumer,
                        canonicalizeAttemptPath(message, attemptDir, config.getOutputModelDir()));
                try {
                    YoloTrainingAttemptResult completed = Yolo.trainAttempt(
                            config.getEpochs(), config.getBaseModelPath(),
                            config.isFineTune() ? null : config.getTrainingModelSource(),
                            resolvedYaml.getAbsolutePath(), attemptOutput.getAbsolutePath(),
                            config.getImageSize(), config.getPreviewEpochPeriod(),
                            config.getBackendTrainingOptions(), batch,
                            progressConsumer, previewConsumer, attemptLog, config.getDevice(),
                            cancelFile.getAbsolutePath(), this::setRunningPython, runtime -> {
                                latest.set(runtime);
                                config.updateRuntime(runtime);
                                try {
                                    config.writeConfig(resolvedYaml);
                                } catch (IOException e) {
                                    log(logConsumer, "Could not update resolved YOLO runtime in config: "
                                            + e.getMessage());
                                }
                            });
                    config.updateRuntime(completed);
                    config.writeConfig(resolvedYaml);
                    promoteAttempt(attemptDir, attemptOutput, config);
                    if (cancelRequested) {
                        results.addAttempt(attemptNumber, batch, completed.getResolvedBatch(), "cancelled");
                        results.cancel(completed);
                        throw new InterruptedException("YOLO training was cancelled.");
                    }
                    results.addAttempt(attemptNumber, batch, completed.getResolvedBatch(), "completed");
                    results.complete(completed);
                    break;
                } catch (TaskException e) {
                    YoloTrainingAttemptResult partial = latest.get();
                    config.updateRuntime(partial);
                    results.retainPartial(partial);
                    config.writeConfig(resolvedYaml);
                    Integer resolved = partial == null ? null : partial.getResolvedBatch();
                    if (cancelRequested) {
                        results.addAttempt(attemptNumber, batch, resolved, "cancelled");
                        preserveInterruptedAttempt(attemptDir, attemptOutput, config);
                        if (partial == null) {
                            results.cancel();
                        } else {
                            results.cancel(partial);
                        }
                        throw e;
                    }
                    if (!config.isOomRetryEnabled() || !isOutOfMemory(e)) {
                        results.addAttempt(attemptNumber, batch, resolved, "failed");
                        preserveInterruptedAttempt(attemptDir, attemptOutput, config);
                        throw e;
                    }
                    Integer previous = resolved;
                    if (previous == null && batch instanceof Number) {
                        previous = Integer.valueOf(((Number) batch).intValue());
                    }
                    if (previous == null) {
                        int next = Math.max(config.getMinimumBatch(),
                                safeBatch(config.getTrainingModelSource(), config.getDevice()));
                        results.addAttempt(attemptNumber, batch, null, "out_of_memory");
                        cleanAttempt(attemptDir);
                        log(logConsumer, "Training exceeded available memory before the automatic batch "
                                + "was resolved. Restarting in a fresh Python process with batch " + next + ".");
                        batch = Integer.valueOf(next);
                        continue;
                    }
                    if (previous <= config.getMinimumBatch()) {
                        results.addAttempt(attemptNumber, batch, resolved, "out_of_memory");
                        cleanAttempt(attemptDir);
                        throw e;
                    }
                    int next = Math.max(config.getMinimumBatch(), previous.intValue() / 2);
                    results.addAttempt(attemptNumber, batch, resolved, "out_of_memory");
                    cleanAttempt(attemptDir);
                    log(logConsumer, "Training exceeded available memory with batch " + previous
                            + ". Restarting in a fresh Python process with batch " + next + ".");
                    batch = Integer.valueOf(next);
                }
            }
        } catch (BuildException | ExecutionException e) {
            results.fail("environment_failure", e);
            throw e;
        } catch (TaskException e) {
            if (!cancelRequested) {
                results.fail(isValidationFailure(e) ? "validation_failure" : "backend_failure", e);
            }
            throw e;
        } catch (IOException e) {
            results.fail(dataset == null ? "dataset_failure" : "backend_failure", e);
            throw e;
        } catch (InterruptedException e) {
            if (cancelRequested) {
                results.cancel();
            } else {
                results.fail("backend_failure", e);
            }
            throw e;
        } catch (RuntimeException e) {
            results.fail(dataset == null ? "dataset_failure" : "backend_failure", e);
            throw e;
        } finally {
            finishCancelSignal(cancelFile);
            try {
                results.write(config);
            } catch (IOException e) {
                log(logConsumer, "Could not write training results: " + e.getMessage());
            }
            cleanupStaleAttempts(config.getOutputModelDir());
        }
    }

    /**
     * Performs request cancel.
     */
    public synchronized void requestCancel() {
        cancelRequested = true;
        if (cancelSignalFile != null) {
            try {
                cancelSignalFile.createNewFile();
            } catch (IOException e) {
                killRunningPython();
                return;
            }
            killRunningPythonIfStillAliveLater();
        } else {
            killRunningPython();
        }
    }

    /**
     * Closes resources held by this object.
     */
    public void close() {
        requestCancel();
    }

    private static void validate(YoloTrainingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Training configuration cannot be null.");
        }
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a name for the YOLO model.");
        }
        if (config.getModelName().contains("/") || config.getModelName().contains("\\")
                || config.getModelName().contains("..")) {
            throw new IllegalArgumentException("The YOLO model name cannot contain path separators or '..'.");
        }
        if (config.getDatasetYamlPath() == null || config.getDatasetYamlPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide the YOLO training dataset path.");
        }
        File dataset = new File(config.getDatasetYamlPath());
        if (!dataset.exists()) {
            throw new IllegalArgumentException("The training dataset path does not exist: "
                    + config.getDatasetYamlPath());
        }
        if (config.getEpochs() <= 0) {
            throw new IllegalArgumentException("The number of epochs must be greater than zero.");
        }
        if (config.getImageSize() <= 0) {
            throw new IllegalArgumentException("The YOLO image size must be greater than zero.");
        }
        if (config.isFineTune() && (config.getBaseModelPath() == null || config.getBaseModelPath().trim().isEmpty())) {
            throw new IllegalArgumentException("Please select a base YOLO model for fine tuning.");
        }
        if (!config.isFineTune() && !YoloModelRegistry.isKnownScratchArchitecture(config.getScratchArchitecture())) {
            throw new IllegalArgumentException("Please select a valid YOLO architecture for training from scratch.");
        }
    }

    private static void log(Consumer<String> logConsumer, String message) {
        if (logConsumer != null) {
            logConsumer.accept(message);
        }
    }

    private static File attemptDirectory(YoloTrainingConfig config, int attempt) {
        return new File(config.getOutputModelDir(), ".jdll-attempt-" + attempt);
    }

    private static String canonicalizeAttemptPath(String message, File attemptDir, File outputDir) {
        return message == null ? null
                : message.replace(attemptDir.getAbsolutePath(), outputDir.getAbsolutePath());
    }

    private static int safeBatch(String modelSource, String device) {
        String source = modelSource == null ? "" : modelSource.toLowerCase();
        if ("cuda".equals(device)) {
            if (source.contains("yolo26x")) return 2;
            if (source.contains("yolo26l")) return 4;
            if (source.contains("yolo26m")) return 8;
            return 16;
        }
        if (source.contains("yolo26x") || source.contains("yolo26l")) return 1;
        if (source.contains("yolo26m")) return 2;
        if (source.contains("yolo26s")) return 4;
        return 8;
    }

    private static boolean isOutOfMemory(Throwable error) {
        String text = throwableText(error).toLowerCase();
        return text.contains("out of memory")
                || text.contains("outofmemory")
                || text.contains("cuda error: out of memory")
                || text.contains("mps backend out of memory")
                || text.contains("defaultcpuallocator")
                || text.contains("exit code 137");
    }

    private static boolean isValidationFailure(Throwable error) {
        String text = throwableText(error).toLowerCase();
        return text.contains("validation") || text.contains("validator");
    }

    private static String throwableText(Throwable error) {
        StringBuilder text = new StringBuilder();
        Throwable current = error;
        while (current != null) {
            text.append(' ').append(current.toString());
            if (current.getMessage() != null) {
                text.append(' ').append(current.getMessage());
            }
            current = current.getCause();
        }
        return text.toString();
    }

    private static void promoteAttempt(File attemptDir, File attemptOutput,
            YoloTrainingConfig config) throws IOException {
        File[] children = attemptDir.listFiles();
        if (children != null) {
            for (File child : children) {
                if (child.equals(attemptOutput)) {
                    moveReplacing(child.toPath(), new File(config.getOutputWeightsPath()).toPath());
                } else {
                    moveReplacing(child.toPath(), new File(config.getOutputModelDir(), child.getName()).toPath());
                }
            }
        }
        ensureSelectedModel(config);
        cleanAttempt(attemptDir);
    }

    private static void preserveInterruptedAttempt(File attemptDir, File attemptOutput,
            YoloTrainingConfig config) {
        try {
            promoteAttempt(attemptDir, attemptOutput, config);
        } catch (IOException e) {
            cleanAttempt(attemptDir);
        }
    }

    private static void ensureSelectedModel(YoloTrainingConfig config) throws IOException {
        File selected = new File(config.getOutputWeightsPath());
        if (selected.isFile()) {
            return;
        }
        File weights = new File(config.getOutputModelDir(), "weights");
        File best = new File(weights, "best.pt");
        File last = new File(weights, "last.pt");
        File source = best.isFile() ? best : last;
        if (source.isFile()) {
            Files.copy(source.toPath(), selected.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void moveReplacing(Path source, Path target) throws IOException {
        if (Files.exists(target)) {
            deleteTree(target);
        }
        Files.createDirectories(target.getParent());
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            copyTree(source, target);
            deleteTree(source);
        }
    }

    private static void copyTree(Path source, Path target) throws IOException {
        if (!Files.isDirectory(source)) {
            Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
            return;
        }
        try (Stream<Path> paths = Files.walk(source)) {
            for (Path path : (Iterable<Path>) paths::iterator) {
                Path destination = target.resolve(source.relativize(path));
                if (Files.isDirectory(path)) {
                    Files.createDirectories(destination);
                } else {
                    Files.createDirectories(destination.getParent());
                    Files.copy(path, destination, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    private static void cleanAttempt(File attemptDir) {
        try {
            deleteTree(attemptDir.toPath());
        } catch (IOException e) {
            attemptDir.deleteOnExit();
        }
    }

    private static void cleanupStaleAttempts(File outputDir) {
        File[] stale = outputDir.listFiles(file -> file.isDirectory()
                && file.getName().startsWith(".jdll-attempt-"));
        if (stale != null) {
            for (File attempt : stale) {
                cleanAttempt(attempt);
            }
        }
    }

    private static void deleteTree(Path root) throws IOException {
        if (root == null || !Files.exists(root)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(root)) {
            Path[] ordered = paths.sorted(Comparator.reverseOrder()).toArray(Path[]::new);
            for (Path path : ordered) {
                Files.deleteIfExists(path);
            }
        }
    }

    private synchronized File beginCancelSignal() throws IOException {
        File signal = File.createTempFile("jdll-yolo-cancel-", ".flag");
        Files.deleteIfExists(signal.toPath());
        signal.deleteOnExit();
        cancelSignalFile = signal;
        return signal;
    }

    private synchronized void finishCancelSignal(File signal) {
        if (cancelSignalFile == signal) {
            cancelSignalFile = null;
        }
        if (signal == null) {
            return;
        }
        try {
            Files.deleteIfExists(signal.toPath());
        } catch (IOException e) {
            // Best-effort cleanup of an out-of-process cancellation signal.
        }
    }

    private synchronized void setRunningPython(Service python) {
        runningPython = python;
    }

    private void killRunningPythonIfStillAliveLater() {
        Service python = runningPython;
        if (python == null) {
            return;
        }
        Thread fallback = new Thread(() -> {
            try {
                Thread.sleep(CANCEL_FALLBACK_TIMEOUT_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            synchronized (YoloTrainingService.this) {
                if (python.isAlive()) {
                    if (runningPython == python) {
                        runningPython = null;
                    }
                    python.kill();
                }
            }
        }, "yolo-training-cancel-fallback");
        fallback.setDaemon(true);
        fallback.start();
    }

    private synchronized void killRunningPython() {
        Service python = runningPython;
        runningPython = null;
        if (python != null && python.isAlive()) {
            python.kill();
        }
    }
}
