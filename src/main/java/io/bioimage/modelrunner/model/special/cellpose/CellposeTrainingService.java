package io.bioimage.modelrunner.model.special.cellpose;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer;
import io.bioimage.modelrunner.gui.custom.training.ValidationRequests;
import java.util.function.Consumer;

import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.TaskEvent;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;

/** One training job. Instantiate a new service for each independently cancellable run. */
public final class CellposeTrainingService {
    private final String modelsDir;
    private volatile Task task;
    private volatile ValidationRequests requests;
    private volatile File directory;
    private volatile boolean cancelled;
    private final java.util.concurrent.atomic.AtomicBoolean used = new java.util.concurrent.atomic.AtomicBoolean();

    public CellposeTrainingService(String modelsDir) { this.modelsDir = modelsDir; }

    /** The reserved directory becomes available before dataset preparation starts. */
    public File directory() { return directory; }

    public synchronized void cancel() {
        cancelled = true;
        Task active = task;
        if (active != null) active.cancel();
    }

    public String requestFullValidation() throws IOException {
        ValidationRequests active = requests;
        return active == null || cancelled ? null : active.request();
    }

    private void recordStoppedWorker(Exception failure) throws IOException {
        java.nio.file.Path path = directory.toPath().resolve("run.json");
        com.google.gson.JsonObject run = Files.isRegularFile(path)
                ? com.google.gson.JsonParser.parseString(Files.readString(path)).getAsJsonObject()
                : new com.google.gson.JsonObject();
        if (run.has("status") && !"running".equals(run.get("status").getAsString())) return;
        run.addProperty("status", cancelled ? "cancelled" : "failed");
        if (!cancelled) run.addProperty("error", failure.toString());
        java.nio.file.Path temp = directory.toPath().resolve("worker-status.tmp");
        Files.writeString(temp, run.toString());
        Files.move(temp, path, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    private void checkCancelled() throws InterruptedException {
        if (cancelled || Thread.currentThread().isInterrupted())
            throw new InterruptedException("Cellpose training cancelled");
    }

    private File reserve(String name) throws IOException {
        if (name.trim().isEmpty() || name.contains("..") || name.equals(".")
                || !new File(name).getName().equals(name) || name.matches(".*[\\\\/:*?\"<>|].*"))
            throw new IllegalArgumentException("Invalid Cellpose model name: " + name);
        File root = CellposeModels.familyDirectory(modelsDir);
        Files.createDirectories(root.toPath());
        for (int suffix = 0; ; suffix++) {
            File candidate = new File(root, suffix == 0 ? name : name + "-" + suffix);
            try { Files.createDirectory(candidate.toPath()); return candidate; }
            catch (FileAlreadyExistsException used) { /* Reserve the next name atomically. */ }
        }
    }

    /** Train a single model and return the directory accepted by inference and future fine-tuning. */
    public File train(CellposeTrainingRequest request, Consumer<TaskEvent> listener, Consumer<String> log) throws Exception {
        if (!used.compareAndSet(false, true)) throw new IllegalStateException("This training job has already started.");
        checkCancelled();
        String base = request.baseModel;
        int[] channels = request.channels();
        if (log != null) log.accept("Checking image/mask pairs in " + request.dataset);
        Map<String, java.util.List<java.util.List<String>>> sources = SegmentationDatasetPreparer.discoverSourcePairs(request.dataset);
        try (PixiEnvironmentManager.Operation operation = PixiEnvironmentManager.beginOperation();
             ValidationRequests control = new ValidationRequests()) {
            requests = control;
            PixiEnvironmentSpec spec = Cellpose.resolvePytorchEnv();
            Map<String,Object> config = new LinkedHashMap<>();
            config.put("backend", CellposeModels.NATIVE);
            config.put("dataset", request.dataset.getCanonicalPath());
            config.put("epochs", request.epochs);
            config.put("channels", Arrays.asList(channels[0], channels[1]));
            config.put("device", request.device);
            if (base != null) {
                config.put("_jdll_legacy", io.bioimage.modelrunner.gui.custom.training.TrainingContinuation.legacyMetadata(base));
                config.put("base_model", CellposeModels.OFFICIAL.contains(base) ? base : new File(base).getName());
                config.put("base_weights", CellposeModels.resolve(base, modelsDir, log));
            }
            checkCancelled();
            String init = "import torch\nfrom cellpose import models, train\nfrom jdll_unet import geometry\n"
                    + TrainingCodeUtils.pythonModule("jdll_devices", "python/device_selection.py")
                    + TrainingCodeUtils.pythonModule("jdll_validation_control", "python/validation_control.py")
                    + TrainingCodeUtils.pythonModule("jdll_training_continuation", "python/training_continuation.py")
                + TrainingCodeUtils.pythonModule("jdll_cellpose_data", "python/cellpose_data.py")
                    + TrainingCodeUtils.pythonModule("jdll_cellpose_training", "python/cellpose_training.py");
            try (Service service = PixiEnvironmentManager.pythonService(
                    PixiEnvironmentManager.openEnvironment(spec, log), init).start()) {
                checkCancelled();
                directory = reserve(request.name);
                if (log != null) log.accept("Training directory: " + directory);
                Map<String,Object> inputs = new LinkedHashMap<>();
                inputs.put("config", config);
                inputs.put("output_dir", directory.getAbsolutePath());
                inputs.put("sources", sources);
                inputs.put("request_dir", control.getDirectory());
                Task active = service.task("task.outputs['weights'] = jdll_cellpose_training.train_model(config, output_dir, task, sources, request_dir)", inputs);
                active.listen(event -> {
                    if (event.info != null) control.accept(event.info);
                    if (listener != null) listener.accept(event);
                });
                synchronized (this) { checkCancelled(); task = active; active.start(); }
                active.waitFor();
                if (active.status != Service.TaskStatus.COMPLETE) checkCancelled();
                if (active.status != Service.TaskStatus.COMPLETE)
                    throw new IOException(active.error == null ? "Cellpose training " + active.status : active.error);
                CellposeModels.weights(directory);
                return directory;
            } catch (Exception failure) {
                // The Python handler records ordinary failures. A crashed interpreter cannot do that itself.
                if (directory != null) {
                    try { recordStoppedWorker(failure); }
                    catch (Exception reportingError) { failure.addSuppressed(reportingError); }
                }
                if (cancelled && !(failure instanceof InterruptedException)) {
                    InterruptedException stopped = new InterruptedException("Cellpose training cancelled");
                    stopped.initCause(failure);
                    throw stopped;
                }
                throw failure;
            } finally { task = null; }
        } finally { requests = null; }
    }
}
