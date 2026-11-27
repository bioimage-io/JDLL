package io.bioimage.modelrunner.gui.custom.cellpose;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;

import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.TaskEvent;
import io.bioimage.modelrunner.model.special.cellpose.CellposeTrainingService;
import io.bioimage.modelrunner.model.special.cellpose.CellposeTrainingRequest;
import io.bioimage.modelrunner.gui.custom.gui.CellposeGUI;
import io.bioimage.modelrunner.model.python.envs.EnvironmentSetupLog;

/** Adapts structured native training events to the shared DeepIcy training views. */
public final class CellposeTrainingController {
    private final CellposeGUI gui;
    private final CellposeTrainPanel panel;
    private final String modelsDir;
    private final Supplier<String> device;
    private final BooleanSupplier inferenceRunning;
    private final Consumer<File> onSaved;
    private final Runnable releaseInference;
    private final Timer timer;
    private CellposeTrainingService current;
    private boolean stopping;
    private boolean errorReported;
    private long startTime, firstStepTime;
    private int step, firstStep, total, epochs;

    public CellposeTrainingController(CellposeGUI gui, String modelsDir, Supplier<String> device,
            BooleanSupplier inferenceRunning, Runnable releaseInference, Consumer<File> onSaved) {
        this.gui = gui;
        this.panel = gui.getTrainPanel();
        this.modelsDir = modelsDir;
        this.device = device;
        this.inferenceRunning = inferenceRunning;
        this.onSaved = onSaved;
        this.releaseInference = releaseInference;
        timer = new Timer(1000, e -> status(true));
        panel.getTrainActionPanel().getRunButton().addActionListener(e -> start());
        panel.getTrainActionPanel().getCancelButton().addActionListener(e -> cancel());
        panel.getFullValidationButton().addActionListener(e -> {
            if (current == null) return;
            try {
                String id = current.requestFullValidation();
                if (id != null) panel.fullValidationRequested(id);
            } catch (IOException error) { log(error.getMessage()); }
        });
    }

    public boolean running() { return current != null; }

    private void start() {
        if (running() || inferenceRunning.getAsBoolean() || !panel.validateTrainingFields()) return;
        int[] channels;
        try { channels = panel.channels(); }
        catch (java.text.ParseException e) {
            JOptionPane.showMessageDialog(panel, "Channel numbers must be nonnegative integers.");
            return;
        }
        final String modelName = panel.getModelNameField().getText().trim();
        final File dataset = new File(panel.getDatasetField().getText().trim());
        final String base = panel.getFineTuneRadio().isSelected() ? panel.getSelectedBaseModelValue() : null;
        final int count = Integer.parseInt(panel.getEpochsField().getText().trim());
        final String selectedDevice = device.get();
        final CellposeTrainingRequest request;
        try {
            request = new CellposeTrainingRequest(modelName, dataset, count, channels, selectedDevice, base);
        } catch (IllegalArgumentException error) {
            JOptionPane.showMessageDialog(panel, error.getMessage(), "Cellpose training", JOptionPane.ERROR_MESSAGE);
            return;
        }
        CellposeTrainingService job = new CellposeTrainingService(modelsDir);
        current = job;
        stopping = false;
        errorReported = false;
        step = total = 0;
        epochs = count;
        firstStepTime = 0;
        startTime = System.currentTimeMillis();
        panel.getTrainingLogPanel().clearLog();
        panel.getLossGraphPanel().clearValues();
        panel.getMetricGraphPanel().clearValues();
        panel.getValidationPreviewPanel().clearPreview();
        panel.setTrainingRunning(true);
        panel.setFullValidationAvailable(false);
        gui.getTabs().setEnabledAt(0, false);
        gui.getAccelerationCheckBox().setEnabled(false);
        status(true);
        timer.start();
        log("Preparing Cellpose " + (base == null ? "scratch training" : "fine-tuning") + " for " + modelName + ".");
        new Thread(() -> {
            try (EnvironmentSetupLog.Scope setup = panel.getTrainingLogPanel().openEnvironmentLog()) {
                releaseInference.run();
                File result = job.train(request,
                        event -> dispatch(job, () -> update(event)),
                        message -> dispatch(job, () -> {
                            File dir = job.directory();
                            File file = panel.getTrainingLogPanel().getLogFile();
                            if (dir != null && (file == null || !dir.equals(file.getParentFile())))
                                panel.getTrainingLogPanel().startDiskLog(dir);
                            log(message);
                        }));
                dispatch(job, () -> {
                    log("Training completed. Saved model in " + result);
                    onSaved.accept(result);
                });
            } catch (Exception e) {
                dispatch(job, () -> {
                    File stopped = job.directory();
                    if (stopped != null && io.bioimage.modelrunner.gui.custom.training.TrainingContinuation.find(stopped.getAbsolutePath()) != null)
                        onSaved.accept(stopped);
                    if (!errorReported) log(stopping ? "Training cancelled." : "Training failed: " + e.getMessage());
                });
            } finally {
                dispatch(job, () -> {
                    timer.stop();
                    panel.getTrainingLogPanel().closeDiskLog();
                    status(false);
                    current = null;
                    panel.setTrainingRunning(false);
                    gui.getTabs().setEnabledAt(0, true);
                    gui.getAccelerationCheckBox().setEnabled(true);
                });
            }
        }, "Cellpose training").start();
    }

    private void dispatch(CellposeTrainingService job, Runnable action) {
        SwingUtilities.invokeLater(() -> { if (current == job) action.run(); });
    }

    public void cancel() {
        if (current == null || stopping) return;
        stopping = true;
        current.cancel();
        log("Stopping Cellpose training.");
        panel.getTrainActionPanel().getCancelButton().setEnabled(false);
    }

    public void close() {
        cancel();
        current = null;
        timer.stop();
        panel.getTrainingLogPanel().closeDiskLog();
    }

    private void log(String message) { panel.getTrainingLogPanel().appendLine(message); }

    private void update(TaskEvent event) {
        if (event.responseType != ResponseType.UPDATE) return;
        if (event.message != null && !event.message.isEmpty()) log(event.message);
        Map<String,Object> info = event.info;
        if (info == null) return;
        panel.handleValidationEvent(info);
        Object type = info.get("type");
        if ("training_error".equals(type)) {
            errorReported = true;
        } else if ("preview_unavailable".equals(type)) {
            panel.getValidationPreviewPanel().showUnavailable((String) info.get("reason"));
        } else if ("preview".equals(type)) {
            panel.getValidationPreviewPanel().loadPreview((String) info.get("preview_path"));
        } else if ("progress".equals(type)) {
            step = ((Number) info.get("step")).intValue();
            total = ((Number) info.get("total_steps")).intValue();
            epochs = ((Number) info.get("total_epochs")).intValue();
            int epoch = ((Number) info.get("epoch")).intValue();
            if (firstStepTime == 0) {
                firstStep = step;
                firstStepTime = System.currentTimeMillis();
            }
            Number train = value(info.get("losses"), "train/total_loss");
            Number val = value(info.get("losses"), "val/total_loss");
            Number pq = value(info.get("metrics"), "val/panoptic_quality");
            if (train != null && val == null)
                panel.getLossGraphPanel().addTrainValue("Cellpose loss", step, epoch, train.doubleValue());
            if (val != null) panel.getLossGraphPanel().addValidationValue("Cellpose loss", step, epoch, val.doubleValue());
            if (pq != null) panel.getMetricGraphPanel().addValidationValue("Panoptic quality", step, epoch, pq.doubleValue());
            status(true);
        }
    }

    private static Number value(Object map, String key) {
        Object value = map instanceof Map ? ((Map<?,?>) map).get(key) : null;
        return value instanceof Number ? (Number) value : null;
    }

    private void status(boolean active) {
        long now = System.currentTimeMillis(), elapsed = now - startTime;
        double seconds = firstStepTime != 0 && step > firstStep ? (now - firstStepTime) / (1000.0 * (step - firstStep)) : Double.NaN;
        panel.getTrainingLogPanel().setTrainingStatus(active, step, total, epochs, elapsed, seconds);
        panel.getLossGraphPanel().setTrainingStatus(active, step, total, epochs, elapsed, seconds);
        panel.getMetricGraphPanel().setTrainingStatus(active, step, total, epochs, elapsed, seconds);
        panel.getValidationPreviewPanel().setTrainingStatus(active, step, total, epochs, elapsed, seconds);
    }
}
