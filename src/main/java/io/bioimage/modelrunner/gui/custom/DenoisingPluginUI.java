/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import java.awt.Color;
import java.awt.Window;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingEffort;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingGUI;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingImageIO;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingImageIO.ImageData;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingImages;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingInstaller;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingMethod;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingNoiseStructure;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingService;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSelectionEntry;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;
import io.bioimage.modelrunner.model.special.denoise.DenoisingConfig;
import io.bioimage.modelrunner.model.special.denoise.DenoisingCapabilities;
import io.bioimage.modelrunner.model.special.denoise.DenoisingProgress;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/** Single-panel microscopy denoising UI backed by {@code jdll-denoise}. */
public final class DenoisingPluginUI extends DenoisingGUI {

    private static final long serialVersionUID = 1L;
    private static final Color ERROR_COLOR = new Color(190, 45, 45);

    private final ConsumerInterface consumer;
    private final DenoisingService service = new DenoisingService(new DenoisingInstaller());
    private volatile boolean running;
    private volatile boolean cancelled;
    private Thread worker;
    private File selectedSystemPath;
    private File selectedSystemPreview;
    private RandomAccessibleInterval<?> cachedOriginal;
    private RandomAccessibleInterval<FloatType> cachedDenoised;
    private String cacheKey;
    private boolean cacheIsCompleteInput;
    private Runnable cancelCallback;
    private boolean windowHookInstalled;
    private volatile DenoisingCapabilities capabilities;
    private volatile boolean capabilitiesLoading = true;

    public DenoisingPluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
        super(adapter);
        this.consumer = consumer;
        installListeners();
        if (consumer != null) {
            List<JComponent> components = new ArrayList<JComponent>();
            components.add(getImageSourcePanel().getOpenImagesComboBox());
            components.add(getImageSourcePanel().getFocusButton());
            components.add(getComparisonPanel().getOriginalPanel());
            consumer.setVariableNames(null);
            consumer.setComponents(components);
            consumer.updateGUI();
        }
        updateMethodState();
        updateActionState();
        loadCapabilities();
    }

    public void setCancelCallback(Runnable callback) {
        this.cancelCallback = callback;
    }

    @Override public void addNotify() {
        super.addNotify();
        installWindowCloseHook();
    }

    public void close() {
        cancelled = true;
        service.close();
        if (worker != null) worker.interrupt();
        stopLogTimer();
        running = false;
        updateActionStateOnEdt();
    }

    private void installListeners() {
        YoloImageSourcePanel source = getImageSourcePanel();
        source.getBrowseButton().addActionListener(e -> browse());
        source.setSystemPathDropConsumer(this::selectSystemPath);
        source.getSystemPathField().addActionListener(e -> selectSystemPathFromField());
        source.getSystemPathField().addFocusListener(new FocusAdapter() {
            @Override public void focusLost(FocusEvent e) { selectSystemPathFromField(); }
        });
        source.getSystemImagesRadio().addActionListener(e -> {
            invalidateCache();
            getComparisonPanel().getOriginalPanel().clearImage();
            getComparisonPanel().getDenoisedPanel().clearImage();
            updateActionState();
        });
        source.getOpenImagesRadio().addActionListener(e -> {
            selectedSystemPath = null;
            selectedSystemPreview = null;
            invalidateCache();
            if (consumer != null) consumer.updateGUI();
            updateActionState();
        });
        source.getOpenImagesComboBox().addActionListener(e -> {
            invalidateCache();
            getComparisonPanel().getDenoisedPanel().clearImage();
            updateActionState();
        });
        getOptionsPanel().getMethodComboBox().addActionListener(e -> {
            invalidateCache(); updateMethodState();
        });
        getOptionsPanel().addEffortActionListener(e -> invalidateCache());
        getOptionsPanel().getNoiseStructureComboBox().addActionListener(e -> invalidateCache());
        getAccelerationCheckBox().addActionListener(e -> invalidateCache());
        getAccelerationCheckBox().addPropertyChangeListener("enabled", e -> {
            DenoisingMethod method = getOptionsPanel().getMethod();
            if (method != null && !method.supportsAcceleration()
                    && getAccelerationCheckBox().isEnabled()) {
                getAccelerationCheckBox().setEnabled(false);
                getAccelerationCheckBox().setSelected(false);
            }
        });
        getStrengthPanel().setStrengthConsumer(value -> refreshBlendedPreview());
        getActionPanel().getPreviewButton().addActionListener(e -> start(false));
        getActionPanel().getRunButton().addActionListener(e -> start(true));
        getActionPanel().getCancelButton().addActionListener(e -> cancel());
    }

    private void updateMethodState() {
        DenoisingMethod method = getOptionsPanel().getMethod();
        boolean methodSupportsAcceleration = method != null && method.supportsAcceleration()
                && (capabilities == null || capabilities.supportsDevice(method.getId(), acceleratedDevice()));
        if (!methodSupportsAcceleration) getAccelerationCheckBox().setSelected(false);
        getAccelerationCheckBox().setEnabled(methodSupportsAcceleration
                && getAccelerationCheckBox().isVisible());
        getOptionsPanel().updateNoiseState();
        updateActionState();
    }

    private void loadCapabilities() {
        startLogTimer();
        appendLog("Preparing denoising environment.");
        getStatusPanel().setIdle();
        Thread loader = new Thread(() -> {
            try {
                DenoisingCapabilities detected = service.capabilities(this::showInstallProgress);
                capabilities = detected;
                SwingUtilities.invokeLater(() -> {
                    capabilitiesLoading = false;
                    updateMethodState();
                    appendLog("Denoising environment ready.");
                    stopLogTimer();
                    if (!running) getStatusPanel().setIdle();
                });
            } catch (Throwable error) {
                SwingUtilities.invokeLater(() -> {
                    capabilitiesLoading = false;
                    updateActionState();
                    appendLog("Environment check failed; it will be retried when denoising starts: "
                            + rootMessage(error));
                    stopLogTimer();
                    if (!running) getStatusPanel().setIdle();
                });
            }
        }, "jdll-denoise-capabilities");
        loader.setDaemon(true);
        loader.start();
    }

    private void browse() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        getImageSourcePanel().getSystemImagesRadio().setSelected(true);
        getImageSourcePanel().getSystemPathField().setText(chooser.getSelectedFile().getAbsolutePath());
        selectSystemPath(chooser.getSelectedFile());
    }

    private void selectSystemPathFromField() {
        if (!getImageSourcePanel().getSystemImagesRadio().isSelected()) return;
        String value = getImageSourcePanel().getSystemPathField().getText();
        selectSystemPath(value == null ? null : new File(value.trim()));
    }

    private void selectSystemPath(File source) {
        invalidateCache();
        selectedSystemPath = null;
        selectedSystemPreview = null;
        getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        File preview = source != null && source.isDirectory()
                ? firstImage(source) : source;
        if (preview == null || !YoloImageFiles.canReadImage(preview)) {
            getComparisonPanel().getOriginalPanel().clearImage();
            getComparisonPanel().getOriginalPanel().setEmptyMessage(
                    source != null && source.isDirectory()
                            ? "Folder does not contain valid images" : "Please provide a valid image", ERROR_COLOR);
            getComparisonPanel().getDenoisedPanel().clearImage();
            updateActionState();
            return;
        }
        selectedSystemPath = source;
        selectedSystemPreview = preview;
        getImageSourcePanel().setSystemPathSelectionConfirmed(true);
        getComparisonPanel().getDenoisedPanel().clearImage();
        getStatusPanel().setIdle();
        appendLog("Loading source preview: " + preview.getAbsolutePath());
        Thread loader = new Thread(() -> {
            try {
                ImageData data = DenoisingImageIO.read(preview);
                RandomAccessibleInterval<FloatType> plane = representativePlane(data);
                SwingUtilities.invokeLater(() -> {
                    if (preview.equals(selectedSystemPreview)) {
                        getComparisonPanel().getOriginalPanel().setImage(plane, "Original - " + preview.getName());
                        getComparisonPanel().setOriginalInfo(filePlaneInfo(preview, data));
                        getStatusPanel().setIdle();
                        appendLog("Source preview loaded.");
                    }
                });
            } catch (IOException error) {
                SwingUtilities.invokeLater(() -> showError(error));
            }
        }, "jdll-denoise-preview-loader");
        loader.setDaemon(true);
        loader.start();
        updateActionState();
    }

    private static File firstImage(File folder) {
        List<File> images = YoloImageFiles.readableImagesInDirectory(folder);
        return images.isEmpty() ? null : images.get(0);
    }

    private void start(boolean complete) {
        if (running || !hasSource()) return;
        cancelled = false;
        running = true;
        getStatusPanel().setIdle();
        startLogTimer();
        appendLog(complete ? "Starting denoising." : "Generating denoising preview.");
        appendLog("Method: " + getOptionsPanel().getMethod() + "; effort: "
                + getOptionsPanel().getEffort() + "; device: " + selectedDevice()
                + "; strength: " + String.format(java.util.Locale.US, "%.2f",
                        getStrengthPanel().getStrength()) + ".");
        updateActionState();
        worker = new Thread(() -> {
            try {
                if (complete) runComplete(); else runPreview();
                if (!cancelled) SwingUtilities.invokeLater(() -> {
                    getStatusPanel().setComplete();
                    appendLog(complete ? "Denoising finished." : "Denoising preview ready.");
                });
            } catch (Throwable error) {
                if (!cancelled) SwingUtilities.invokeLater(() -> showError(error));
            } finally {
                stopLogTimer();
                running = false;
                updateActionStateOnEdt();
            }
        }, complete ? "jdll-denoise-run" : "jdll-denoise-preview");
        worker.start();
    }

    private void cancel() {
        if (!running) {
            if (cancelCallback != null) cancelCallback.run();
            return;
        }
        cancelled = true;
        service.cancel();
        if (worker != null) worker.interrupt();
        getStatusPanel().setIdle();
        appendLog("Denoising cancelled.");
        stopLogTimer();
    }

    private void runPreview() throws Exception {
        if (usesSystemSource()) {
            ImageData data = DenoisingImageIO.read(selectedSystemPreview);
            RandomAccessibleInterval<FloatType> original = representativePlane(data);
            String key = cacheKey(fileCacheId(selectedSystemPreview), "xy");
            RandomAccessibleInterval<FloatType> denoised = service.run(config("xy"), original, true,
                    this::showProgress, this::showInstallProgress);
            setPreviewCache(original, denoised, key, data.getPages() == 1 && data.getChannels() == 1);
            return;
        }
        Object image = selectedOpenImage();
        RandomAccessibleInterval<?> complete = convert(image);
        String axes = DenoisingImages.axesFor(complete.numDimensions());
        RandomAccessibleInterval<?> original = previewPlane(complete, axes,
                consumer.getImageChannelPosition(image), consumer.getImageZPosition(image),
                consumer.getImageTimePosition(image));
        String key = cacheKey(openImageCacheId(image), "xy");
        SwingUtilities.invokeLater(() -> getComparisonPanel().setOriginalInfo(openPlaneInfo(image, complete, axes)));
        RandomAccessibleInterval<FloatType> denoised = runService(config("xy"), original, true);
        setPreviewCache(original, denoised, key, complete.numDimensions() == 2);
    }

    private void runComplete() throws Exception {
        if (usesSystemSource()) {
            runSystemSource();
            return;
        }
        Object image = selectedOpenImage();
        RandomAccessibleInterval<?> original = convert(image);
        String axes = DenoisingImages.axesFor(original.numDimensions());
        String key = cacheKey(openImageCacheId(image), axes);
        RandomAccessibleInterval<FloatType> denoised = cacheIsCompleteInput && key.equals(cacheKey)
                ? cachedDenoised : runService(config(axes), original, false);
        RandomAccessibleInterval<?> blended = blend(original, denoised, getStrengthPanel().getStrength());
        String name = selectedOpenTitle() + "_denoised";
        display(blended, axes, name);
    }

    private void runSystemSource() throws Exception {
        List<File> files = DenoisingImageIO.inputFiles(selectedSystemPath);
        if (files.isEmpty()) throw new IOException("No readable images found.");
        File outputDirectory = selectedSystemPath.isDirectory()
                ? DenoisingImageIO.outputDirectory(selectedSystemPath) : null;
        File workingDirectory = outputDirectory == null ? null
                : DenoisingImageIO.temporaryDirectoryFor(outputDirectory);
        boolean published = outputDirectory == null;
        try {
            if (workingDirectory != null && !workingDirectory.mkdirs()) {
                throw new IOException("Could not create temporary output directory: " + workingDirectory);
            }
            for (int i = 0; i < files.size(); i++) {
                if (cancelled || Thread.currentThread().isInterrupted()) return;
                File source = files.get(i);
                final int index = i + 1;
                SwingUtilities.invokeLater(() -> getStatusPanel().setProgress(index - 1, files.size()));
                appendLog("Processing image " + index + "/" + files.size() + ": " + source.getName());
                ImageData data = DenoisingImageIO.read(source);
                DenoisingConfig config = config(data.getAxes());
                String key = cacheKey(fileCacheId(source), data.getAxes());
                RandomAccessibleInterval<FloatType> denoised = cacheIsCompleteInput && key.equals(cacheKey)
                        ? cachedDenoised : service.run(config, data.getImage(), false,
                                this::showProgress, this::showInstallProgress);
                RandomAccessibleInterval<FloatType> blended = DenoisingImages.blend(
                        data.getImage(), denoised, getStrengthPanel().getStrength());
                File output = outputDirectory == null ? DenoisingImageIO.outputFile(source)
                        : new File(workingDirectory, source.getName());
                DenoisingImageIO.write(output, blended, data);
                File reportedOutput = outputDirectory == null ? output : new File(outputDirectory, source.getName());
                DenoisingImageIO.writeSidecar(output, reportedOutput, source,
                        getStrengthPanel().getStrength(), config.toMap(), service.getMetadata());
                appendLog("Saved denoised image: " + reportedOutput.getAbsolutePath());
            }
            if (workingDirectory != null) {
                DenoisingImageIO.publishDirectory(workingDirectory, outputDirectory);
                published = true;
                appendLog("Published denoised folder: " + outputDirectory.getAbsolutePath());
            }
        } finally {
            if (!published) DenoisingImageIO.deleteRecursively(workingDirectory);
        }
    }

    private void setPreviewCache(RandomAccessibleInterval<?> original,
            RandomAccessibleInterval<FloatType> denoised, String key, boolean complete) {
        cachedOriginal = original;
        cachedDenoised = denoised;
        cacheKey = key;
        cacheIsCompleteInput = complete;
        SwingUtilities.invokeLater(() -> {
            setOriginalPreview(original);
            getComparisonPanel().setDenoisedInfo(getComparisonPanelInfo());
            refreshBlendedPreview();
            updateActionState();
        });
    }

    private void refreshBlendedPreview() {
        if (cachedOriginal == null || cachedDenoised == null) return;
        RandomAccessibleInterval<?> blended = blend(cachedOriginal, cachedDenoised,
                getStrengthPanel().getStrength());
        getComparisonPanel().setDenoisedInfo(getComparisonPanelInfo());
        setDenoisedPreview(blended);
    }

    private void showProgress(DenoisingProgress progress) {
        if (progress == null) return;
        SwingUtilities.invokeLater(() -> {
            getStatusPanel().setProgress(progress.getCurrent(), progress.getMaximum());
            appendLog(progressMessage(progress));
        });
    }

    private void showInstallProgress(String message) {
        SwingUtilities.invokeLater(() -> {
            getStatusPanel().setProgress(0, 0);
            appendLog(message);
        });
    }

    private void startLogTimer() {
        Runnable start = () -> getLogPanel().startRunTimer();
        if (SwingUtilities.isEventDispatchThread()) {
            start.run();
            return;
        }
        try {
            SwingUtilities.invokeAndWait(start);
        } catch (Exception error) {
            SwingUtilities.invokeLater(start);
        }
    }

    private void stopLogTimer() {
        Runnable stop = () -> getLogPanel().stopRunTimer();
        if (SwingUtilities.isEventDispatchThread()) stop.run();
        else SwingUtilities.invokeLater(stop);
    }

    private void appendLog(String message) {
        if (message == null || message.trim().isEmpty()) return;
        Runnable append = () -> {
            String clean = message.replaceAll("\\u001B\\[[;\\d]*m", "");
            for (String line : clean.split("\\R")) {
                if (!line.trim().isEmpty()) getLogPanel().appendHtml(escapeHtml(line.trim()));
            }
        };
        if (SwingUtilities.isEventDispatchThread()) append.run();
        else SwingUtilities.invokeLater(append);
    }

    private static String progressMessage(DenoisingProgress progress) {
        if (progress.getMaximum() > 0L) {
            if ("optimization".equals(progress.getPhase())) {
                return "Iteration " + progress.getCurrent() + "/" + progress.getMaximum() + " completed.";
            }
            if ("tiling".equals(progress.getPhase())) {
                return "Patch " + progress.getCurrent() + "/" + progress.getMaximum() + " completed.";
            }
        }
        String message = progress.getMessage();
        return message == null || message.trim().isEmpty()
                ? "Denoising phase: " + progress.getPhase() + "." : message;
    }

    private static String escapeHtml(String value) {
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    private DenoisingConfig config(String axes) {
        DenoisingMethod method = getOptionsPanel().getMethod();
        DenoisingEffort effort = getOptionsPanel().getEffort();
        DenoisingNoiseStructure structure = getOptionsPanel().getNoiseStructure();
        return DenoisingConfig.builder()
                .method(method.getId()).effort(effort.getId()).axes(axes)
                .noiseStructure(structure.getId()).device(selectedDevice()).build();
    }

    private String selectedDevice() {
        if (!getAccelerationCheckBox().isEnabled() || !getAccelerationCheckBox().isSelected()) return "cpu";
        String text = getAccelerationCheckBox().getText().toLowerCase();
        return text.contains("mps") ? "mps" : "cuda";
    }

    private String acceleratedDevice() {
        return getAccelerationCheckBox().getText().toLowerCase().contains("mps") ? "mps" : "cuda";
    }

    private String cacheKey(String source, String axes) {
        return source + "|" + axes + "|" + config(axes).toMap().toString();
    }

    private void invalidateCache() {
        cachedOriginal = null;
        cachedDenoised = null;
        cacheKey = null;
        cacheIsCompleteInput = false;
    }

    private boolean usesSystemSource() {
        return getImageSourcePanel().getSystemImagesRadio().isSelected();
    }

    private boolean hasSource() {
        return usesSystemSource() ? selectedSystemPath != null : selectedOpenImage() != null;
    }

    private Object selectedOpenImage() {
        Object selected = getImageSourcePanel().getOpenImagesComboBox().getSelectedItem();
        return selected instanceof YoloImageSelectionEntry ? ((YoloImageSelectionEntry) selected).getImage() : null;
    }

    private String selectedOpenTitle() {
        Object selected = getImageSourcePanel().getOpenImagesComboBox().getSelectedItem();
        return selected instanceof YoloImageSelectionEntry ? ((YoloImageSelectionEntry) selected).getTitle() : "image";
    }

    private String openImageId() {
        Object selected = getImageSourcePanel().getOpenImagesComboBox().getSelectedItem();
        return selected instanceof YoloImageSelectionEntry ? ((YoloImageSelectionEntry) selected).getId() : "";
    }

    private String openImageCacheId(Object image) {
        return openImageId() + "@" + System.identityHashCode(image)
                + ":c" + consumer.getImageChannelPosition(image)
                + ":z" + consumer.getImageZPosition(image)
                + ":t" + consumer.getImageTimePosition(image);
    }

    private static String fileCacheId(File file) {
        return file.getAbsolutePath() + ":" + file.length() + ":" + file.lastModified();
    }

    private void updateActionState() {
        DenoisingMethod selectedMethod = getOptionsPanel().getMethod();
        boolean methodAvailable = capabilities == null || selectedMethod == null
                || capabilities.isMethodAvailable(selectedMethod.getId());
        boolean valid = hasSource() && methodAvailable && !capabilitiesLoading;
        getActionPanel().getCancelButton().setEnabled(running || cancelCallback != null);
        getActionPanel().getPreviewButton().setEnabled(valid && !running);
        getActionPanel().getRunButton().setEnabled(valid && !running);
        getImageSourcePanel().setInteractionEnabled(!running);
        getOptionsPanel().setInteractionEnabled(!running);
        getStrengthPanel().setInteractionEnabled(!running);
        getAccelerationCheckBox().setEnabled(!running && selectedMethod != null
                && selectedMethod.supportsAcceleration()
                && (capabilities == null
                        || capabilities.supportsDevice(selectedMethod.getId(), acceleratedDevice())));
    }

    private void updateActionStateOnEdt() {
        SwingUtilities.invokeLater(this::updateActionState);
    }

    private void showError(Throwable error) {
        getStatusPanel().setIdle();
        appendLog("Denoising failed: " + rootMessage(error));
        stopLogTimer();
        JOptionPane.showMessageDialog(this, rootMessage(error), "Denoising failed", JOptionPane.ERROR_MESSAGE);
    }

    private void installWindowCloseHook() {
        if (windowHookInstalled) return;
        Window window = SwingUtilities.getWindowAncestor(this);
        if (window == null) return;
        window.addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent e) { close(); }
            @Override public void windowClosed(WindowEvent e) { close(); }
        });
        windowHookInstalled = true;
    }

    private static String rootMessage(Throwable error) {
        Throwable cause = error;
        while (cause.getCause() != null) cause = cause.getCause();
        return cause.getMessage() == null ? cause.toString() : cause.getMessage();
    }

    private static String filePlaneInfo(File file, ImageData data) {
        long z = data.getPages() > 1 ? data.getPages() / 2L + 1L : 1L;
        return file.getName() + " | C 1/" + data.getChannels() + " | Z " + z + "/"
                + data.getPages() + " | T 1/1";
    }

    private String openPlaneInfo(Object image, RandomAccessibleInterval<?> rai, String axes) {
        long channels = axisSize(rai, axes, 'c');
        long slices = axisSize(rai, axes, 'z');
        long frames = axisSize(rai, axes, 't');
        return selectedOpenTitle() + " | C " + (consumer.getImageChannelPosition(image) + 1L) + "/" + channels
                + " | Z " + (consumer.getImageZPosition(image) + 1L) + "/" + slices
                + " | T " + (consumer.getImageTimePosition(image) + 1L) + "/" + frames;
    }

    private String getComparisonPanelInfo() {
        DenoisingMethod method = getOptionsPanel().getMethod();
        return "Denoised | " + (method == null ? "" : method.toString())
                + " | strength " + String.format(java.util.Locale.US, "%.2f", getStrengthPanel().getStrength());
    }

    private static long axisSize(RandomAccessibleInterval<?> image, String axes, char axis) {
        int index = axes.indexOf(axis);
        return index < 0 ? 1L : image.dimension(index);
    }

    private static RandomAccessibleInterval<FloatType> representativePlane(ImageData data) {
        long z = data.getPages() > 1 ? data.getPages() / 2L : 0L;
        return DenoisingImages.previewPlane(data.getImage(), data.getAxes(), 0L, z, 0L);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private RandomAccessibleInterval<FloatType> runService(DenoisingConfig config,
            RandomAccessibleInterval<?> image, boolean preview) throws Exception {
        return service.run(config, (RandomAccessibleInterval) image, preview,
                this::showProgress, this::showInstallProgress);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private RandomAccessibleInterval<?> convert(Object image) {
        return consumer.convertIntoRai(image);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static RandomAccessibleInterval<?> blend(RandomAccessibleInterval<?> original,
            RandomAccessibleInterval<FloatType> denoised, double strength) {
        return DenoisingImages.blend((RandomAccessibleInterval) original, denoised, strength);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static RandomAccessibleInterval<?> previewPlane(RandomAccessibleInterval<?> image,
            String axes, long channel, long z, long time) {
        return DenoisingImages.previewPlane((RandomAccessibleInterval) image, axes, channel, z, time);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void display(RandomAccessibleInterval<?> image, String axes, String name) {
        consumer.displayImage((RandomAccessibleInterval) image, axes, name);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void setOriginalPreview(RandomAccessibleInterval<?> image) {
        getComparisonPanel().getOriginalPanel().setImage((RandomAccessibleInterval) image, "Original");
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void setDenoisedPreview(RandomAccessibleInterval<?> image) {
        getComparisonPanel().getDenoisedPanel().setImage((RandomAccessibleInterval) image, "Denoised preview");
        getComparisonPanel().getDenoisedPanel().setViewport(
                getComparisonPanel().getOriginalPanel().getViewport());
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("JDLL Denoising");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.add(new DenoisingPluginUI(null, null));
            frame.setSize(900, 720);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
