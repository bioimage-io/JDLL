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
package io.bioimage.modelrunner.gui.custom;

import java.awt.Color;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.unet.UnetGUI;
import io.bioimage.modelrunner.gui.custom.unet.UnetInferenceService;
import io.bioimage.modelrunner.gui.custom.unet.UnetInstaller;
import io.bioimage.modelrunner.gui.custom.unet.UnetModelRegistry;
import io.bioimage.modelrunner.gui.custom.unet.UnetTrainingConfig;
import io.bioimage.modelrunner.gui.custom.unet.UnetTrainingService;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSelectionEntry;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class UNetPluginUI extends UnetGUI implements ActionListener {

    private static final long serialVersionUID = -1498287706411357145L;
    private static final String DEFAULT_PREVIEW_MESSAGE = "Preview will appear here";
    private static final String SYSTEM_PREVIEW_PROMPT = "Please select an image/folder from the file system";
    private static final String INVALID_IMAGE_MESSAGE = "Please provide a valid image file";
    private static final String EMPTY_FOLDER_MESSAGE = "Folder does not contain valid images";
    private static final Color PREVIEW_ERROR_COLOR = new Color(210, 40, 40);
    private static final String APPOSE_STREAM_CLOSED = "java.io.IOException: Stream closed";
    private static final String UNET_MASK_SUFFIX = "_unet_labels";

    private final ConsumerInterface consumer;
    private final UnetInstaller installer = new UnetInstaller();
    private final UnetInferenceService inferenceService = new UnetInferenceService(installer);
    private final UnetTrainingService trainingService = new UnetTrainingService(installer);
    private volatile boolean cancelled;
    private Timer trainingTimer;
    private long trainingStartMillis;
    private long lastProgressMillis;
    private int lastProgressStep;
    private int currentTrainingStep;
    private int totalTrainingSteps;
    private int totalTrainingEpochs;
    private double currentSecondsPerStep = Double.NaN;
    private final Deque<Double> secondsPerStepSamples = new ArrayDeque<Double>();
    private File selectedSystemPath;
    private File selectedSystemImageFile;
    private volatile boolean inferenceRunning;
    private volatile boolean trainingRunning;
    private long trainingUiRunId;
    private int selectedTabIndex;
    private boolean revertingTabSelection;
    private boolean windowCloseHookInstalled;
    private Thread workerThread;
    private int lastLoggedEpochStart;
    private int lastLoggedTrainEpoch;
    private int lastLoggedValidationEpoch;
    private int lastLoggedPreviewEpoch;

    /**
     * Creates a new UNetPluginUI instance.
     *
     * @param consumer the consumer callback.
     * @param adapter the adapter.
     */
    public UNetPluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
        super(adapter);
        this.consumer = consumer;

        String modelsDir = consumer == null ? null : consumer.getModelsDir();
        LinkedHashMap<String, String> unetModelEntries = UnetModelRegistry.buildModelEntries(modelsDir);
        this.inferencePanel.getModelSelectionPanel().setModels(unetModelEntries);
        this.trainPanel.setBaseModels(unetModelEntries);

        this.inferencePanel.getModelSelectionPanel().getBrowseButton().addActionListener(e -> browseInferenceModel());
        this.inferencePanel.getActionPanel().getRunButton().addActionListener(this);
        this.inferencePanel.getActionPanel().getCancelButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getRunButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getCancelButton().addActionListener(this);
        installInferenceSourceListeners();
        installTabLifecycleListener();

        if (this.consumer == null) {
            return;
        }
        List<JComponent> componentList = new ArrayList<JComponent>();
        this.consumer.setVariableNames(null);
        componentList.add(this.inferencePanel.getModelSelectionPanel().getModelComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getOpenImagesComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getFocusButton());
        componentList.add(this.inferencePanel.getImageDisplayPanel());
        this.consumer.setComponents(componentList);
        this.consumer.updateGUI();
    }

    /**
     * Executes close.
     */
    public void close() {
        cancelled = true;
        trainingService.close();
        inferenceService.close();
        if (workerThread != null && workerThread.isAlive()) {
            workerThread.interrupt();
        }
        if (trainingTimer != null) {
            trainingTimer.stop();
            trainingTimer = null;
        }
        inferenceRunning = false;
        trainingRunning = false;
        updateTabLocks();
    }

    /**
     * Performs add notify.
     */
    @Override
    public void addNotify() {
        super.addNotify();
        installWindowCloseHook();
    }

    private void installWindowCloseHook() {
        if (windowCloseHookInstalled) {
            return;
        }
        java.awt.Window window = SwingUtilities.getWindowAncestor(this);
        if (window == null) {
            return;
        }
        window.addWindowListener(new WindowAdapter() {
            /**
             * Performs window closing.
             *
             * @param e the e.
             */
            @Override
            public void windowClosing(WindowEvent e) {
                close();
            }

            /**
             * Performs window closed.
             *
             * @param e the e.
             */
            @Override
            public void windowClosed(WindowEvent e) {
                close();
            }
        });
        windowCloseHookInstalled = true;
    }

    private void installTabLifecycleListener() {
        tabs.addChangeListener(e -> {
            if (revertingTabSelection) {
                return;
            }
            int selected = tabs.getSelectedIndex();
            if ((trainingRunning && selected == 0) || (inferenceRunning && selected == 1)) {
                revertingTabSelection = true;
                tabs.setSelectedIndex(selectedTabIndex);
                revertingTabSelection = false;
                return;
            }
            selectedTabIndex = selected;
            if (selected == 0 || selected == 1) {
                inferenceService.close();
            }
        });
    }

    private void startInferenceUiState() {
        inferenceRunning = true;
        updateTabLocks();
    }

    private void finishInferenceUiState() {
        inferenceRunning = false;
        updateTabLocks();
    }

    private void updateTabLocks() {
        tabs.setEnabledAt(0, !trainingRunning);
        tabs.setEnabledAt(1, !inferenceRunning);
    }

    private void installInferenceSourceListeners() {
        YoloImageSourcePanel sourcePanel = inferencePanel.getImageSourcePanel();
        sourcePanel.getSystemImagesRadio().addActionListener(e -> showSystemPathPrompt());
        sourcePanel.getOpenImagesRadio().addActionListener(e -> showOpenImageSource());
        sourcePanel.getSystemPathField().addActionListener(e -> updateSystemPathPreviewFromField());
        sourcePanel.getSystemPathField().addFocusListener(new FocusAdapter() {
            /**
             * Performs focus lost.
             *
             * @param e the e.
             */
            @Override
            public void focusLost(FocusEvent e) {
                updateSystemPathPreviewFromField();
            }
        });
        sourcePanel.setSystemPathDropConsumer(file -> updateSystemPathPreview(file));
        sourcePanel.getBrowseButton().addActionListener(e -> browseSystemImagePath());
    }

    private void browseInferenceModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.setFileFilter(new FileNameExtensionFilter("UNet weights (*.pt, *.pth)", "pt", "pth"));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        if (selected == null) {
            return;
        }
        File modelFile = selected.isDirectory() ? UnetModelRegistry.findModelFile(selected) : selected;
        String label = selected.isDirectory() ? "[Custom] " + selected.getName()
                : "[Custom] " + UnetModelRegistry.removeWeightsExtension(selected.getName());
        inferencePanel.getModelSelectionPanel().addOrSelectModel(label,
                modelFile == null ? selected.getAbsolutePath() : modelFile.getAbsolutePath());
    }

    private void showSystemPathPrompt() {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        inferencePanel.getImageDisplayPanel().setEmptyMessage(SYSTEM_PREVIEW_PROMPT);
        inferencePanel.getImageDisplayPanel().clearImage();
        inferencePanel.updateImageActionState();
    }

    private void showOpenImageSource() {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        inferencePanel.getImageDisplayPanel().setEmptyMessage(DEFAULT_PREVIEW_MESSAGE);
        if (consumer != null) {
            consumer.updateGUI();
        } else {
            inferencePanel.getImageDisplayPanel().clearImage();
        }
        inferencePanel.updateImageActionState();
    }

    private void browseSystemImagePath() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.setAcceptAllFileFilterUsed(true);
        int result = chooser.showOpenDialog(this);
        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        inferencePanel.getImageSourcePanel().getSystemImagesRadio().setSelected(true);
        inferencePanel.getImageSourcePanel().getSystemPathField().setText(selected.getAbsolutePath());
        updateSystemPathPreview(selected);
    }

    private void updateSystemPathPreviewFromField() {
        if (!inferencePanel.getImageSourcePanel().getSystemImagesRadio().isSelected()) {
            return;
        }
        String text = inferencePanel.getImageSourcePanel().getSystemPathField().getText();
        if (text == null || text.trim().isEmpty()) {
            showSystemPathPrompt();
            return;
        }
        updateSystemPathPreview(new File(text.trim()));
    }

    private void updateSystemPathPreview(File path) {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        if (path == null) {
            showPreviewError(INVALID_IMAGE_MESSAGE);
            inferencePanel.updateImageActionState();
            return;
        }
        if (path.isDirectory()) {
            File previewImage = YoloImageFiles.previewImageInDirectory(path);
            if (previewImage == null) {
                showPreviewError(EMPTY_FOLDER_MESSAGE);
                inferencePanel.updateImageActionState();
                return;
            }
            selectedSystemPath = path;
            showSystemPreviewImage(previewImage);
            return;
        }
        if (!YoloImageFiles.canReadImage(path)) {
            showPreviewError(INVALID_IMAGE_MESSAGE);
            inferencePanel.updateImageActionState();
            return;
        }
        selectedSystemPath = path;
        showSystemPreviewImage(path);
    }

    private void showSystemPreviewImage(File imageFile) {
        try {
            inferencePanel.getImageDisplayPanel().setImageFile(imageFile, imageFile.getName(), true);
            selectedSystemImageFile = imageFile;
            inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(true);
        } catch (IOException e) {
            selectedSystemPath = null;
            selectedSystemImageFile = null;
            inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
            showPreviewError(INVALID_IMAGE_MESSAGE);
        }
        inferencePanel.updateImageActionState();
    }

    private void showPreviewError(String message) {
        inferencePanel.getImageDisplayPanel().setEmptyMessage(message, PREVIEW_ERROR_COLOR);
        inferencePanel.getImageDisplayPanel().clearImage();
    }

    /**
     * Executes action performed.
     *
     * @param e the e parameter.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == this.inferencePanel.getActionPanel().getRunButton()) {
            cancelled = false;
            startInferenceUiState();
            workerThread = new Thread(() -> {
                try {
                    runUnet();
                } catch (Exception e1) {
                    if (!cancelled) {
                        e1.printStackTrace();
                    }
                } finally {
                    SwingUtilities.invokeLater(this::finishInferenceUiState);
                }
            }, "unet-inference");
            workerThread.start();
        } else if (e.getSource() == this.inferencePanel.getActionPanel().getCancelButton()) {
            cancel();
        } else if (e.getSource() == this.trainPanel.getTrainActionPanel().getRunButton()) {
            trainUnet();
        } else if (e.getSource() == this.trainPanel.getTrainActionPanel().getCancelButton()) {
            cancel();
        }
    }

    private void cancel() {
        cancelled = true;
        if (trainingRunning) {
            trainingService.close();
            finishCancelledTrainingUiState();
            return;
        }
        if (inferenceRunning) {
            inferenceService.cancelCurrentInference();
        } else if (workerThread != null && workerThread.isAlive()) {
            workerThread.interrupt();
        }
    }

    private void runUnet()
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        if (consumer != null) {
            consumer.notifyParams(null);
        }
        String modelPath = this.inferencePanel.getModelSelectionPanel().getSelectedModelValue();
        if (this.inferencePanel.getImageSourcePanel().getSystemImagesRadio().isSelected()) {
            runUnetOnSystemImage(modelPath);
            return;
        }
        runUnetOnOpenImage(modelPath);
    }

    private <T extends RealType<T> & NativeType<T>> void runUnetOnOpenImage(String modelPath)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        Object selected = ((YoloImageSelectionEntry) inferencePanel.getImageSourcePanel()
                .getOpenImagesComboBox().getSelectedItem()).getImage();
        if (selected == null) {
            JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
            return;
        }
        RandomAccessibleInterval<T> rai = consumer.convertIntoRai(selected);
        Consumer<String> logConsumer = str -> SwingUtilities.invokeLater(() ->
                UNetPluginUI.this.inferencePanel.getLogPanel().appendHtml(str));
        startInferenceLogTimer();
        try {
            List<Tensor<T>> outputs = inferenceService.run(modelPath, rai, logConsumer, true,
                    selectedInferenceDevice());
            for (Tensor<T> output : outputs) {
                consumer.displayImage(output.getData(), output.getAxesOrderString(), output.getName());
            }
        } finally {
            SwingUtilities.invokeLater(() -> UNetPluginUI.this.inferencePanel.getLogPanel().stopRunTimer());
        }
    }

    private <T extends RealType<T> & NativeType<T>> void runUnetOnSystemImage(String modelPath)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        File source = selectedSystemPath == null ? selectedSystemImageFile : selectedSystemPath;
        List<File> images = systemInferenceImages(source);
        if (images.isEmpty()) {
            SwingUtilities.invokeLater(() -> {
                showPreviewError(source != null && source.isDirectory() ? EMPTY_FOLDER_MESSAGE : INVALID_IMAGE_MESSAGE);
                inferencePanel.updateImageActionState();
            });
            return;
        }
        Consumer<String> logConsumer = str -> SwingUtilities.invokeLater(() ->
                UNetPluginUI.this.inferencePanel.getLogPanel().appendHtml(str));
        startInferenceLogTimer();
        logConsumer.accept("Starting inference on " + images.size() + " image(s).");
        int savedMasks = 0;
        try {
            for (int i = 0; i < images.size(); i ++) {
                if (cancelled || Thread.currentThread().isInterrupted()) {
                    break;
                }
                File imageFile = images.get(i);
                final int imageIndex = i + 1;
                final int totalImages = images.size();
                final boolean[] emittedPatchProgress = new boolean[] {false};
                try {
                    RandomAccessibleInterval<UnsignedByteType> rai = readImageFileAsRai(imageFile);
                    List<Tensor<T>> outputs = inferenceService.runWithProgress(modelPath, rai, progress -> {
                        if (progress == null) {
                            return;
                        }
                        if (progress.getPhase() == InferenceProgress.Phase.MODEL_LOADING) {
                            logConsumer.accept("Loading model: " + progress.getDetail());
                        } else if (progress.getPhase() == InferenceProgress.Phase.MODEL_LOADED) {
                            logConsumer.accept("Model loaded.");
                        } else if (progress.getPhase() == InferenceProgress.Phase.PATCH_END
                                && progress.getTotalPatches() > 1) {
                            emittedPatchProgress[0] = true;
                            logConsumer.accept(imageProgressBar(imageIndex, totalImages,
                                    progress.getPatchIndex(), progress.getTotalPatches()));
                        } else if (progress.getPhase() == InferenceProgress.Phase.TASK_RETRY) {
                            logConsumer.accept(progress.getDetail());
                        }
                    }, selectedInferenceDevice());
                    if (outputs.isEmpty()) {
                        throw new IOException("UNet did not return a labels image for " + imageFile.getName());
                    }
                    File outputMask = maskOutputFileFor(imageFile);
                    writeLabelMask(outputs.get(0), outputMask);
                    savedMasks++;
                    logConsumer.accept("Saved labels: " + outputMask.getAbsolutePath());
                    if (!emittedPatchProgress[0]) {
                        logConsumer.accept(imageProgressBar(imageIndex, totalImages));
                    }
                } catch (IOException ex) {
                    logConsumer.accept("Skipping image " + imageIndex + "/" + totalImages + ": " + ex.getMessage());
                }
            }
        } finally {
            SwingUtilities.invokeLater(() -> UNetPluginUI.this.inferencePanel.getLogPanel().stopRunTimer());
        }
        if (cancelled || Thread.currentThread().isInterrupted()) {
            return;
        }
        logConsumer.accept("Saved UNet label masks for " + savedMasks + " image(s).");
    }

    private String selectedInferenceDevice() {
        if (!isAccelerationEnabled()) {
            return "cpu";
        }
        String label = getAccelerationCheckBox().getText();
        return label != null && label.toLowerCase().contains("mps") ? "mps" : "cuda";
    }

    private static List<File> systemInferenceImages(File source) {
        if (source == null) {
            return Collections.emptyList();
        }
        if (source.isDirectory()) {
            return YoloImageFiles.readableImagesInDirectory(source);
        }
        if (YoloImageFiles.canReadImage(source)) {
            return Collections.singletonList(source);
        }
        return Collections.emptyList();
    }

    private static RandomAccessibleInterval<UnsignedByteType> readImageFileAsRai(File imageFile) throws IOException {
        BufferedImage image = ImageIO.read(imageFile);
        if (image == null) {
            throw new IOException("Unsupported image file: " + imageFile);
        }
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] pixels = new byte[width * height * 3];
        int offset = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                pixels[offset++] = (byte) ((rgb >> 16) & 0xff);
                pixels[offset++] = (byte) ((rgb >> 8) & 0xff);
                pixels[offset++] = (byte) (rgb & 0xff);
            }
        }
        return ArrayImgs.unsignedBytes(pixels, width, height, 3);
    }

    private static File maskOutputFileFor(File imageFile) {
        String fileName = imageFile.getName();
        int dot = fileName.lastIndexOf('.');
        String baseName = dot > 0 ? fileName.substring(0, dot) : fileName;
        String extension = outputMaskExtension(fileName);
        File parent = imageFile.getParentFile();
        return new File(parent == null ? new File(".") : parent, baseName + UNET_MASK_SUFFIX + "." + extension);
    }

    private static String outputMaskExtension(String fileName) {
        String lower = fileName == null ? "" : fileName.toLowerCase();
        return lower.endsWith(".tif") || lower.endsWith(".tiff") ? "tif" : "png";
    }

    private static <T extends RealType<T> & NativeType<T>> void writeLabelMask(Tensor<T> tensor, File outputFile)
            throws IOException {
        RandomAccessibleInterval<T> labels = tensor.getData();
        String axes = tensor.getAxesOrderString();
        BufferedImage image = labelMaskImage(labels, axes);
        String format = "tif".equals(outputMaskExtension(outputFile.getName())) ? "TIFF" : "png";
        if (!ImageIO.write(image, format, outputFile)) {
            throw new IOException("No ImageIO writer available for " + format + " masks.");
        }
    }

    private static <T extends RealType<T> & NativeType<T>>
    BufferedImage labelMaskImage(RandomAccessibleInterval<T> labels, String axes) {
        long[] dims = labels.dimensionsAsLongArray();
        int xAxis = axisIndex(axes, 'x', dims.length, dims.length > 1 ? 1 : 0);
        int yAxis = axisIndex(axes, 'y', dims.length, 0);
        int width = Math.toIntExact(dims[xAxis]);
        int height = Math.toIntExact(dims[yAxis]);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
        WritableRaster raster = image.getRaster();
        RandomAccess<T> access = labels.randomAccess();
        long[] position = new long[dims.length];
        for (int y = 0; y < height; y++) {
            position[yAxis] = y;
            for (int x = 0; x < width; x++) {
                position[xAxis] = x;
                access.setPosition(position);
                raster.setSample(x, y, 0, toUnsignedShortLabel(access.get().getRealDouble()));
            }
        }
        return image;
    }

    private static int axisIndex(String axes, char axis, int dimensions, int fallback) {
        if (axes != null) {
            int index = axes.toLowerCase().indexOf(axis);
            if (index >= 0 && index < dimensions) {
                return index;
            }
        }
        return Math.max(0, Math.min(fallback, dimensions - 1));
    }

    private static int toUnsignedShortLabel(double value) {
        if (!Double.isFinite(value) || value <= 0) {
            return 0;
        }
        return (int) Math.min(65535, Math.round(value));
    }

    private static String imageProgressBar(int imageIndex, int totalImages) {
        return imageProgressBar(imageIndex, totalImages, 0, 0);
    }

    private static String imageProgressBar(int imageIndex, int totalImages, int patchIndex, int totalPatches) {
        int safeTotal = Math.max(1, totalImages);
        int safeIndex = Math.max(0, Math.min(imageIndex, safeTotal));
        int safePatch = Math.max(0, Math.min(patchIndex, Math.max(0, totalPatches)));
        int safeTotalPatches = Math.max(1, totalPatches);
        double completedImages = safeIndex;
        boolean hasPatchProgress = safePatch > 0 && totalPatches > 1;
        if (hasPatchProgress) {
            completedImages = Math.max(0, safeIndex - 1) + safePatch / (double) safeTotalPatches;
        }
        int hashes = (int) Math.floor((completedImages / (double) safeTotal) * 20.0);
        StringBuilder builder = new StringBuilder(32);
        for (int i = 0; i < 20; i++) {
            builder.append(i < hashes ? '#' : '.');
        }
        builder.append(" Image ").append(safeIndex);
        if (hasPatchProgress) {
            builder.append('.').append(safePatch);
        }
        builder.append('/').append(safeTotal);
        return builder.toString();
    }

    private void startInferenceLogTimer() {
        Runnable start = () -> UNetPluginUI.this.inferencePanel.getLogPanel().startRunTimer();
        if (SwingUtilities.isEventDispatchThread()) {
            start.run();
            return;
        }
        try {
            SwingUtilities.invokeAndWait(start);
        } catch (Exception e) {
            SwingUtilities.invokeLater(start);
        }
    }

    /**
     * Runs model training.
     */
    public void trainUnet() {
        if (!trainPanel.validateTrainingFields()) {
            return;
        }
        inferenceService.close();
        cancelled = false;
        final long trainingRunId = startTrainingUiState();
        workerThread = new Thread(() -> {
            try {
                if (consumer != null) {
                    consumer.notifyParams(null);
                }
                UnetTrainingConfig config = readTrainingConfig();
                trainPanel.getTrainingLogPanel().startDiskLog(new File(config.getOutputModelDir()));
                File uiLog = trainPanel.getTrainingLogPanel().getLogFile();
                if (uiLog != null) {
                    appendTrainingLog("UI log file: " + uiLog.getAbsolutePath());
                }
                appendTrainingHeader("UNet", config.getModelName(),
                        config.getOutputModelDir(), config.getDatasetPath(),
                        unetConfigSummary(config));
                Consumer<String> logConsumer = str -> {
                    if (trainingRunId != trainingUiRunId) {
                        return;
                    }
                    if (cancelled && isExpectedApposeStreamClosed(str)) {
                        return;
                    }
                    SwingUtilities.invokeLater(() -> {
                        appendBackendTrainingLog(str);
                    });
                };
                trainingService.train(config,
                        progress -> SwingUtilities.invokeLater(() -> {
                            if (trainingRunId == trainingUiRunId) {
                                handleTrainingProgress(progress);
                            }
                        }),
                        preview -> SwingUtilities.invokeLater(() -> {
                            if (trainingRunId == trainingUiRunId) {
                                handleTrainingPreview(preview);
                            }
                        }),
                        logConsumer);
                if (trainingRunId == trainingUiRunId) {
                    appendTrainingLog("Exported/final UNet model file: " + config.getOutputModelPath());
                    appendTrainingLog("Training finished successfully.");
                    refreshUnetModels();
                }
            } catch (Exception | Error ex) {
                if (trainingRunId == trainingUiRunId && !cancelled) {
                    appendTrainingLog(TrainingLogUtils.failureStatus(ex) + ": " + errorMessage(ex));
                    ex.printStackTrace();
                }
            } finally {
                SwingUtilities.invokeLater(() -> finishTrainingUiState(trainingRunId));
            }
        }, "unet-training");
        workerThread.start();
    }

    private long startTrainingUiState() {
        long runId = ++trainingUiRunId;
        trainingStartMillis = System.currentTimeMillis();
        lastProgressMillis = trainingStartMillis;
        lastProgressStep = 0;
        currentTrainingStep = 0;
        totalTrainingSteps = 0;
        totalTrainingEpochs = 0;
        currentSecondsPerStep = Double.NaN;
        lastLoggedEpochStart = 0;
        lastLoggedTrainEpoch = 0;
        lastLoggedValidationEpoch = 0;
        lastLoggedPreviewEpoch = 0;
        secondsPerStepSamples.clear();
        trainingRunning = true;
        trainPanel.setTrainingRunning(true);
        updateTabLocks();
        trainPanel.getLossGraphPanel().clearValues();
        trainPanel.getMetricGraphPanel().clearValues();
        trainPanel.getTrainingLogPanel().clearLog();
        trainPanel.getLossGraphPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
        trainPanel.getMetricGraphPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
        trainPanel.getValidationPreviewPanel().clearPreview();
        trainPanel.getValidationPreviewPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
        trainPanel.getTrainingLogPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
        if (trainingTimer != null) {
            trainingTimer.stop();
        }
        trainingTimer = new Timer(1000, e -> updateTrainingGraphStatus());
        trainingTimer.start();
        return runId;
    }

    private void finishTrainingUiState(long runId) {
        if (runId != trainingUiRunId) {
            return;
        }
        if (trainingTimer != null) {
            trainingTimer.stop();
            trainingTimer = null;
        }
        trainPanel.setTrainingRunning(false);
        trainingRunning = false;
        updateTabLocks();
        long elapsed = Math.max(0L, System.currentTimeMillis() - trainingStartMillis);
        trainPanel.getLossGraphPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getMetricGraphPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getValidationPreviewPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getTrainingLogPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getTrainingLogPanel().closeDiskLog();
    }

    private void finishCancelledTrainingUiState() {
        long runId = trainingUiRunId;
        appendTrainingLog("Training cancelled by user.");
        finishTrainingUiState(runId);
        trainingUiRunId++;
    }

    private static boolean isExpectedApposeStreamClosed(String str) {
        return str != null
                && str.contains(APPOSE_STREAM_CLOSED)
                && str.contains("org.apposed.appose.Service.")
                && (str.contains("stdoutLoop") || str.contains("stderrLoop"));
    }

    private void handleTrainingProgress(UnetTrainingProgress progress) {
        updateTrainingProgressState(progress);
        logUnetTrainingProgress(progress);
        Double trainLoss = progress.getTrainingTotalLoss();
        if (trainLoss != null) {
            trainPanel.getLossGraphPanel().addTrainValue(
                    UnetTrainingProgress.UNET_TOTAL_LOSS_LABEL,
                    progress.getStep(),
                    progress.getEpoch(),
                    trainLoss);
        }
        Double validationLoss = progress.getValidationTotalLoss();
        if (validationLoss != null) {
            trainPanel.getLossGraphPanel().addValidationValue(
                    UnetTrainingProgress.UNET_TOTAL_LOSS_LABEL,
                    progress.getStep(),
                    progress.getEpoch(),
                    validationLoss);
        }
        Double metric = progress.getPrimaryMetric();
        if (metric != null) {
            trainPanel.getMetricGraphPanel().addValidationValue(
                    progress.getPrimaryMetricName(),
                    progress.getStep(),
                    progress.getEpoch(),
                    metric);
        }
    }

    private void handleTrainingPreview(UnetValidationPreview preview) {
        if (preview == null) {
            return;
        }
        String path = preview.getPreviewJsonPath();
        if (path == null || path.trim().isEmpty()) {
            path = preview.getLatestPreviewJsonPath();
        }
        if (path != null && !path.trim().isEmpty()) {
            trainPanel.getValidationPreviewPanel().loadPreview(path);
            logValidationPreview(preview.getEpoch(), path);
        }
    }

    private void updateTrainingProgressState(UnetTrainingProgress progress) {
        updateTrainingProgressState(progress.getStep(), progress.getTotalSteps(), progress.getTotalEpochs());
    }

    private void updateTrainingProgressState(int step, int totalSteps, int totalEpochs) {
        long now = System.currentTimeMillis();
        if (totalSteps > 0) {
            totalTrainingSteps = totalSteps;
        }
        if (totalEpochs > 0) {
            totalTrainingEpochs = totalEpochs;
        }
        if (step > lastProgressStep) {
            addSecondsPerStepSample((now - lastProgressMillis) / 1000.0d / (step - lastProgressStep));
            lastProgressMillis = now;
            lastProgressStep = step;
        }
        currentTrainingStep = Math.max(currentTrainingStep, step);
        updateTrainingGraphStatus();
    }

    private void addSecondsPerStepSample(double secondsPerStep) {
        if (Double.isNaN(secondsPerStep) || Double.isInfinite(secondsPerStep) || secondsPerStep <= 0.0d) {
            return;
        }
        secondsPerStepSamples.addLast(secondsPerStep);
        while (secondsPerStepSamples.size() > 3) {
            secondsPerStepSamples.removeFirst();
        }
        double sum = 0.0d;
        for (Double sample : secondsPerStepSamples) {
            sum += sample.doubleValue();
        }
        currentSecondsPerStep = sum / secondsPerStepSamples.size();
    }

    private void updateTrainingGraphStatus() {
        long elapsed = Math.max(0L, System.currentTimeMillis() - trainingStartMillis);
        trainPanel.getLossGraphPanel().setTrainingStatus(true, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getMetricGraphPanel().setTrainingStatus(true, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getValidationPreviewPanel().setTrainingStatus(true, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getTrainingLogPanel().setTrainingStatus(true, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
    }

    private void appendTrainingHeader(String framework, String modelName, String outputDir,
            String datasetPath, String configSummary) {
        appendTrainingLog("Starting training for " + framework + " model \"" + modelName + "\".");
        appendTrainingLog("Output directory: " + outputDir);
        appendTrainingLog("Analyzing dataset: " + datasetPath);
        appendTrainingLog("Training configuration: " + configSummary);
    }

    private void appendBackendTrainingLog(String message) {
        if (message == null || message.trim().isEmpty() || isNoisyTrainingMessage(message)) {
            return;
        }
        appendTrainingLog(message);
    }

    private void appendTrainingLog(String message) {
        trainPanel.getTrainingLogPanel().appendLine(message);
    }

    private void logValidationPreview(int epoch, String previewPath) {
        if (epoch > 0 && epoch == lastLoggedPreviewEpoch) {
            return;
        }
        lastLoggedPreviewEpoch = Math.max(lastLoggedPreviewEpoch, epoch);
        appendTrainingLog("Saved validation examples for epoch " + epoch + " at: " + previewPath);
    }

    private void logUnetTrainingProgress(UnetTrainingProgress progress) {
        int epoch = progress.getEpoch();
        if (epoch > 0 && epoch != lastLoggedEpochStart) {
            lastLoggedEpochStart = epoch;
            appendTrainingLog("Starting epoch " + epoch + "/" + progress.getTotalEpochs() + ".");
        }
        boolean epochSummary = progress.getValidationTotalLoss() != null
                || progress.getPrimaryMetric() != null
                || !progress.getMetrics().isEmpty();
        if (!epochSummary || epoch <= 0) {
            return;
        }
        Double trainLoss = progress.getTrainingTotalLoss();
        if (trainLoss != null && epoch != lastLoggedTrainEpoch) {
            lastLoggedTrainEpoch = epoch;
            appendTrainingLog("Training loss of epoch " + epoch + ": " + formatNumber(trainLoss) + ".");
        }
        Double validationLoss = progress.getValidationTotalLoss();
        if (validationLoss != null && epoch != lastLoggedValidationEpoch) {
            lastLoggedValidationEpoch = epoch;
            String metric = progress.getPrimaryMetric() == null ? ""
                    : ", " + progress.getPrimaryMetricName() + "="
                            + formatNumber(progress.getPrimaryMetric());
            appendTrainingLog("Validation of epoch " + epoch + ": loss="
                    + formatNumber(validationLoss) + metric + ".");
        }
    }

    private static boolean isNoisyTrainingMessage(String message) {
        String clean = message.trim();
        return clean.startsWith("step ")
                || clean.startsWith("epoch ")
                || clean.matches("^(YOLO|StarDist|UNet) training step .*")
                || clean.matches("^(YOLO|StarDist|UNet) training epoch .*")
                || clean.matches("^(YOLO|StarDist|UNet) validation preview epoch .*")
                || clean.matches("^(YOLO|StarDist|UNet) training started.*");
    }

    private static String unetConfigSummary(UnetTrainingConfig config) {
        String start = config.isFineTune() ? "fine tune from " + config.getBaseModelPath()
                : "train from scratch using " + config.getScratchArchitecture();
        return "epochs=" + config.getEpochs()
                + ", device=" + config.getDevice()
                + ", starting_point=" + start;
    }

    private static String formatNumber(Double value) {
        if (value == null || value.isNaN() || value.isInfinite()) {
            return "--";
        }
        return String.format(Locale.US, "%.5f", value.doubleValue());
    }

    private static String errorMessage(Throwable error) {
        return TrainingLogUtils.errorMessage(error);
    }

    private UnetTrainingConfig readTrainingConfig() {
        String modelsDir = consumer == null ? null : consumer.getModelsDir();
        return UnetTrainingConfig.fromUi(
                trainPanel.getModelNameField().getText(),
                trainPanel.getDatasetField().getText(),
                Integer.parseInt(trainPanel.getEpochsField().getText().trim()),
                trainPanel.getFineTuneRadio().isSelected(),
                trainPanel.getSelectedBaseModelValue(),
                trainPanel.getSelectedScratchArchitectureValue(),
                modelsDir,
                selectedTrainingDevice());
    }

    private String selectedTrainingDevice() {
        return selectedInferenceDevice();
    }

    private void refreshUnetModels() {
        String modelsDir = consumer == null ? null : consumer.getModelsDir();
        LinkedHashMap<String, String> unetModelEntries = UnetModelRegistry.buildModelEntries(modelsDir);
        SwingUtilities.invokeLater(() -> {
            inferencePanel.getModelSelectionPanel().setModels(unetModelEntries);
            trainPanel.setBaseModels(unetModelEntries);
        });
    }

    /**
     * Runs this class from the command line.
     *
     * @param args command-line arguments.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("UNet Plugin");
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            frame.getContentPane().add(new UNetPluginUI(null, null));
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            frame.setResizable(true);
            frame.setSize(400, 300);
        });
    }
}
