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

import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.imageio.ImageIO;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.stardist.StardistInferenceService;
import io.bioimage.modelrunner.gui.custom.stardist.StardistInstaller;
import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry;
import io.bioimage.modelrunner.gui.custom.stardist.StardistTrainingConfig;
import io.bioimage.modelrunner.gui.custom.stardist.StardistTrainingService;
import io.bioimage.modelrunner.gui.custom.yolo.StardistGUI;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.special.stardist.StardistTrainingProgress;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.awt.Color;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class StarDistPluginUI extends StardistGUI implements ActionListener {

    private static final long serialVersionUID = 5754022448277417301L;
	private static final String DEFAULT_PREVIEW_MESSAGE = "Preview will appear here";
    private static final String SYSTEM_PREVIEW_PROMPT = "Please select an image/folder from the file system";
    private static final String INVALID_IMAGE_MESSAGE = "Please provide a valid image file";
    private static final String EMPTY_FOLDER_MESSAGE = "Folder does not contain valid images";
    private static final Color PREVIEW_ERROR_COLOR = new Color(210, 40, 40);
    private static final String APPOSE_STREAM_CLOSED = "java.io.IOException: Stream closed";
    private static final String STARDIST_MASK_SUFFIX = "_stardist_labels";

    private final ConsumerInterface consumer;
    private final StardistInstaller installer = new StardistInstaller();
    private final StardistInferenceService inferenceService = new StardistInferenceService(installer);
    private final StardistTrainingService trainingService = new StardistTrainingService(installer);
    private volatile boolean cancelled = false;
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
    private boolean windowCloseHookInstalled;
    
    private Runnable cancelCallback;
    Thread workerThread;

    /**
     * Creates a new CellposePluginUI.
     *
     * @param consumer the consumer callback.
     * @param adapter the adapter.
     */
    public StarDistPluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
    	super(adapter);
        // Set a modern-looking border layout with padding
    	this.consumer = consumer;
    	List<JComponent> componentList = new ArrayList<JComponent>();

    	String modelsDir = consumer == null ? null : consumer.getModelsDir();
    	LinkedHashMap<String, String> stardistModelEntries = StardistModelRegistry.buildModelEntries(modelsDir);
    	this.inferencePanel.getModelSelectionPanel().setModels(stardistModelEntries);
    	this.trainPanel.setBaseModels(stardistModelEntries);
        if (this.consumer == null) {
            return;
        }
        this.consumer.setVariableNames(null);
        componentList.add(this.inferencePanel.getModelSelectionPanel().getModelComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getOpenImagesComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getFocusButton());
        componentList.add(this.inferencePanel.getImageDisplayPanel());
        this.consumer.setComponents(componentList);
        this.inferencePanel.getDrawButton().addActionListener(this);
        this.inferencePanel.getRefreshButton().addActionListener(this);
        this.inferencePanel.getActionPanel().getCancelButton().addActionListener(this);
        this.inferencePanel.getActionPanel().getRunButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getCancelButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getRunButton().addActionListener(this);
        installInferenceSourceListeners();
        installTabLifecycleListener();
        consumer.updateGUI();
    }

    /**
     * Sets cancel callback.
     *
     * @param cancelCallback the cancelCallback parameter.
     */
    public void setCancelCallback(Runnable cancelCallback) {
    	this.cancelCallback = cancelCallback;
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
    }

    /**
     * Performs add notify.
     */
    @Override
    public void addNotify() {
        super.addNotify();
        installWindowCloseHook();
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

    private void installTabLifecycleListener() {
        tabs.addChangeListener(e -> {
            if (tabs.getSelectedIndex() == 0) {
                if (trainingRunning) {
                    trainingService.requestCancel();
                }
                inferenceService.close();
            } else if (tabs.getSelectedIndex() == 1) {
                inferenceService.close();
            }
        });
    }

    private void installWindowCloseHook() {
        if (windowCloseHookInstalled) {
            return;
        }
        Window window = SwingUtilities.getWindowAncestor(this);
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

    // For demonstration purposes: a main method to show the UI in a JFrame.
    /**
     * Executes main.
     *
     * @param args the args parameter.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            /**
             * Executes run.
             */
            public void run() {
                JFrame frame = new JFrame("Yolo Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new StarDistPluginUI(null, null));
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setResizable(true);
                frame.setSize(400, 200);
            }
        });
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
    		workerThread = new Thread(() -> {
        		try {
                    inferenceRunning = true;
        			runStardist();
    			} catch (Exception e1) {
    				if (cancelled)
    					return;
    				e1.printStackTrace();
    			} finally {
                    inferenceRunning = false;
    			}
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.inferencePanel.getActionPanel().getCancelButton()) {
    		cancel();
    	} else if (e.getSource() == this.trainPanel.getTrainActionPanel().getRunButton()) {
    		trainStardist();
    	} else if (e.getSource() == this.trainPanel.getTrainActionPanel().getCancelButton()) {
    		cancel();
    	}
    }
    
    private void cancel() {
    	cancelled = true;
        if (trainingRunning) {
            trainingService.close();
            finishCancelledTrainingUiState();
            if (cancelCallback != null) {
                cancelCallback.run();
            }
            return;
        }
        if (inferenceRunning) {
            inferenceService.cancelCurrentInference();
        } else if (workerThread != null && workerThread.isAlive()) {
    		workerThread.interrupt();
        }
    	if (cancelCallback != null)
    		cancelCallback.run();
    }
    
    private void saveParams() {
    	this.consumer.notifyParams(null);
    }
    
    private void runStardist()
    		throws RunModelException, LoadModelException, BuildException, IOException,
    		ExecutionException, InterruptedException {
    	saveParams();
    	String modelPath = this.inferencePanel.getModelSelectionPanel().getSelectedModelValue();
    	if (this.inferencePanel.getImageSourcePanel().getSystemImagesRadio().isSelected()) {
    		runStardistOnSystemImage(modelPath);
    		return;
    	}
    	runStardistOnOpenImage(modelPath);
    }

    private < T extends RealType< T > & NativeType< T > > void runStardistOnOpenImage(String modelPath)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
        List<Rectangle2D.Double> boxes = inferencePanel.getImageDisplayPanel().getBoxes();
        inferenceService.setObjectSize(boxes);
    	Consumer<String> logConsumer = str -> SwingUtilities.invokeLater(() ->
    			StarDistPluginUI.this.inferencePanel.getLogPanel().appendHtml(str));
        startInferenceLogTimer();
        try {
            List<Tensor<T>> detections = inferenceService.run(modelPath, rai, logConsumer, true,
                    selectedInferenceDevice());
            consumer.displayImage(detections.get(0).getData(), detections.get(0).getAxesOrderString(), detections.get(0).getName());
        } finally {
            SwingUtilities.invokeLater(() -> StarDistPluginUI.this.inferencePanel.getLogPanel().stopRunTimer());
        }
    }

    private < T extends RealType< T > & NativeType< T > > void runStardistOnSystemImage(String modelPath)
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
        List<Rectangle2D.Double> boxes = inferencePanel.getImageDisplayPanel().getBoxes();
        inferenceService.setObjectSize(boxes);
        Consumer<String> logConsumer = str -> SwingUtilities.invokeLater(() ->
                StarDistPluginUI.this.inferencePanel.getLogPanel().appendHtml(str));
        startInferenceLogTimer();
        logConsumer.accept("Starting inference on " + images.size() + " image(s).");
        int savedMasks = 0;
        try {
            for (int i = 0; i < images.size(); i++) {
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
                        throw new IOException("StarDist did not return a labels image for " + imageFile.getName());
                    }
                    File outputMask = maskOutputFileFor(imageFile);
                    writeLabelMask(outputs.get(0), outputMask);
                    savedMasks++;
                    logConsumer.accept("Saved labels: " + outputMask.getAbsolutePath());
                    if (!emittedPatchProgress[0]) {
                        logConsumer.accept(imageProgressBar(imageIndex, totalImages));
                    }
                } catch (IOException e) {
                    logConsumer.accept("Skipping image " + imageIndex + "/" + totalImages + ": " + e.getMessage());
                }
            }
        } finally {
            SwingUtilities.invokeLater(() -> StarDistPluginUI.this.inferencePanel.getLogPanel().stopRunTimer());
        }
        if (cancelled || Thread.currentThread().isInterrupted()) {
            return;
        }
        logConsumer.accept("Saved StarDist label masks for " + savedMasks + " image(s).");
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
        return new File(parent == null ? new File(".") : parent, baseName + STARDIST_MASK_SUFFIX + "." + extension);
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
        String format = imageIoFormat(outputFile.getName());
        if (!ImageIO.write(image, format, outputFile)) {
            throw new IOException("No ImageIO writer available for " + format + " masks.");
        }
    }

    private static String imageIoFormat(String fileName) {
        return "tif".equals(outputMaskExtension(fileName)) ? "TIFF" : "png";
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
        Runnable start = () -> StarDistPluginUI.this.inferencePanel.getLogPanel().startRunTimer();
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
    public void trainStardist() {
        if (!trainPanel.validateTrainingFields()) {
            return;
        }
        cancelled = false;
        inferenceService.close();
        final long trainingRunId = startTrainingUiState();
        workerThread = new Thread(() -> {
            try {
                StardistTrainingConfig config = readTrainingConfig();
                Consumer<String> logConsumer = str -> {
                    if (trainingRunId != trainingUiRunId) {
                        return;
                    }
                    if (cancelled && isExpectedApposeStreamClosed(str)) {
                        return;
                    }
                    SwingUtilities.invokeLater(() -> System.err.println(str));
                };
                trainingService.train(config,
                        progress -> SwingUtilities.invokeLater(() -> {
                            if (trainingRunId != trainingUiRunId) {
                                return;
                            }
                            handleTrainingProgress(progress);
                        }),
                        preview -> SwingUtilities.invokeLater(() -> {
                            if (trainingRunId != trainingUiRunId) {
                                return;
                            }
                            if (preview.getPreviewJsonPath() != null) {
                                trainPanel.getValidationPreviewPanel().loadPreview(preview.getPreviewJsonPath());
                                logConsumer.accept("Validation preview epoch " + preview.getEpoch()
                                        + ": " + preview.getPreviewJsonPath());
                            }
                        }),
                        logConsumer);
                if (trainingRunId == trainingUiRunId) {
                    refreshStardistModels();
                }
            } catch (Exception | Error e) {
                if (trainingRunId == trainingUiRunId && !cancelled) {
                    e.printStackTrace();
                }
            } finally {
                SwingUtilities.invokeLater(() -> finishTrainingUiState(trainingRunId));
            }
        });
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
        secondsPerStepSamples.clear();
        trainingRunning = true;
        trainPanel.setTrainingRunning(true);
        trainPanel.getLossGraphPanel().clearValues();
        trainPanel.getMetricGraphPanel().clearValues();
        trainPanel.getLossGraphPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
        trainPanel.getMetricGraphPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
        trainPanel.getValidationPreviewPanel().clearPreview();
        trainPanel.getValidationPreviewPanel().setTrainingStatus(true, 0, 0, 0, 0L, Double.NaN);
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
        tabs.setEnabledAt(0, true);
        long elapsed = Math.max(0L, System.currentTimeMillis() - trainingStartMillis);
        trainPanel.getLossGraphPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getMetricGraphPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
        trainPanel.getValidationPreviewPanel().setTrainingStatus(false, currentTrainingStep, totalTrainingSteps,
                totalTrainingEpochs, elapsed, currentSecondsPerStep);
    }

    private void finishCancelledTrainingUiState() {
        long runId = trainingUiRunId;
        finishTrainingUiState(runId);
        trainingUiRunId++;
    }

    private static boolean isExpectedApposeStreamClosed(String str) {
        return str != null
                && str.contains(APPOSE_STREAM_CLOSED)
                && str.contains("org.apposed.appose.Service.")
                && (str.contains("stdoutLoop") || str.contains("stderrLoop"));
    }

    private void handleTrainingProgress(StardistTrainingProgress progress) {
        updateTrainingProgressState(progress);
        Double trainLoss = progress.getTrainingTotalLoss();
        if (trainLoss != null) {
            trainPanel.getLossGraphPanel().addTrainValue(
            		StardistTrainingProgress.STARDIST_TOTAL_LOSS_LABEL,
                    progress.getStep(),
                    progress.getEpoch(),
                    trainLoss);
        }
        Double validationLoss = progress.getValidationTotalLoss();
        if (validationLoss != null) {
            trainPanel.getLossGraphPanel().addValidationValue(
            		StardistTrainingProgress.STARDIST_TOTAL_LOSS_LABEL,
                    progress.getStep(),
                    progress.getEpoch(),
                    validationLoss);
        }
        Double metric = progress.getLearningRate();
        if (metric != null) {
            trainPanel.getMetricGraphPanel().addValidationValue(
                    StardistTrainingProgress.LEARNING_RATE,
                    progress.getStep(),
                    progress.getEpoch(),
                    metric);
        }
    }

    private void updateTrainingProgressState(StardistTrainingProgress progress) {
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
    }

    private StardistTrainingConfig readTrainingConfig() {
    	String modelsDir = consumer == null ? null : consumer.getModelsDir();
    	return StardistTrainingConfig.fromUi(
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

    private void refreshStardistModels() {
    	String modelsDir = consumer == null ? null : consumer.getModelsDir();
    	LinkedHashMap<String, String> stardistModelEntries = StardistModelRegistry.buildModelEntries(modelsDir);
    	SwingUtilities.invokeLater(() -> {
    		inferencePanel.getModelSelectionPanel().setModels(stardistModelEntries);
    		trainPanel.setBaseModels(stardistModelEntries);
    	});
    }

}
