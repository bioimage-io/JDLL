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
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageInputStream;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.cellpose.CellposeInferencePanel;
import io.bioimage.modelrunner.gui.custom.cellpose.CellposeInferenceService;
import io.bioimage.modelrunner.gui.custom.cellpose.CellposeInstaller;
import io.bioimage.modelrunner.gui.custom.gui.CellposeGUI;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSelectionEntry;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Standard Cellpose inference GUI and host integration.
 */
public class CellposePluginUI extends CellposeGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    private static final String DEFAULT_PREVIEW_MESSAGE = "Preview will appear here";
    private static final String SYSTEM_PREVIEW_PROMPT =
            "Please select an image/folder from the file system";
    private static final String INVALID_IMAGE_MESSAGE = "Please provide a valid image file";
    private static final String EMPTY_FOLDER_MESSAGE = "Folder does not contain valid images";
    private static final Color PREVIEW_ERROR_COLOR = new Color(210, 40, 40);
    private static final String CELLPOSE_MASK_SUFFIX = "_cellpose_labels";

    public static final String[] RGB_LIST = {"red", "green", "blue"};
    public static final String[] GRAYSCALE_LIST = {"gray"};
    public static final String[] ALL_LIST = {"gray", "red", "green", "blue"};
    public static final Map<String, Integer> CHANNEL_MAP = channelMap();

    private final ConsumerInterface consumer;
    private final CellposeInferenceService inferenceService;
    private volatile boolean cancelled;
    private volatile boolean inferenceRunning;
    private Thread workerThread;
    private File selectedSystemPath;
    private File selectedSystemImageFile;
    private boolean windowCloseHookInstalled;
    private boolean accelerationAvailableBeforeRun;

    /**
     * Creates a Cellpose GUI without host-specific title customization.
     *
     * @param consumer the host image consumer.
     */
    public CellposePluginUI(ConsumerInterface consumer) {
        this(consumer, null);
    }

    /**
     * Creates a Cellpose GUI.
     *
     * @param consumer the host image consumer.
     * @param adapter the host GUI adapter.
     */
    public CellposePluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
        super(adapter);
        this.consumer = consumer;
        String modelsDirectory = consumer == null ? null : consumer.getModelsDir();
        this.inferenceService = new CellposeInferenceService(new CellposeInstaller(modelsDirectory));

        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        models.put("cyto3", "cyto3");
        models.put("cyto2", "cyto2");
        models.put("cyto", "cyto");
        models.put("nuclei", "nuclei");
        getInferencePanel().getModelSelectionPanel().setModels(models);

        getInferencePanel().getModelSelectionPanel().getBrowseButton().addActionListener(this);
        getInferencePanel().getActionPanel().getRunButton().addActionListener(this);
        getInferencePanel().getActionPanel().getCancelButton().addActionListener(this);
        getInferencePanel().getActionPanel().getCancelButton().setEnabled(false);
        installInferenceSourceListeners();

        if (consumer != null) {
            consumer.setVariableNames(null);
            List<JComponent> components = new ArrayList<JComponent>();
            components.add(getInferencePanel().getModelSelectionPanel().getModelComboBox());
            components.add(getInferencePanel().getImageSourcePanel().getOpenImagesComboBox());
            components.add(getInferencePanel().getImageSourcePanel().getFocusButton());
            components.add(getInferencePanel().getImageDisplayPanel());
            components.add(getInferencePanel().getOptionsPanel().getCytoplasmComboBox());
            components.add(getInferencePanel().getOptionsPanel().getNucleiComboBox());
            consumer.setComponents(components);
            consumer.updateGUI();
        }
    }

    @Override
    public void addNotify() {
        super.addNotify();
        installWindowCloseHook();
    }

    /**
     * Closes the active Cellpose Python service.
     */
    public void close() {
        cancelled = true;
        inferenceService.close();
        if (workerThread != null && workerThread.isAlive()) {
            workerThread.interrupt();
        }
        inferenceRunning = false;
    }

    /**
     * Retained for source compatibility with older host plugins.
     *
     * @param cancelCallback ignored; inference cancellation no longer closes the GUI.
     */
    public void setCancelCallback(Runnable cancelCallback) {
        // Standard inference cancellation keeps the model and window available.
    }

    @Override
    public void actionPerformed(ActionEvent event) {
        Object source = event.getSource();
        if (source == getInferencePanel().getModelSelectionPanel().getBrowseButton()) {
            browseModel();
        } else if (source == getInferencePanel().getActionPanel().getRunButton()) {
            startInference();
        } else if (source == getInferencePanel().getActionPanel().getCancelButton()) {
            cancelInference();
        }
    }

    private void startInference() {
        if (inferenceRunning) {
            return;
        }
        cancelled = false;
        setInferenceRunning(true);
        getInferencePanel().getLogPanel().startRunTimer();
        appendLog("Starting Cellpose inference.");
        workerThread = new Thread(() -> {
            try {
                runCellpose();
                if (!cancelled) {
                    appendLog("Cellpose inference finished.");
                }
            } catch (Exception error) {
                if (!cancelled) {
                    appendLog("Cellpose inference failed: " + rootMessage(error));
                    SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(this,
                            rootMessage(error), "Cellpose inference failed", JOptionPane.ERROR_MESSAGE));
                }
            } finally {
                SwingUtilities.invokeLater(() -> {
                    getInferencePanel().getLogPanel().stopRunTimer();
                    setInferenceRunning(false);
                });
            }
        }, "Cellpose inference");
        workerThread.start();
    }

    private void cancelInference() {
        if (!inferenceRunning) {
            return;
        }
        cancelled = true;
        appendLog("Cellpose inference cancelled.");
        inferenceService.cancelCurrentInference();
        if (workerThread != null) {
            workerThread.interrupt();
        }
    }

    private void runCellpose() throws Exception {
        saveParams();
        String model = getInferencePanel().getModelSelectionPanel().getSelectedModelValue();
        if (model == null || model.trim().isEmpty()) {
            throw new IllegalArgumentException("Please select a Cellpose model.");
        }
        if (getInferencePanel().getImageSourcePanel().getSystemImagesRadio().isSelected()) {
            runOnSystemImages(model);
        } else {
            runOnOpenImage(model);
        }
    }

    private <T extends RealType<T> & NativeType<T>> void runOnOpenImage(String model)
            throws Exception {
        Object selected = getInferencePanel().getImageSourcePanel()
                .getOpenImagesComboBox().getSelectedItem();
        if (!(selected instanceof YoloImageSelectionEntry)) {
            throw new IllegalArgumentException("Please select an open image.");
        }
        YoloImageSelectionEntry entry = (YoloImageSelectionEntry) selected;
        RandomAccessibleInterval<T> image = consumer.convertIntoRai(entry.getImage());
        List<Tensor<T>> outputs = runModel(model, image);
        displayOutputs(outputs, entry.getTitle());
    }

    private void runOnSystemImages(String model) throws Exception {
        File source = selectedSystemPath == null ? selectedSystemImageFile : selectedSystemPath;
        List<File> images = systemImages(source);
        if (images.isEmpty()) {
            throw new IllegalArgumentException(source != null && source.isDirectory()
                    ? EMPTY_FOLDER_MESSAGE : INVALID_IMAGE_MESSAGE);
        }
        appendLog("Starting inference on " + images.size() + " image(s).");
        int saved = 0;
        for (int index = 0; index < images.size(); index++) {
            if (cancelled || Thread.currentThread().isInterrupted()) {
                return;
            }
            File imageFile = images.get(index);
            appendLog("Processing image " + (index + 1) + "/" + images.size()
                    + ": " + imageFile.getName());
            List<Tensor<FloatType>> outputs = runModel(model, readImageFileAsRai(imageFile));
            File outputFile = maskOutputFileFor(imageFile);
            writeLabelMask(outputs.get(0), outputFile);
            appendLog("Saved labels: " + outputFile.getAbsolutePath());
            saved++;
        }
        appendLog("Saved Cellpose label masks for " + saved + " image(s).");
    }

    private <T extends RealType<T> & NativeType<T>,
            R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> runModel(String model, RandomAccessibleInterval<T> image) throws Exception {
        Consumer<String> log = this::appendLog;
        return inferenceService.run(model, image, selectedChannels(), selectedDiameter(),
                selectedInferenceDevice(), log);
    }

    private <T extends RealType<T> & NativeType<T>> void displayOutputs(
            List<Tensor<T>> outputs, String inputTitle) {
        int count = getInferencePanel().getOptionsPanel()
                .getDisplayIntermediateOutputsCheckBox().isSelected() ? outputs.size() : 1;
        for (int index = 0; index < count; index++) {
            Tensor<T> output = outputs.get(index);
            consumer.displayImage(output.getData(), output.getAxesOrderString(),
                    outputName(inputTitle, output.getName()));
        }
    }

    private int[] selectedChannels() {
        String cytoplasm = (String) getInferencePanel().getOptionsPanel()
                .getCytoplasmComboBox().getSelectedItem();
        String nuclei = (String) getInferencePanel().getOptionsPanel()
                .getNucleiComboBox().getSelectedItem();
        return new int[] {CHANNEL_MAP.get(cytoplasm), CHANNEL_MAP.get(nuclei)};
    }

    private Float selectedDiameter() {
        List<Rectangle2D.Double> boxes = getInferencePanel().getImageDisplayPanel().getBoxes();
        if (boxes.isEmpty()) {
            return null;
        }
        Rectangle2D.Double box = boxes.get(0);
        return (float) Math.sqrt(box.width * box.height);
    }

    private String selectedInferenceDevice() {
        if (!isAccelerationEnabled()) {
            return "cpu";
        }
        String text = getAccelerationCheckBox().getText();
        return text != null && text.toLowerCase().contains("mps") ? "mps" : "cuda";
    }

    private void saveParams() {
        if (consumer == null) {
            return;
        }
        LinkedHashMap<String, String> parameters = new LinkedHashMap<String, String>();
        parameters.put("model", getInferencePanel().getModelSelectionPanel().getSelectedModelValue());
        Float diameter = selectedDiameter();
        parameters.put("diameter", diameter == null ? "auto" : diameter.toString());
        parameters.put("cyto_color", (String) getInferencePanel().getOptionsPanel()
                .getCytoplasmComboBox().getSelectedItem());
        parameters.put("nuclei_color", (String) getInferencePanel().getOptionsPanel()
                .getNucleiComboBox().getSelectedItem());
        parameters.put("display_all", Boolean.toString(getInferencePanel().getOptionsPanel()
                .getDisplayIntermediateOutputsCheckBox().isSelected()));
        consumer.notifyParams(parameters);
    }

    private void setInferenceRunning(boolean running) {
        inferenceRunning = running;
        CellposeInferencePanel panel = getInferencePanel();
        if (running) {
            accelerationAvailableBeforeRun = getAccelerationCheckBox().isEnabled();
        }
        panel.getModelSelectionPanel().getModelComboBox().setEnabled(!running);
        panel.getModelSelectionPanel().getBrowseButton().setEnabled(!running);
        panel.getOptionsPanel().getCytoplasmComboBox().setEnabled(!running);
        panel.getOptionsPanel().getNucleiComboBox().setEnabled(!running);
        panel.getOptionsPanel().getDisplayIntermediateOutputsCheckBox().setEnabled(!running);
        getAccelerationCheckBox().setEnabled(!running && accelerationAvailableBeforeRun);
        panel.getImageSourcePanel().setInteractionEnabled(!running);
        panel.getActionPanel().getCancelButton().setEnabled(running);
        panel.updateImageActionState();
        if (running) {
            panel.getActionPanel().getRunButton().setEnabled(false);
            panel.getDrawButton().setEnabled(false);
            panel.getRefreshButton().setEnabled(false);
        }
    }

    private void installInferenceSourceListeners() {
        YoloImageSourcePanel source = getInferencePanel().getImageSourcePanel();
        source.getSystemImagesRadio().addActionListener(e -> showSystemPathPrompt());
        source.getOpenImagesRadio().addActionListener(e -> showOpenImageSource());
        source.getSystemPathField().addActionListener(e -> updateSystemPathPreviewFromField());
        source.getSystemPathField().addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(FocusEvent event) {
                updateSystemPathPreviewFromField();
            }
        });
        source.setSystemPathDropConsumer(this::updateSystemPathPreview);
        source.getBrowseButton().addActionListener(e -> browseSystemImagePath());
    }

    private void showSystemPathPrompt() {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        getInferencePanel().getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        getInferencePanel().getImageDisplayPanel().setEmptyMessage(SYSTEM_PREVIEW_PROMPT);
        getInferencePanel().getImageDisplayPanel().clearImage();
        getInferencePanel().updateImageActionState();
    }

    private void showOpenImageSource() {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        getInferencePanel().getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        getInferencePanel().getImageDisplayPanel().setEmptyMessage(DEFAULT_PREVIEW_MESSAGE);
        if (consumer != null) {
            consumer.updateGUI();
        }
        getInferencePanel().updateImageActionState();
    }

    private void browseSystemImagePath() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File selected = chooser.getSelectedFile();
            getInferencePanel().getImageSourcePanel().getSystemImagesRadio().setSelected(true);
            getInferencePanel().getImageSourcePanel().getSystemPathField()
                    .setText(selected.getAbsolutePath());
            updateSystemPathPreview(selected);
        }
    }

    private void updateSystemPathPreviewFromField() {
        if (!getInferencePanel().getImageSourcePanel().getSystemImagesRadio().isSelected()) {
            return;
        }
        String path = getInferencePanel().getImageSourcePanel().getSystemPathField().getText();
        updateSystemPathPreview(path == null || path.trim().isEmpty() ? null : new File(path.trim()));
    }

    private void updateSystemPathPreview(File path) {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        getInferencePanel().getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        File preview = path != null && path.isDirectory()
                ? YoloImageFiles.previewImageInDirectory(path) : path;
        if (preview == null || !YoloImageFiles.canReadImage(preview)) {
            showPreviewError(path != null && path.isDirectory()
                    ? EMPTY_FOLDER_MESSAGE : INVALID_IMAGE_MESSAGE);
            return;
        }
        selectedSystemPath = path;
        try {
            getInferencePanel().getImageDisplayPanel()
                    .setImageFile(preview, preview.getName(), true);
            selectedSystemImageFile = preview;
            getInferencePanel().getImageSourcePanel().setSystemPathSelectionConfirmed(true);
        } catch (IOException error) {
            selectedSystemPath = null;
            selectedSystemImageFile = null;
            showPreviewError(INVALID_IMAGE_MESSAGE);
        }
        getInferencePanel().updateImageActionState();
    }

    private void showPreviewError(String message) {
        getInferencePanel().getImageDisplayPanel().setEmptyMessage(message, PREVIEW_ERROR_COLOR);
        getInferencePanel().getImageDisplayPanel().clearImage();
        getInferencePanel().updateImageActionState();
    }

    private void browseModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File model = chooser.getSelectedFile();
            getInferencePanel().getModelSelectionPanel()
                    .addOrSelectModel(model.getName(), model.getAbsolutePath());
        }
    }

    private void appendLog(String message) {
        Runnable append = () -> getInferencePanel().getLogPanel().appendHtml(escapeHtml(message));
        if (SwingUtilities.isEventDispatchThread()) {
            append.run();
        } else {
            SwingUtilities.invokeLater(append);
        }
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
            @Override
            public void windowClosing(WindowEvent event) {
                close();
            }

            @Override
            public void windowClosed(WindowEvent event) {
                close();
            }
        });
        windowCloseHookInstalled = true;
    }

    private static Map<String, Integer> channelMap() {
        Map<String, Integer> channels = new HashMap<String, Integer>();
        channels.put("gray", 0);
        channels.put("red", 1);
        channels.put("green", 2);
        channels.put("blue", 3);
        return Collections.unmodifiableMap(channels);
    }

    private static String outputName(String inputTitle, String output) {
        String title = inputTitle == null || inputTitle.trim().isEmpty() ? "image" : inputTitle;
        int extension = title.lastIndexOf('.');
        String base = extension > 0 ? title.substring(0, extension) : title;
        return base + "_" + output + ".tif";
    }

    private static String rootMessage(Throwable error) {
        Throwable root = error;
        while (root.getCause() != null && root.getCause() != root) {
            root = root.getCause();
        }
        String message = root.getMessage();
        return message == null || message.trim().isEmpty()
                ? root.getClass().getSimpleName() : message;
    }

    private static String escapeHtml(String text) {
        return text == null ? "" : text.replace("&", "&amp;")
                .replace("<", "&lt;").replace(">", "&gt;");
    }

    private static List<File> systemImages(File source) {
        if (source == null) {
            return Collections.emptyList();
        }
        if (source.isDirectory()) {
            return YoloImageFiles.readableImagesInDirectory(source);
        }
        return YoloImageFiles.canReadImage(source)
                ? Collections.singletonList(source) : Collections.emptyList();
    }

    private static RandomAccessibleInterval<FloatType> readImageFileAsRai(File imageFile)
            throws IOException {
        try (ImageInputStream input = ImageIO.createImageInputStream(imageFile)) {
            Iterator<ImageReader> readers = input == null
                    ? Collections.<ImageReader>emptyList().iterator()
                    : ImageIO.getImageReaders(input);
            if (!readers.hasNext()) {
                throw new IOException("Unsupported image file: " + imageFile);
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(input, false, true);
                int depth = Math.max(1, reader.getNumImages(true));
                BufferedImage first = reader.read(0);
                int width = first.getWidth();
                int height = first.getHeight();
                int channels = first.getRaster().getNumBands() == 1 ? 1 : 3;
                float[] pixels = new float[Math.multiplyExact(
                        Math.multiplyExact(width, height), Math.multiplyExact(channels, depth))];
                for (int z = 0; z < depth; z++) {
                    BufferedImage plane = z == 0 ? first : reader.read(z);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int base = x + width * (y + height * (channels * z));
                            if (channels == 1) {
                                pixels[base] = plane.getRaster().getSampleFloat(x, y, 0);
                            } else {
                                int rgb = plane.getRGB(x, y);
                                pixels[base] = (rgb >> 16) & 0xff;
                                pixels[base + width * height] = (rgb >> 8) & 0xff;
                                pixels[base + 2 * width * height] = rgb & 0xff;
                            }
                        }
                    }
                }
                return ArrayImgs.floats(pixels, width, height, channels, depth);
            } finally {
                reader.dispose();
            }
        }
    }

    private static File maskOutputFileFor(File imageFile) {
        String name = imageFile.getName();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        String extension = name.toLowerCase().endsWith(".tif")
                || name.toLowerCase().endsWith(".tiff") ? "tif" : "png";
        return new File(imageFile.getParentFile(), base + CELLPOSE_MASK_SUFFIX + "." + extension);
    }

    private static <T extends RealType<T> & NativeType<T>> void writeLabelMask(
            Tensor<T> tensor, File outputFile) throws IOException {
        RandomAccessibleInterval<T> labels = tensor.getData();
        String axes = tensor.getAxesOrderString();
        int batchAxis = axes == null ? -1 : axes.toLowerCase().indexOf('b');
        int depth = batchAxis >= 0 ? Math.toIntExact(labels.dimension(batchAxis)) : 1;
        String format = outputFile.getName().toLowerCase().endsWith(".tif") ? "TIFF" : "png";
        if (depth == 1) {
            if (!ImageIO.write(labelImage(labels, axes, batchAxis, 0), format, outputFile)) {
                throw new IOException("No ImageIO writer available for " + format + ".");
            }
            return;
        }
        if (!"TIFF".equals(format)) {
            throw new IOException("Multi-frame Cellpose labels must be saved as TIFF.");
        }
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("TIFF");
        if (!writers.hasNext()) {
            throw new IOException("No TIFF writer is available.");
        }
        ImageWriter writer = writers.next();
        try (ImageOutputStream output = ImageIO.createImageOutputStream(outputFile)) {
            writer.setOutput(output);
            writer.prepareWriteSequence(null);
            for (int frame = 0; frame < depth; frame++) {
                writer.writeToSequence(new javax.imageio.IIOImage(
                        labelImage(labels, axes, batchAxis, frame), null, null), null);
            }
            writer.endWriteSequence();
        } finally {
            writer.dispose();
        }
    }

    private static <T extends RealType<T> & NativeType<T>> BufferedImage labelImage(
            RandomAccessibleInterval<T> labels, String axes, int batchAxis, int frame) {
        int xAxis = axes == null ? 0 : axes.toLowerCase().indexOf('x');
        int yAxis = axes == null ? 1 : axes.toLowerCase().indexOf('y');
        int width = Math.toIntExact(labels.dimension(xAxis));
        int height = Math.toIntExact(labels.dimension(yAxis));
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
        WritableRaster raster = image.getRaster();
        RandomAccess<T> access = labels.randomAccess();
        long[] position = new long[labels.numDimensions()];
        if (batchAxis >= 0) {
            position[batchAxis] = frame;
        }
        for (int y = 0; y < height; y++) {
            position[yAxis] = y;
            for (int x = 0; x < width; x++) {
                position[xAxis] = x;
                access.setPosition(position);
                raster.setSample(x, y, 0, Math.max(0,
                        Math.min(65535, (int) Math.round(access.get().getRealDouble()))));
            }
        }
        return image;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Cellpose Plugin");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setContentPane(new CellposePluginUI(null, null));
            frame.setSize(550, 700);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
