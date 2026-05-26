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
package io.bioimage.modelrunner.gui.custom.stardist;

import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.border.LineBorder;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import io.bioimage.modelrunner.gui.custom.yolo.TrainingValidationPreview;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageDisplayPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;
import io.bioimage.modelrunner.numpy.DecodeNumpy;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

public class StardistValidationPreviewPanel extends JPanel implements TrainingValidationPreview {

    private static final long serialVersionUID = -849272014619012798L;

    private static final int OUTER_PAD = 6;
    private static final int VIEWER_TO_ARROWS_GAP = 2;
    private static final int ARROWS_TO_STATUS_GAP = 4;
    private static final int ARROW_BUTTON_MIN_HEIGHT = 13;
    private static final int ARROW_BUTTON_MAX_HEIGHT = 21;
    private static final int STATUS_H = 36;
    private static final Color TEXT_COLOR = new Color(70, 78, 98);
    private static final Color GT_COLOR = new Color(80, 220, 120);
    private static final Color PREDICTION_COLOR = new Color(230, 44, 140);
    private static final String PREVIOUS_SYMBOL = "\u25C0";
    private static final String NEXT_SYMBOL = "\u25B6";
    private static final String WAITING_MESSAGE = "Validation examples available after the first epoch finishes";
    private static final String ERROR_MESSAGE = "Could not load StarDist validation preview";

    private final YoloImageDisplayPanel imagePanel = new PreviewImagePanel();
    private final TrainingStatusPanel statusPanel = new TrainingStatusPanel();
    private final JButton previousButton = new JButton(PREVIOUS_SYMBOL);
    private final JButton nextButton = new JButton(NEXT_SYMBOL);
    private final List<PreviewSample> samples = new ArrayList<PreviewSample>();

    private int currentIndex;
    private int previewEpoch;
    private boolean trainingActive;
    private int currentStep;
    private int totalSteps;
    private int totalEpochs;
    private long elapsedMillis;
    private double secondsPerIteration = Double.NaN;

    public StardistValidationPreviewPanel() {
        setLayout(null);
        setOpaque(true);
        setBackground(Color.WHITE);
        setBorder(new LineBorder(new Color(205, 210, 221)));

        imagePanel.setEmptyMessage(WAITING_MESSAGE);
        statusPanel.setOpaque(false);
        YoloUiUtils.styleFlatSecondaryButton(previousButton);
        YoloUiUtils.styleFlatSecondaryButton(nextButton);
        previousButton.addActionListener(e -> showSample(currentIndex - 1));
        nextButton.addActionListener(e -> showSample(currentIndex + 1));

        add(imagePanel);
        add(previousButton);
        add(nextButton);
        add(statusPanel);
        clearPreview();
    }

    @Override
    public void clearPreview() {
        samples.clear();
        currentIndex = 0;
        previewEpoch = 0;
        imagePanel.setEmptyMessage(WAITING_MESSAGE);
        imagePanel.clearImage();
        updateStatusPanel();
        updateButtons();
    }

    @Override
    public void loadPreview(String jsonPath) {
        if (jsonPath == null || jsonPath.trim().isEmpty()) {
            return;
        }
        try {
            JsonObject root = readJson(jsonPath.trim());
            List<PreviewSample> loaded = parseSamples(root);
            String selectedImage = getSelectedImagePath();
            int selectedIndex = currentIndex;
            samples.clear();
            samples.addAll(loaded);
            previewEpoch = getInt(root, "epoch", previewEpoch);
            if (samples.isEmpty()) {
                currentIndex = 0;
                imagePanel.setEmptyMessage(WAITING_MESSAGE);
                imagePanel.clearImage();
            } else {
                currentIndex = selectUpdatedIndex(selectedImage, selectedIndex, samples);
                showSample(currentIndex);
            }
            updateStatusPanel();
            updateButtons();
        } catch (Exception e) {
            samples.clear();
            imagePanel.setEmptyMessage(ERROR_MESSAGE, new Color(180, 30, 30));
            imagePanel.clearImage();
            updateStatusPanel();
            updateButtons();
        }
    }

    @Override
    public void setTrainingStatus(boolean active, int currentStep, int totalSteps,
            int totalEpochs, long elapsedMillis, double secondsPerIteration) {
        this.trainingActive = active;
        this.currentStep = Math.max(0, currentStep);
        this.totalSteps = Math.max(0, totalSteps);
        this.totalEpochs = Math.max(0, totalEpochs);
        this.elapsedMillis = Math.max(0L, elapsedMillis);
        this.secondsPerIteration = secondsPerIteration;
        updateStatusPanel();
    }

    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int innerW = Math.max(1, w - 2 * OUTER_PAD);
        int arrowH = Math.max(ARROW_BUTTON_MIN_HEIGHT,
                Math.min(ARROW_BUTTON_MAX_HEIGHT, Math.min(w, h) / 15));
        int statusH = Math.max(YoloUiUtils.MIN_FONT_SIZE * 2,
                Math.min(STATUS_H, h / 5));
        int bottomH = arrowH + VIEWER_TO_ARROWS_GAP + ARROWS_TO_STATUS_GAP + statusH;
        int imageH = Math.max(1, h - 2 * OUTER_PAD - bottomH);
        int y = OUTER_PAD;

        imagePanel.setBounds(OUTER_PAD, y, innerW, imageH);
        y += imageH + VIEWER_TO_ARROWS_GAP;

        int leftArrowW = innerW / 2;
        previousButton.setBounds(OUTER_PAD, y, leftArrowW, arrowH);
        nextButton.setBounds(OUTER_PAD + leftArrowW, y, innerW - leftArrowW, arrowH);
        y += arrowH + ARROWS_TO_STATUS_GAP;

        statusPanel.setBounds(OUTER_PAD, y, innerW, statusH);

        YoloUiUtils.applyResponsiveText(previousButton, leftArrowW - 4, arrowH);
        YoloUiUtils.applyResponsiveText(nextButton, innerW - leftArrowW - 4, arrowH);
    }

    private void showSample(int requestedIndex) {
        if (samples.isEmpty()) {
            imagePanel.setEmptyMessage(WAITING_MESSAGE);
            imagePanel.clearImage();
            updateButtons();
            return;
        }
        currentIndex = wrap(requestedIndex, samples.size());
        PreviewSample sample = samples.get(currentIndex);
        try {
            BufferedImage image = buildOverlay(sample);
            imagePanel.setBufferedImage(image, buildImageTitle(sample), false);
            imagePanel.setTitleColor(Color.WHITE);
        } catch (Exception e) {
            imagePanel.setEmptyMessage(ERROR_MESSAGE, new Color(180, 30, 30));
            imagePanel.clearImage();
        }
        updateStatusPanel();
        updateButtons();
    }

    private BufferedImage buildOverlay(PreviewSample sample) throws IOException {
        BufferedImage image = toBufferedImage(DecodeNumpy.loadNpy(sample.imagePath));
        if (sample.labelPath != null) {
            drawMaskContours(image, DecodeNumpy.loadNpy(sample.labelPath), GT_COLOR);
        }
        if (sample.predictionPath != null) {
            drawMaskContours(image, DecodeNumpy.loadNpy(sample.predictionPath), PREDICTION_COLOR);
        }
        return image;
    }

    private void updateButtons() {
        boolean enabled = samples.size() > 1;
        previousButton.setEnabled(enabled);
        nextButton.setEnabled(enabled);
    }

    private String buildImageTitle(PreviewSample sample) {
        return sample.title + " --- " + (currentIndex + 1) + "/" + samples.size()
                + " | GT green | prediction magenta";
    }

    private String getSelectedImagePath() {
        if (samples.isEmpty() || currentIndex < 0 || currentIndex >= samples.size()) {
            return null;
        }
        return samples.get(currentIndex).imagePath;
    }

    private void updateStatusPanel() {
        statusPanel.repaint();
    }

    private static JsonObject readJson(String jsonPath) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(jsonPath)), StandardCharsets.UTF_8);
        return JsonParser.parseString(content).getAsJsonObject();
    }

    private static List<PreviewSample> parseSamples(JsonObject root) {
        JsonArray array = root == null ? null : root.getAsJsonArray("samples");
        if (array == null || array.size() == 0) {
            return Collections.emptyList();
        }
        List<PreviewSample> result = new ArrayList<PreviewSample>();
        for (JsonElement element : array) {
            if (!element.isJsonObject()) {
                continue;
            }
            JsonObject sample = element.getAsJsonObject();
            String imagePath = getExistingPath(sample, "image_path");
            if (imagePath == null) {
                continue;
            }
            result.add(new PreviewSample(
                    imagePath,
                    getExistingPath(sample, "label_path"),
                    getExistingPath(sample, "prediction_path"),
                    getExistingPath(sample, "prob_path"),
                    "epoch " + getInt(root, "epoch", 0)));
        }
        return result;
    }

    private static String getExistingPath(JsonObject object, String key) {
        String path = getString(object, key);
        return path == null || !new File(path).isFile() ? null : path;
    }

    private static int selectUpdatedIndex(String selectedPath, int selectedIndex, List<PreviewSample> samples) {
        if (samples == null || samples.isEmpty()) {
            return 0;
        }
        if (selectedPath != null) {
            for (int i = 0; i < samples.size(); i++) {
                if (selectedPath.equals(samples.get(i).imagePath)) {
                    return i;
                }
            }
        }
        return Math.max(0, Math.min(selectedIndex, samples.size() - 1));
    }

    private static BufferedImage toBufferedImage(RandomAccessibleInterval<?> rai) {
        if (rai == null || rai.numDimensions() < 2) {
            throw new IllegalArgumentException("StarDist preview image must have at least two dimensions.");
        }
        ImageLayout layout = ImageLayout.from(rai);
        double[] range = valueRange(rai, layout);
        BufferedImage image = new BufferedImage(layout.width, layout.height, BufferedImage.TYPE_INT_RGB);
        RandomAccess<?> access = rai.randomAccess();
        long[] position = new long[rai.numDimensions()];
        for (int y = 0; y < layout.height; y++) {
            for (int x = 0; x < layout.width; x++) {
                int r = channelToByte(access, position, layout, x, y, 0, range);
                int g = layout.channels > 1 ? channelToByte(access, position, layout, x, y, 1, range) : r;
                int b = layout.channels > 2 ? channelToByte(access, position, layout, x, y, 2, range) : r;
                image.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return image;
    }

    private static double[] valueRange(RandomAccessibleInterval<?> rai, ImageLayout layout) {
        RandomAccess<?> access = rai.randomAccess();
        long[] position = new long[rai.numDimensions()];
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (int y = 0; y < layout.height; y++) {
            for (int x = 0; x < layout.width; x++) {
                for (int c = 0; c < layout.channels; c++) {
                    double value = imageValue(access, position, layout, x, y, c);
                    if (Double.isFinite(value)) {
                        min = Math.min(min, value);
                        max = Math.max(max, value);
                    }
                }
            }
        }
        if (!Double.isFinite(min) || !Double.isFinite(max) || max <= min) {
            return new double[] {0.0d, 1.0d};
        }
        return new double[] {min, max};
    }

    private static int channelToByte(RandomAccess<?> access, long[] position, ImageLayout layout,
            int x, int y, int channel, double[] range) {
        double value = imageValue(access, position, layout, x, y, channel);
        double normalized = (value - range[0]) / (range[1] - range[0]);
        int byteValue = (int) Math.round(Math.max(0.0d, Math.min(1.0d, normalized)) * 255.0d);
        return Math.max(0, Math.min(255, byteValue));
    }

    private static double imageValue(RandomAccess<?> access, long[] position, ImageLayout layout,
            int x, int y, int channel) {
        for (int d = 0; d < position.length; d++) {
            position[d] = 0;
        }
        position[layout.xDim] = x;
        position[layout.yDim] = y;
        if (layout.channelDim >= 0) {
            position[layout.channelDim] = Math.min(channel, layout.channels - 1);
        }
        return realValue(access, position);
    }

    private static void drawMaskContours(BufferedImage image, RandomAccessibleInterval<?> mask, Color color) {
        if (mask == null || mask.numDimensions() < 2) {
            return;
        }
        MaskLayout layout = MaskLayout.from(mask);
        RandomAccess<?> access = mask.randomAccess();
        long[] position = new long[mask.numDimensions()];
        int thickness = Math.max(1, Math.min(image.getWidth(), image.getHeight()) / 250);
        for (int y = 0; y < layout.height; y++) {
            for (int x = 0; x < layout.width; x++) {
                double value = maskValue(access, position, layout, x, y);
                if (value <= 0.0d || !isContour(access, position, layout, x, y, value)) {
                    continue;
                }
                int imageX = Math.max(0, Math.min(image.getWidth() - 1,
                        (int) Math.round((x + 0.5d) * image.getWidth() / layout.width - 0.5d)));
                int imageY = Math.max(0, Math.min(image.getHeight() - 1,
                        (int) Math.round((y + 0.5d) * image.getHeight() / layout.height - 0.5d)));
                paintSquare(image, imageX, imageY, thickness, color);
            }
        }
    }

    private static boolean isContour(RandomAccess<?> access, long[] position, MaskLayout layout,
            int x, int y, double value) {
        if (x == 0 || y == 0 || x == layout.width - 1 || y == layout.height - 1) {
            return true;
        }
        return maskValue(access, position, layout, x - 1, y) != value
                || maskValue(access, position, layout, x + 1, y) != value
                || maskValue(access, position, layout, x, y - 1) != value
                || maskValue(access, position, layout, x, y + 1) != value;
    }

    private static double maskValue(RandomAccess<?> access, long[] position, MaskLayout layout, int x, int y) {
        for (int d = 0; d < position.length; d++) {
            position[d] = 0;
        }
        position[layout.xDim] = x;
        position[layout.yDim] = y;
        return realValue(access, position);
    }

    private static double realValue(RandomAccess<?> access, long[] position) {
        access.setPosition(position);
        Object value = access.get();
        return value instanceof RealType ? ((RealType<?>) value).getRealDouble() : 0.0d;
    }

    private static void paintSquare(BufferedImage image, int centerX, int centerY, int radius, Color color) {
        for (int y = Math.max(0, centerY - radius); y <= Math.min(image.getHeight() - 1, centerY + radius); y++) {
            for (int x = Math.max(0, centerX - radius); x <= Math.min(image.getWidth() - 1, centerX + radius); x++) {
                image.setRGB(x, y, blend(image.getRGB(x, y), color, 0.82d));
            }
        }
    }

    private static int blend(int rgb, Color color, double alpha) {
        int r = (rgb >> 16) & 0xff;
        int g = (rgb >> 8) & 0xff;
        int b = rgb & 0xff;
        int nr = (int) Math.round(r * (1.0d - alpha) + color.getRed() * alpha);
        int ng = (int) Math.round(g * (1.0d - alpha) + color.getGreen() * alpha);
        int nb = (int) Math.round(b * (1.0d - alpha) + color.getBlue() * alpha);
        return (nr << 16) | (ng << 8) | nb;
    }

    private static int wrap(int index, int size) {
        if (size <= 0) {
            return 0;
        }
        int wrapped = index % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }

    private static String getString(JsonObject object, String key) {
        JsonElement element = object == null ? null : object.get(key);
        return element == null || element.isJsonNull() ? null : element.getAsString();
    }

    private static int getInt(JsonObject object, String key, int fallback) {
        JsonElement element = object == null ? null : object.get(key);
        if (element == null || element.isJsonNull()) {
            return fallback;
        }
        try {
            return element.getAsInt();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static String formatDuration(long millis) {
        long totalSeconds = Math.max(0L, millis / 1000L);
        long hours = totalSeconds / 3600L;
        long minutes = (totalSeconds % 3600L) / 60L;
        long seconds = totalSeconds % 60L;
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    private static final class PreviewImagePanel extends YoloImageDisplayPanel {

        private static final long serialVersionUID = -7590216692430039386L;

        private PreviewImagePanel() {
            super();
        }
    }

    private static final class ImageLayout {
        private final int xDim;
        private final int yDim;
        private final int channelDim;
        private final int width;
        private final int height;
        private final int channels;

        private ImageLayout(int xDim, int yDim, int channelDim, int width, int height, int channels) {
            this.xDim = xDim;
            this.yDim = yDim;
            this.channelDim = channelDim;
            this.width = width;
            this.height = height;
            this.channels = channels;
        }

        private static ImageLayout from(RandomAccessibleInterval<?> rai) {
            if (rai.numDimensions() == 2) {
                return new ImageLayout(1, 0, -1, safeInt(rai.dimension(1)), safeInt(rai.dimension(0)), 1);
            }
            if (rai.dimension(0) <= 4 && rai.dimension(1) > 4 && rai.dimension(2) > 4) {
                return new ImageLayout(2, 1, 0, safeInt(rai.dimension(2)),
                        safeInt(rai.dimension(1)), safeInt(rai.dimension(0)));
            }
            if (rai.dimension(2) <= 4 && rai.dimension(0) > 4 && rai.dimension(1) > 4) {
                return new ImageLayout(1, 0, 2, safeInt(rai.dimension(1)),
                        safeInt(rai.dimension(0)), safeInt(rai.dimension(2)));
            }
            return new ImageLayout(1, 0, -1, safeInt(rai.dimension(1)), safeInt(rai.dimension(0)), 1);
        }
    }

    private static final class MaskLayout {
        private final int xDim;
        private final int yDim;
        private final int width;
        private final int height;

        private MaskLayout(int xDim, int yDim, int width, int height) {
            this.xDim = xDim;
            this.yDim = yDim;
            this.width = width;
            this.height = height;
        }

        private static MaskLayout from(RandomAccessibleInterval<?> rai) {
            if (rai.numDimensions() >= 3 && rai.dimension(0) == 1) {
                return new MaskLayout(2, 1, safeInt(rai.dimension(2)), safeInt(rai.dimension(1)));
            }
            if (rai.numDimensions() >= 3 && rai.dimension(2) == 1) {
                return new MaskLayout(1, 0, safeInt(rai.dimension(1)), safeInt(rai.dimension(0)));
            }
            return new MaskLayout(1, 0, safeInt(rai.dimension(1)), safeInt(rai.dimension(0)));
        }
    }

    private static int safeInt(long value) {
        return (int) Math.max(1L, Math.min(Integer.MAX_VALUE, value));
    }

    private final class TrainingStatusPanel extends JPanel {

        private static final long serialVersionUID = 8312362540460559408L;

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (!trainingActive) {
                return;
            }
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setFont(getFont().deriveFont((float) Math.max(YoloUiUtils.MIN_FONT_SIZE,
                    Math.min(12, getHeight() / 3))));
            drawTrainingStatus(g2, 0, 0, getWidth());
            g2.dispose();
        }

        private void drawTrainingStatus(Graphics2D g2, int left, int top, int right) {
            FontMetrics fm = g2.getFontMetrics();
            int width = Math.max(1, right - left);
            int colW = Math.max(1, width / 3);
            int rowGap = Math.max(2, fm.getHeight() / 4);
            int row1 = top + fm.getAscent();
            int row2 = row1 + fm.getHeight() + rowGap;
            int singleRow = row1 + (row2 - row1) / 2;

            drawCenteredXLabel(g2, "elapsed " + formatDuration(elapsedMillis),
                    left, colW, singleRow);
            drawCenteredXLabel(g2, buildStepOnlyStatus(),
                    left + colW, colW, row1);
            drawCenteredXLabel(g2, buildEpochStatus(),
                    left + colW, colW, row2);
            drawCenteredXLabel(g2, buildSecondsPerStepStatus(),
                    left + 2 * colW, width - 2 * colW, row1);
            drawCenteredXLabel(g2, buildEpochRemainingStatus(),
                    left + 2 * colW, width - 2 * colW, row2);
        }

        private void drawCenteredXLabel(Graphics2D g2, String label, int x, int width, int y) {
            g2.setColor(TEXT_COLOR);
            FontMetrics fm = g2.getFontMetrics();
            String visibleLabel = ellipsize(label, fm, Math.max(1, width - 4));
            int labelX = x + Math.max(0, (width - fm.stringWidth(visibleLabel)) / 2);
            g2.drawString(visibleLabel, labelX, y);
        }

        private String buildStepOnlyStatus() {
            return "iteration (it) " + currentStep + "/" + totalSteps;
        }

        private String buildEpochStatus() {
            return "epoch (ep) " + currentEpoch() + "/" + totalEpochs;
        }

        private String buildSecondsPerStepStatus() {
            if (!isFinite(secondsPerIteration)) {
                return "-- s/it";
            }
            return String.format("%.2f s/it", secondsPerIteration);
        }

        private String buildEpochRemainingStatus() {
            if (!isFinite(secondsPerIteration)) {
                return "epoch ETA --:--:--";
            }
            int remainingSteps = remainingStepsInCurrentEpoch();
            if (remainingSteps < 0) {
                return "epoch ETA --:--:--";
            }
            return "epoch ETA " + formatDuration(Math.round(remainingSteps * secondsPerIteration * 1000.0d));
        }

        private int remainingStepsInCurrentEpoch() {
            int currentEpoch = currentEpoch();
            if (currentStep < 0 || totalSteps <= 0 || totalEpochs <= 0 || currentEpoch <= 0) {
                return -1;
            }
            int epochEndStep = (int) Math.ceil((double) currentEpoch * totalSteps / totalEpochs);
            return Math.max(0, epochEndStep - currentStep);
        }

        private int currentEpoch() {
            if (currentStep <= 0 || totalSteps <= 0 || totalEpochs <= 0) {
                return 0;
            }
            int epoch = (int) Math.ceil((double) currentStep * totalEpochs / totalSteps);
            return Math.max(1, Math.min(totalEpochs, epoch));
        }

        private String ellipsize(String text, FontMetrics fm, int maxWidth) {
            if (text == null || fm.stringWidth(text) <= maxWidth) {
                return text;
            }
            String suffix = "...";
            int suffixW = fm.stringWidth(suffix);
            if (suffixW >= maxWidth) {
                return suffix;
            }
            int end = text.length();
            while (end > 0 && fm.stringWidth(text.substring(0, end)) + suffixW > maxWidth) {
                end--;
            }
            return text.substring(0, end) + suffix;
        }
    }

    private static final class PreviewSample {
        private final String imagePath;
        private final String labelPath;
        private final String predictionPath;
        @SuppressWarnings("unused")
        private final String probPath;
        private final String title;

        private PreviewSample(String imagePath, String labelPath, String predictionPath,
                String probPath, String title) {
            this.imagePath = imagePath;
            this.labelPath = labelPath;
            this.predictionPath = predictionPath;
            this.probPath = probPath;
            this.title = title;
        }
    }
}
