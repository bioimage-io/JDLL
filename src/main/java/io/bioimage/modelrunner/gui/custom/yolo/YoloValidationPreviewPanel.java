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

import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.border.LineBorder;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class YoloValidationPreviewPanel extends JPanel {

    private static final long serialVersionUID = -4978636324362880905L;

    private static final int OUTER_PAD = 6;
    private static final int VIEWER_TO_ARROWS_GAP = 2;
    private static final int ARROWS_TO_STATUS_GAP = 4;
    private static final int ARROW_BUTTON_MIN_HEIGHT = 13;
    private static final int ARROW_BUTTON_MAX_HEIGHT = 21;
    private static final int STATUS_H = 36;
    private static final Color PREDICTION_BOX_COLOR = new Color(80, 220, 120);
    private static final Color TEXT_COLOR = new Color(70, 78, 98);
    private static final String PREVIOUS_SYMBOL = "\u25C0";
    private static final String NEXT_SYMBOL = "\u25B6";
    private static final String WAITING_MESSAGE = "Validation examples available after the first epoch finishes";
    private static final String ERROR_MESSAGE = "Could not load validation preview";

    private final YoloImageDisplayPanel imagePanel = new YoloImageDisplayPanel();
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

    public YoloValidationPreviewPanel() {
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
        add(statusPanel);
        add(previousButton);
        add(nextButton);
        clearPreview();
    }

    public void clearPreview() {
        samples.clear();
        currentIndex = 0;
        previewEpoch = 0;
        imagePanel.setEmptyMessage(WAITING_MESSAGE);
        imagePanel.clearImage();
        updateStatusPanel();
        updateButtons();
    }

    public void loadPreview(String jsonPath) {
        if (jsonPath == null || jsonPath.trim().isEmpty()) {
            return;
        }
        try {
            JsonObject root = readJson(jsonPath.trim());
            List<PreviewSample> loaded = parseSamples(root);
            String selectedPath = getSelectedImagePath();
            int selectedIndex = currentIndex;
            samples.clear();
            samples.addAll(loaded);
            previewEpoch = getInt(root, "epoch", previewEpoch);
            if (samples.isEmpty()) {
                currentIndex = 0;
                imagePanel.setEmptyMessage(WAITING_MESSAGE);
                imagePanel.clearImage();
            } else {
                currentIndex = selectUpdatedIndex(selectedPath, selectedIndex, samples);
                showSample(currentIndex);
            }
            updateStatusPanel();
            updateButtons();
        } catch (Exception e) {
            samples.clear();
            imagePanel.setEmptyMessage(ERROR_MESSAGE);
            imagePanel.clearImage();
            updateStatusPanel();
            updateButtons();
        }
    }

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
            BufferedImage image = ImageIO.read(new File(sample.imagePath));
            if (image == null) {
                throw new IOException("Unsupported image format: " + sample.imagePath);
            }
            imagePanel.setBufferedImage(image, sample.title, false);
            imagePanel.setReadOnlyBoxes(sample.boxes, PREDICTION_BOX_COLOR);
        } catch (IOException e) {
            imagePanel.setEmptyMessage(ERROR_MESSAGE);
            imagePanel.clearImage();
        }
        updateStatusPanel();
        updateButtons();
    }

    private void updateButtons() {
        boolean enabled = samples.size() > 1;
        previousButton.setEnabled(enabled);
        nextButton.setEnabled(enabled);
    }

    private String getSelectedImagePath() {
        if (samples.isEmpty() || currentIndex < 0 || currentIndex >= samples.size()) {
            return null;
        }
        return samples.get(currentIndex).imagePath;
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

    private void updateStatusPanel() {
        statusPanel.repaint();
    }

    private static JsonObject readJson(String jsonPath) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(jsonPath)), StandardCharsets.UTF_8);
        return JsonParser.parseString(content).getAsJsonObject();
    }

    private static List<PreviewSample> parseSamples(JsonObject root) {
        JsonArray images = root == null ? null : root.getAsJsonArray("images");
        if (images == null || images.size() == 0) {
            return Collections.emptyList();
        }
        List<PreviewSample> result = new ArrayList<PreviewSample>();
        for (JsonElement element : images) {
            if (!element.isJsonObject()) {
                continue;
            }
            JsonObject image = element.getAsJsonObject();
            String path = getString(image, "path");
            if (path == null || path.trim().isEmpty()) {
                continue;
            }
            result.add(new PreviewSample(
                    path,
                    new File(path).getName(),
                    parseBoxes(image.getAsJsonArray("boxes"))));
        }
        return result;
    }

    private static List<Rectangle2D.Double> parseBoxes(JsonArray boxArray) {
        List<Rectangle2D.Double> result = new ArrayList<Rectangle2D.Double>();
        if (boxArray == null) {
            return result;
        }
        for (JsonElement element : boxArray) {
            if (!element.isJsonObject()) {
                continue;
            }
            JsonObject box = element.getAsJsonObject();
            JsonArray xyxy = box.getAsJsonArray("xyxy");
            if (xyxy == null || xyxy.size() < 4) {
                continue;
            }
            double x1 = getDouble(xyxy.get(0), Double.NaN);
            double y1 = getDouble(xyxy.get(1), Double.NaN);
            double x2 = getDouble(xyxy.get(2), Double.NaN);
            double y2 = getDouble(xyxy.get(3), Double.NaN);
            if (!isFinite(x1) || !isFinite(y1) || !isFinite(x2) || !isFinite(y2)) {
                continue;
            }
            double w = Math.max(0.0d, x2 - x1);
            double h = Math.max(0.0d, y2 - y1);
            if (w <= 0.0d || h <= 0.0d) {
                continue;
            }
            result.add(new Rectangle2D.Double(x1, y1, w, h));
        }
        return result;
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

    private static double getDouble(JsonElement element, double fallback) {
        if (element == null || element.isJsonNull()) {
            return fallback;
        }
        try {
            return element.getAsDouble();
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
        private final String title;
        private final List<Rectangle2D.Double> boxes;

        private PreviewSample(String imagePath, String title, List<Rectangle2D.Double> boxes) {
            this.imagePath = imagePath;
            this.title = title;
            this.boxes = boxes == null ? Collections.<Rectangle2D.Double>emptyList() : boxes;
        }
    }
}
