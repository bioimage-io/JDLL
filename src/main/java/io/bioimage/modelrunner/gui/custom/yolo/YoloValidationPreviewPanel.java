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
import java.awt.image.BufferedImage;
import java.awt.geom.Rectangle2D;
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
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.border.LineBorder;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class YoloValidationPreviewPanel extends JPanel {

    private static final long serialVersionUID = -4978636324362880905L;

    private static final int OUTER_PAD = 6;
    private static final int GAP = 5;
    private static final int ARROW_BUTTON_MIN_SIZE = 20;
    private static final int ARROW_BUTTON_MAX_SIZE = 32;
    private static final int STATUS_MAX_HEIGHT = 28;
    private static final Color PREDICTION_BOX_COLOR = new Color(232, 72, 72);
    private static final String PREVIOUS_SYMBOL = "\u25C0";
    private static final String NEXT_SYMBOL = "\u25B6";
    private static final String WAITING_MESSAGE = "Validation examples available after the first epoch finishes";
    private static final String ERROR_MESSAGE = "Could not load validation preview";

    private final YoloImageDisplayPanel imagePanel = new YoloImageDisplayPanel();
    private final JLabel statusLabel = new JLabel("", SwingConstants.CENTER);
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
        statusLabel.setForeground(new Color(70, 78, 98));
        YoloUiUtils.styleFlatSecondaryButton(previousButton);
        YoloUiUtils.styleFlatSecondaryButton(nextButton);
        previousButton.addActionListener(e -> showSample(currentIndex - 1));
        nextButton.addActionListener(e -> showSample(currentIndex + 1));

        add(imagePanel);
        add(statusLabel);
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
        updateStatusLabel();
        updateButtons();
    }

    public void loadPreview(String jsonPath) {
        if (jsonPath == null || jsonPath.trim().isEmpty()) {
            return;
        }
        try {
            JsonObject root = readJson(jsonPath.trim());
            List<PreviewSample> loaded = parseSamples(root);
            samples.clear();
            samples.addAll(loaded);
            previewEpoch = getInt(root, "epoch", previewEpoch);
            currentIndex = 0;
            if (samples.isEmpty()) {
                imagePanel.setEmptyMessage(WAITING_MESSAGE);
                imagePanel.clearImage();
            } else {
                showSample(currentIndex);
            }
            updateStatusLabel();
            updateButtons();
        } catch (Exception e) {
            samples.clear();
            imagePanel.setEmptyMessage(ERROR_MESSAGE);
            imagePanel.clearImage();
            updateStatusLabel();
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
        updateStatusLabel();
    }

    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int innerW = Math.max(1, w - 2 * OUTER_PAD);
        int arrowSize = Math.max(ARROW_BUTTON_MIN_SIZE,
                Math.min(ARROW_BUTTON_MAX_SIZE, Math.min(w, h) / 10));
        int statusH = Math.max(YoloUiUtils.MIN_FONT_SIZE + 4,
                Math.min(STATUS_MAX_HEIGHT, h / 9));
        int bottomH = arrowSize + GAP + statusH;
        int imageH = Math.max(1, h - 2 * OUTER_PAD - bottomH - GAP);
        int y = OUTER_PAD;

        imagePanel.setBounds(OUTER_PAD, y, innerW, imageH);
        y += imageH + GAP;
        statusLabel.setBounds(OUTER_PAD, y, innerW, statusH);
        y += statusH + GAP;

        int centerX = w / 2;
        int arrowGap = Math.max(4, arrowSize / 3);
        previousButton.setBounds(centerX - arrowGap / 2 - arrowSize, y, arrowSize, arrowSize);
        nextButton.setBounds(centerX + arrowGap / 2, y, arrowSize, arrowSize);

        int statusFontSize = Math.max(YoloUiUtils.MIN_FONT_SIZE,
                Math.min(YoloUiUtils.MAX_CONTROL_FONT_SIZE,
                        (int) Math.floor(statusH * YoloUiUtils.CONTROL_FONT_HEIGHT_RATIO)));
        statusLabel.setFont(statusLabel.getFont().deriveFont((float) statusFontSize));
        YoloUiUtils.applyResponsiveText(previousButton, arrowSize - 4, arrowSize);
        YoloUiUtils.applyResponsiveText(nextButton, arrowSize - 4, arrowSize);
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
            imagePanel.setBufferedImage(image, sample.title);
            imagePanel.setReadOnlyBoxes(sample.boxes, PREDICTION_BOX_COLOR);
        } catch (IOException e) {
            imagePanel.setEmptyMessage(ERROR_MESSAGE);
            imagePanel.clearImage();
        }
        updateStatusLabel();
        updateButtons();
    }

    private void updateButtons() {
        boolean enabled = samples.size() > 1;
        previousButton.setEnabled(enabled);
        nextButton.setEnabled(enabled);
    }

    private void updateStatusLabel() {
        String preview = previewEpoch <= 0 || samples.isEmpty()
                ? "validation preview pending"
                : "epoch " + previewEpoch + " preview " + (currentIndex + 1) + "/" + samples.size();
        statusLabel.setText(preview + "    " + trainingStatusText());
    }

    private String trainingStatusText() {
        int epoch = deriveEpoch();
        String iteration = totalSteps > 0
                ? "it " + currentStep + "/" + totalSteps
                : "it " + currentStep + "/?";
        String epochText = totalEpochs > 0
                ? "ep " + epoch + "/" + totalEpochs
                : "ep ?";
        String speed = isFinite(secondsPerIteration)
                ? String.format("%.2f s/it", secondsPerIteration)
                : "-- s/it";
        return "elapsed " + formatDuration(elapsedMillis) + "    "
                + iteration + " | " + epochText + "    "
                + speed + " | epoch ETA " + formatEpochEta(epoch);
    }

    private String formatEpochEta(int epoch) {
        if (!trainingActive || totalSteps <= 0 || totalEpochs <= 0 || !isFinite(secondsPerIteration)) {
            return "--:--:--";
        }
        double stepsPerEpoch = totalSteps / (double) totalEpochs;
        int epochEndStep = (int) Math.ceil(Math.max(1, epoch) * stepsPerEpoch);
        int remainingSteps = Math.max(0, epochEndStep - currentStep);
        return formatDuration((long) Math.round(remainingSteps * secondsPerIteration * 1000.0d));
    }

    private int deriveEpoch() {
        if (totalSteps <= 0 || totalEpochs <= 0 || currentStep <= 0) {
            return 0;
        }
        int epoch = (int) Math.ceil(currentStep * totalEpochs / (double) totalSteps);
        return Math.max(1, Math.min(totalEpochs, epoch));
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
