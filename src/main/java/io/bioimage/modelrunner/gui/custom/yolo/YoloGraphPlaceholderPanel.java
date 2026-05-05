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

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.swing.JPanel;
import javax.swing.border.LineBorder;

public class YoloGraphPlaceholderPanel extends JPanel {

    private static final long serialVersionUID = -5079977598122379171L;

    private static final int PAD = 12;
    private static final int TOP_INFO_H = 24;
    private static final int X_AXIS_LABEL_H = 22;
    private static final int STATUS_GAP_H = 14;
    private static final int STATUS_H = 36;
    private static final int Y_TICKS = 4;
    private static final int X_TICKS = 3;
    private static final int X_TICK_MARK_H = 5;
    private static final Color AXIS_COLOR = new Color(132, 142, 160);
    private static final Color GRID_COLOR = new Color(229, 233, 241);
    private static final Color TEXT_COLOR = new Color(70, 78, 98);
    private static final Color TRAIN_COLOR = new Color(38, 111, 201);
    private static final Color VALIDATION_COLOR = new Color(224, 112, 36);

    private final String fallbackTitle;
    private final LinkedHashMap<String, Series> series = new LinkedHashMap<String, Series>();
    private String selectedMetricName;
    private boolean trainingActive;
    private int currentStep;
    private int totalSteps;
    private int currentEpoch;
    private int totalEpochs;
    private long elapsedMillis;
    private double secondsPerStep = Double.NaN;

    public YoloGraphPlaceholderPanel(String title) {
        this.fallbackTitle = title;
        setOpaque(true);
        setBackground(Color.WHITE);
        setBorder(new LineBorder(new Color(205, 210, 221)));
        setDefaultSeries();
    }

    public void addValue(Double value) {
        addTrainValue(fallbackTitle, valuesCount() + 1, valuesCount() + 1, value);
    }

    public void addTrainValue(String metricName, int step, int epoch, Double value) {
        addValue("Training", metricName, step, epoch, value, TRAIN_COLOR);
    }

    public void addValidationValue(String metricName, int step, int epoch, Double value) {
        addValue("Validation", metricName, step, epoch, value, VALIDATION_COLOR);
    }

    public void addValue(String seriesName, String metricName, int step, int epoch, Double value, Color color) {
        if (value == null || value.isNaN() || value.isInfinite()) {
            return;
        }
        String cleanMetric = metricName == null || metricName.trim().isEmpty() ? fallbackTitle : metricName;
        selectedMetricName = fallbackTitle + " - " + cleanMetric;
        Series s = series.get(seriesName);
        if (s == null) {
            s = new Series(seriesName, color == null ? TRAIN_COLOR : color);
            series.put(seriesName, s);
        }
        s.points.add(new Point(Math.max(1, step), Math.max(1, epoch), value));
        repaint();
    }

    public void clearValues() {
        series.clear();
        selectedMetricName = null;
        trainingActive = false;
        currentStep = 0;
        totalSteps = 0;
        currentEpoch = 0;
        totalEpochs = 0;
        elapsedMillis = 0L;
        secondsPerStep = Double.NaN;
        setDefaultSeries();
        repaint();
    }

    public void setTrainingStatus(boolean active, int currentStep, int totalSteps,
            long elapsedMillis, double secondsPerStep) {
        setTrainingStatus(active, currentStep, totalSteps, currentEpoch, totalEpochs, elapsedMillis, secondsPerStep);
    }

    public void setTrainingStatus(boolean active, int currentStep, int totalSteps,
            int totalEpochs, long elapsedMillis, double secondsPerStep) {
        setTrainingStatus(active, currentStep, totalSteps,
                deriveCurrentEpoch(currentStep, totalSteps, totalEpochs),
                totalEpochs, elapsedMillis, secondsPerStep);
    }

    public void setTrainingStatus(boolean active, int currentStep, int totalSteps,
            int currentEpoch, int totalEpochs, long elapsedMillis, double secondsPerStep) {
        this.trainingActive = active;
        this.currentStep = Math.max(0, currentStep);
        this.totalSteps = Math.max(0, totalSteps);
        this.currentEpoch = Math.max(0, currentEpoch);
        this.totalEpochs = Math.max(0, totalEpochs);
        this.elapsedMillis = Math.max(0L, elapsedMillis);
        this.secondsPerStep = secondsPerStep;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setFont(getFont().deriveFont((float) Math.max(YoloUiUtils.MIN_FONT_SIZE, Math.min(12, getHeight() / 13))));

        int left = Math.max(44, getWidth() / 10);
        int top = PAD + TOP_INFO_H;
        int right = Math.max(left + 40, getWidth() - PAD);
        int bottom = Math.max(top + 40, getHeight() - PAD - statusHeight());
        PlotRange range = buildRange();

        drawTopInfo(g2, left, right);
        drawGridAndAxes(g2, left, top, right, bottom, range);
        drawSeries(g2, left, top, right, bottom, range);
        drawLegend(g2, left, top, right);
        g2.dispose();
    }

    private void drawTopInfo(Graphics2D g2, int left, int right) {
        g2.setColor(TEXT_COLOR);
        FontMetrics fm = g2.getFontMetrics();
        String metric = selectedMetricName == null ? fallbackTitle : selectedMetricName;

        Point latest = latestPoint();
        String value = "";
        if (latest != null) {
            value = String.format("%.4g", latest.value);
            int x = right - fm.stringWidth(value);
            g2.drawString(value, x, PAD + fm.getAscent());
        }
        int availableMetricW = Math.max(1, right - left - fm.stringWidth(value) - 12);
        g2.drawString(ellipsize(metric, fm, availableMetricW), left, PAD + fm.getAscent());
    }

    private void drawGridAndAxes(Graphics2D g2, int left, int top, int right, int bottom, PlotRange range) {
        FontMetrics fm = g2.getFontMetrics();
        for (int i = 0; i <= Y_TICKS; i++) {
            double ratio = (double) i / Y_TICKS;
            int y = bottom - (int) Math.round(ratio * (bottom - top));
            double value = range.minY + ratio * (range.maxY - range.minY);
            g2.setColor(GRID_COLOR);
            g2.drawLine(left, y, right, y);
            g2.setColor(TEXT_COLOR);
            String label = String.format("%.3g", value);
            g2.drawString(label, Math.max(2, left - fm.stringWidth(label) - 6), y + fm.getAscent() / 2 - 2);
        }

        g2.setColor(AXIS_COLOR);
        g2.drawLine(left, bottom, right, bottom);
        g2.drawLine(left, top, left, bottom);
        drawXTickLabels(g2, left, bottom, right, range);

        if (trainingActive) {
            drawTrainingStatus(g2, left, bottom + X_AXIS_LABEL_H + STATUS_GAP_H, right);
        }
    }

    private void drawXTickLabels(Graphics2D g2, int left, int bottom, int right, PlotRange range) {
        if (!range.hasValues || (valuesCount() == 0 && totalSteps <= 0)) {
            return;
        }
        FontMetrics fm = g2.getFontMetrics();
        int y = bottom + Math.max(2, fm.getHeight() / 4) + fm.getAscent();
        int previousEnd = Integer.MIN_VALUE;
        int denominator = Math.max(1, X_TICKS - 1);
        for (int i = 0; i < X_TICKS; i++) {
            int step = range.minStep + (int) Math.round((range.maxStep - range.minStep) * ((double) i / denominator));
            int epoch = epochForStep(step);
            int x = xFor(step, left, right, range);
            if (i > 0) {
                g2.setColor(AXIS_COLOR);
                g2.drawLine(x, bottom - X_TICK_MARK_H, x, bottom + X_TICK_MARK_H);
            }
            String label = epoch <= 0 ? "it " + step : "it " + step + " | ep " + epoch;
            int labelW = fm.stringWidth(label);
            int labelX = Math.max(left, Math.min(right - labelW, x - labelW / 2));
            if (labelX <= previousEnd + 6) {
                continue;
            }
            g2.setColor(TEXT_COLOR);
            g2.drawString(label, labelX, y);
            previousEnd = labelX + labelW;
        }
    }

    private void drawTrainingStatus(Graphics2D g2, int left, int top, int right) {
        FontMetrics fm = g2.getFontMetrics();
        int width = Math.max(1, right - left);
        int colW = Math.max(1, width / 3);
        int rowGap = Math.max(2, fm.getHeight() / 4);
        int row1 = top + fm.getAscent();
        int row2 = row1 + fm.getHeight() + rowGap;
        int singleRow = row1 + (row2 - row1) / 2;

        drawCenteredXLabel(g2, "elapsed " + formatElapsed(elapsedMillis),
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

    private void drawSeries(Graphics2D g2, int left, int top, int right, int bottom, PlotRange range) {
        g2.setStroke(new BasicStroke(2f));
        for (Series s : series.values()) {
            if (s.points.isEmpty()) {
                continue;
            }
            int n = s.points.size();
            int[] xs = new int[n];
            int[] ys = new int[n];
            for (int i = 0; i < n; i++) {
                Point p = s.points.get(i);
                xs[i] = xFor(p.step, left, right, range);
                ys[i] = yFor(p.value, top, bottom, range);
            }
            g2.setColor(s.color);
            if (n > 1) {
                g2.drawPolyline(xs, ys, n);
            }
            for (int i = 0; i < n; i++) {
                g2.fill(new Ellipse2D.Double(xs[i] - 2, ys[i] - 2, 4, 4));
            }
        }
    }

    private void drawLegend(Graphics2D g2, int left, int top, int right) {
        FontMetrics fm = g2.getFontMetrics();
        List<Series> visibleSeries = visibleSeries();
        if (visibleSeries.isEmpty()) {
            return;
        }
        int totalW = 0;
        for (Series s : visibleSeries) {
            totalW += 26 + fm.stringWidth(s.name);
        }
        int x = Math.max(left, right - totalW);
        int y = top + 4;
        for (Series s : visibleSeries) {
            int labelW = fm.stringWidth(s.name);
            if (x + 18 + labelW > right) {
                break;
            }
            g2.setColor(s.color);
            g2.fillRect(x, y + 5, 12, 4);
            g2.setColor(TEXT_COLOR);
            g2.drawString(s.name, x + 18, y + fm.getAscent());
            x += 26 + labelW;
        }
    }

    private int xFor(int step, int left, int right, PlotRange range) {
        if (range.maxStep == range.minStep) {
            return left;
        }
        double ratio = (double) (step - range.minStep) / (double) (range.maxStep - range.minStep);
        return left + (int) Math.round(ratio * (right - left));
    }

    private int yFor(double value, int top, int bottom, PlotRange range) {
        double ratio = (value - range.minY) / (range.maxY - range.minY);
        return bottom - (int) Math.round(ratio * (bottom - top));
    }

    private PlotRange buildRange() {
        PlotRange range = new PlotRange();
        for (Series s : series.values()) {
            for (Point p : s.points) {
                range.accept(p);
            }
        }
        if (!range.hasValues) {
            range.accept(new Point(1, 1, 0.0));
            range.accept(new Point(Math.max(2, totalSteps), 1, 1.0));
        }
        if (range.maxY == range.minY) {
            range.maxY = range.minY + 1.0;
        }
        double pad = (range.maxY - range.minY) * 0.08;
        range.minY -= pad;
        range.maxY += pad;
        return range;
    }

    private int epochForStep(int step) {
        Point nearest = null;
        int nearestDistance = Integer.MAX_VALUE;
        for (Series s : series.values()) {
            for (Point p : s.points) {
                int distance = Math.abs(p.step - step);
                if (nearest == null || distance < nearestDistance) {
                    nearest = p;
                    nearestDistance = distance;
                }
            }
        }
        if (nearest != null) {
            return nearest.epoch;
        }
        int derived = deriveCurrentEpoch(step, totalSteps, totalEpochs);
        return derived > 0 ? derived : currentEpoch;
    }

    private Point latestPoint() {
        Point latest = null;
        for (Series s : series.values()) {
            for (Point p : s.points) {
                if (latest == null || p.step >= latest.step) {
                    latest = p;
                }
            }
        }
        return latest;
    }

    private int valuesCount() {
        int count = 0;
        for (Series s : series.values()) {
            count += s.points.size();
        }
        return count;
    }

    private void setDefaultSeries() {
        series.put("Training", new Series("Training", TRAIN_COLOR));
    }

    private List<Series> visibleSeries() {
        List<Series> visible = new ArrayList<Series>();
        for (Series s : series.values()) {
            if (!s.points.isEmpty()) {
                visible.add(s);
            }
        }
        return visible;
    }

    private String buildStepOnlyStatus() {
        return "iteration (it) " + currentStep + "/" + totalSteps;
    }

    private String buildEpochStatus() {
        return "epoch (ep) " + currentEpoch + "/" + totalEpochs;
    }

    private String buildSecondsPerStepStatus() {
        if (!isFinite(secondsPerStep)) {
            return "-- s/it";
        }
        return String.format("%.2f s/it", secondsPerStep);
    }

    private String buildEpochRemainingStatus() {
        if (!isFinite(secondsPerStep)) {
            return "epoch ETA --:--:--";
        }
        int remainingSteps = remainingStepsInCurrentEpoch();
        if (remainingSteps < 0) {
            return "epoch ETA --:--:--";
        }
        return "epoch ETA " + formatElapsed(Math.round(remainingSteps * secondsPerStep * 1000.0d));
    }

    private int remainingStepsInCurrentEpoch() {
        if (currentStep < 0 || totalSteps <= 0 || totalEpochs <= 0 || currentEpoch <= 0) {
            return -1;
        }
        int epochEndStep = (int) Math.ceil((double) currentEpoch * totalSteps / totalEpochs);
        return Math.max(0, epochEndStep - currentStep);
    }

    private int statusHeight() {
        return X_AXIS_LABEL_H + (trainingActive ? STATUS_GAP_H + STATUS_H : 0);
    }

    private static String formatElapsed(long elapsedMillis) {
        long totalSeconds = elapsedMillis / 1000L;
        long hours = totalSeconds / 3600L;
        long minutes = (totalSeconds % 3600L) / 60L;
        long seconds = totalSeconds % 60L;
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    private static int deriveCurrentEpoch(int currentStep, int totalSteps, int totalEpochs) {
        if (currentStep <= 0 || totalSteps <= 0 || totalEpochs <= 0) {
            return 0;
        }
        int epoch = (int) Math.ceil((double) currentStep * totalEpochs / totalSteps);
        return Math.max(1, Math.min(totalEpochs, epoch));
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static String ellipsize(String text, FontMetrics fm, int maxWidth) {
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

    private static final class Series {
        private final String name;
        private final Color color;
        private final List<Point> points = new ArrayList<Point>();

        private Series(String name, Color color) {
            this.name = name;
            this.color = color;
        }
    }

    private static final class Point {
        private final int step;
        private final int epoch;
        private final double value;

        private Point(int step, int epoch, double value) {
            this.step = step;
            this.epoch = epoch;
            this.value = value;
        }
    }

    private static final class PlotRange {
        private boolean hasValues;
        private int minStep = Integer.MAX_VALUE;
        private int maxStep = Integer.MIN_VALUE;
        private int minEpoch = Integer.MAX_VALUE;
        private int maxEpoch = Integer.MIN_VALUE;
        private double minY = Double.POSITIVE_INFINITY;
        private double maxY = Double.NEGATIVE_INFINITY;

        private void accept(Point p) {
            hasValues = true;
            minStep = Math.min(minStep, p.step);
            maxStep = Math.max(maxStep, p.step);
            minEpoch = Math.min(minEpoch, p.epoch);
            maxEpoch = Math.max(maxEpoch, p.epoch);
            minY = Math.min(minY, p.value);
            maxY = Math.max(maxY, p.value);
        }
    }
}
