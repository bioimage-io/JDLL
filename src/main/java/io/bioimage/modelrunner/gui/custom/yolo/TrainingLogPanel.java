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
import java.awt.Component;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.datatransfer.StringSelection;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

import javax.swing.JButton;
import javax.swing.Icon;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.border.LineBorder;
import javax.swing.text.DefaultCaret;
import javax.swing.text.AttributeSet;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;
import javax.swing.text.Element;
import javax.swing.text.Style;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledDocument;

public class TrainingLogPanel extends JPanel {

    private static final long serialVersionUID = -3551012784767392591L;

    private static final int PAD = 6;
    private static final int STATUS_H = 36;
    private static final int CONTROLS_H = 28;
    private static final int MAX_UI_LINES = 2000;
    private static final String PLACEHOLDER = "Nothing yet...";
    private static final String ACTIVITY_TEXT = "Working";
    private static final DateTimeFormatter TIME_FORMAT = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final Color TEXT_COLOR = new Color(70, 78, 98);

    private final JTextPane textPane = new NoWrapTextPane();
    private final StyledDocument logDocument = textPane.getStyledDocument();
    private final JScrollPane scrollPane = new JScrollPane(textPane);
    private final JButton copyButton = new JButton("Copy");
    private final JButton clearButton = new JButton("Clear");
    private final StatusPanel statusPanel = new StatusPanel();
    private final Timer activityTimer;
    private final Style timestampStyle;
    private final Style messageStyle;
    private final Style placeholderStyle;
    private final Style activityStyle;
    private boolean followTail = true;
    private boolean adjustingScroll;
    private boolean suppressScrollTracking;
    private boolean trainingActive;
    private boolean placeholderVisible;
    private int activityLineStart = -1;
    private int activityDots;
    private BufferedWriter diskWriter;
    private File logFile;
    private int uiLineCount;
    private int currentStep;
    private int totalSteps;
    private int currentEpoch;
    private int totalEpochs;
    private long elapsedMillis;
    private double secondsPerStep = Double.NaN;

    /**
     * Creates a new TrainingLogPanel instance.
     */
    public TrainingLogPanel() {
        setLayout(null);
        setOpaque(true);
        setBackground(Color.WHITE);

        timestampStyle = logDocument.addStyle("timestamp", null);
        StyleConstants.setFontFamily(timestampStyle, Font.MONOSPACED);
        StyleConstants.setFontSize(timestampStyle, 11);
        StyleConstants.setBold(timestampStyle, true);
        StyleConstants.setForeground(timestampStyle, new Color(104, 115, 138));

        messageStyle = logDocument.addStyle("message", null);
        StyleConstants.setFontFamily(messageStyle, Font.MONOSPACED);
        StyleConstants.setFontSize(messageStyle, 12);
        StyleConstants.setForeground(messageStyle, new Color(31, 38, 54));

        placeholderStyle = logDocument.addStyle("placeholder", null);
        StyleConstants.setFontFamily(placeholderStyle, Font.MONOSPACED);
        StyleConstants.setFontSize(placeholderStyle, 12);
        StyleConstants.setItalic(placeholderStyle, true);
        StyleConstants.setForeground(placeholderStyle, new Color(138, 147, 166));

        activityStyle = logDocument.addStyle("activity", null);
        StyleConstants.setFontFamily(activityStyle, Font.MONOSPACED);
        StyleConstants.setFontSize(activityStyle, 12);
        StyleConstants.setItalic(activityStyle, true);
        StyleConstants.setForeground(activityStyle, new Color(44, 111, 179));

        textPane.setEditable(false);
        textPane.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        textPane.setBackground(Color.WHITE);
        textPane.setForeground(new Color(35, 42, 56));
        ((DefaultCaret) textPane.getCaret()).setUpdatePolicy(DefaultCaret.NEVER_UPDATE);
        scrollPane.getViewport().setBackground(Color.WHITE);
        scrollPane.setBorder(new LineBorder(new Color(205, 210, 221)));
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        scrollPane.getVerticalScrollBar().addAdjustmentListener(e -> {
            if (!adjustingScroll && !suppressScrollTracking) {
                followTail = isAtBottom();
            }
        });

        copyButton.setText("");
        copyButton.setIcon(new CopyIcon());
        copyButton.setToolTipText("Copy the full training log to the clipboard");
        styleLogButton(copyButton);
        copyButton.addActionListener(e -> copyLogToClipboard());

        clearButton.setText("");
        clearButton.setIcon(new EraserIcon());
        clearButton.setToolTipText("Clear the visible log");
        styleLogButton(clearButton);
        clearButton.addActionListener(e -> clearVisibleLog());

        activityTimer = new Timer(550, e -> updateActivityLine());

        statusPanel.setOpaque(false);
        add(scrollPane);
        add(statusPanel);
        add(clearButton);
        add(copyButton);
        setComponentZOrder(copyButton, 0);
        setComponentZOrder(clearButton, 0);
        showPlaceholder();
    }

    /**
     * Appends a timestamped log line.
     *
     * @param message the log message.
     */
    public void appendLine(String message) {
        if (message == null || message.trim().isEmpty()) {
            return;
        }
        String timestamp = LocalTime.now().format(TIME_FORMAT);
        String line = "[" + timestamp + "] " + message.trim();
        String diskError = writeDiskLine(line);
        Runnable append = () -> {
            boolean shouldFollow = followTail && isAtBottom();
            withScrollTrackingSuppressed(() -> {
                appendUiLine(timestamp, message.trim(), messageStyle);
                if (diskError != null) {
                    appendUiLine(LocalTime.now().format(TIME_FORMAT),
                            "Could not write training UI log to disk: " + diskError,
                            messageStyle);
                }
            });
            if (shouldFollow) {
                scrollToBottom();
            } else {
                followTail = false;
            }
        };
        if (SwingUtilities.isEventDispatchThread()) {
            append.run();
        } else {
            SwingUtilities.invokeLater(append);
        }
    }

    /**
     * Clears the log.
     */
    public void clearLog() {
        Runnable clear = () -> {
            clearVisibleDocument();
            uiLineCount = 0;
            followTail = true;
            showPlaceholder();
            setTrainingStatus(false, 0, 0, 0, 0L, Double.NaN);
        };
        if (SwingUtilities.isEventDispatchThread()) {
            clear.run();
        } else {
            SwingUtilities.invokeLater(clear);
        }
    }

    /**
     * Starts persisting appended log lines to {@code training-ui.log} inside the output folder.
     *
     * @param outputDir the training output folder.
     */
    public void startDiskLog(File outputDir) {
        closeDiskLog();
        if (outputDir == null) {
            return;
        }
        File target = outputDir.isDirectory() || !outputDir.getName().endsWith(".log")
                ? new File(outputDir, "training-ui.log")
                : outputDir;
        try {
            File parent = target.getParentFile();
            if (parent != null) {
                Files.createDirectories(parent.toPath());
            }
            diskWriter = Files.newBufferedWriter(target.toPath(), StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
            logFile = target;
        } catch (IOException e) {
            diskWriter = null;
            logFile = null;
            appendLine("Could not create training UI log at " + target.getAbsolutePath() + ": "
                    + e.getMessage());
        }
    }

    /**
     * Closes the disk log.
     */
    public synchronized void closeDiskLog() {
        if (diskWriter == null) {
            return;
        }
        try {
            diskWriter.flush();
            diskWriter.close();
        } catch (IOException e) {
            // The UI log has already been written; failing close should not affect training UI cleanup.
        } finally {
            diskWriter = null;
        }
    }

    /**
     * Gets the current disk log file.
     *
     * @return the disk log file, or null if disabled.
     */
    public File getLogFile() {
        return logFile;
    }

    /**
     * Sets the training status.
     *
     * @param active whether training is active.
     * @param currentStep the current step.
     * @param totalSteps the total steps.
     * @param totalEpochs the total epochs.
     * @param elapsedMillis elapsed milliseconds.
     * @param secondsPerStep seconds per step.
     */
    public void setTrainingStatus(boolean active, int currentStep, int totalSteps,
            int totalEpochs, long elapsedMillis, double secondsPerStep) {
        setTrainingStatus(active, currentStep, totalSteps,
                deriveCurrentEpoch(currentStep, totalSteps, totalEpochs),
                totalEpochs, elapsedMillis, secondsPerStep);
    }

    /**
     * Sets the training status.
     *
     * @param active whether training is active.
     * @param currentStep the current step.
     * @param totalSteps the total steps.
     * @param currentEpoch the current epoch.
     * @param totalEpochs the total epochs.
     * @param elapsedMillis elapsed milliseconds.
     * @param secondsPerStep seconds per step.
     */
    public void setTrainingStatus(boolean active, int currentStep, int totalSteps,
            int currentEpoch, int totalEpochs, long elapsedMillis, double secondsPerStep) {
        this.trainingActive = active;
        this.currentStep = Math.max(0, currentStep);
        this.totalSteps = Math.max(0, totalSteps);
        this.currentEpoch = Math.max(0, currentEpoch);
        this.totalEpochs = Math.max(0, totalEpochs);
        this.elapsedMillis = Math.max(0L, elapsedMillis);
        this.secondsPerStep = secondsPerStep;
        if (active) {
            startActivityTimer();
        } else {
            stopActivityTimer();
        }
        statusPanel.repaint();
    }

    /**
     * Performs layout.
     */
    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int statusH = trainingActive ? Math.max(YoloUiUtils.MIN_FONT_SIZE * 2, Math.min(STATUS_H, h / 5)) : 0;
        int controlsH = Math.max(22, Math.min(CONTROLS_H, Math.max(1, h / 7)));
        int scrollY = PAD;
        int scrollH = Math.max(1, h - 3 * PAD - controlsH - statusH - (statusH > 0 ? PAD : 0));
        int controlsY = scrollY + scrollH + PAD;
        int clearW = Math.max(28, Math.min(36, w / 12));
        int copyW = Math.max(28, Math.min(36, w / 12));
        int buttonH = Math.max(20, Math.min(24, controlsH));
        int copyX = Math.max(PAD, w - PAD - copyW);
        int clearX = Math.max(PAD, copyX - PAD - clearW);
        scrollPane.setBounds(PAD, scrollY, Math.max(1, w - 2 * PAD), scrollH);
        clearButton.setBounds(clearX, controlsY, clearW, buttonH);
        copyButton.setBounds(copyX, controlsY, copyW, buttonH);
        statusPanel.setBounds(PAD, controlsY + controlsH + (statusH > 0 ? PAD : 0),
                Math.max(1, w - 2 * PAD), statusH);
    }

    private void appendUiLine(String timestamp, String message, AttributeSet style) {
        try {
            removeActivityLine();
            if (placeholderVisible) {
                clearVisibleDocument();
            }
            logDocument.insertString(logDocument.getLength(), "[" + timestamp + "] ", timestampStyle);
            logDocument.insertString(logDocument.getLength(), message + System.lineSeparator(), style);
            uiLineCount++;
            trimUiLog();
            appendActivityLine();
        } catch (BadLocationException e) {
            // Ignore UI logging failures. The disk log remains the source of truth.
        }
    }

    private void trimUiLog() {
        removeActivityLine();
        Document document = textPane.getDocument();
        while (uiLineCount > MAX_UI_LINES) {
            Element root = document.getDefaultRootElement();
            if (root.getElementCount() <= 1) {
                uiLineCount = Math.min(uiLineCount, MAX_UI_LINES);
                return;
            }
            try {
                document.remove(0, root.getElement(0).getEndOffset());
                uiLineCount--;
            } catch (BadLocationException e) {
                return;
            }
        }
        appendActivityLine();
    }

    private synchronized String writeDiskLine(String line) {
        if (diskWriter == null) {
            return null;
        }
        try {
            diskWriter.write(line);
            diskWriter.newLine();
            diskWriter.flush();
            return null;
        } catch (IOException e) {
            closeDiskLog();
            return e.getMessage();
        }
    }

    private void copyLogToClipboard() {
        String text = textPane.getText();
        File file = logFile;
        if (file != null && file.isFile()) {
            try {
                text = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
            } catch (IOException e) {
                text = textPane.getText();
            }
        }
        Toolkit.getDefaultToolkit().getSystemClipboard().setContents(new StringSelection(text), null);
    }

    private void clearVisibleLog() {
        boolean active = trainingActive;
        withScrollTrackingSuppressed(() -> {
            clearVisibleDocument();
            uiLineCount = 0;
            followTail = true;
            if (active) {
                appendActivityLine();
            } else {
                showPlaceholder();
            }
        });
        if (active) {
            scrollToBottom();
        }
    }

    private void clearVisibleDocument() {
        try {
            activityLineStart = -1;
            logDocument.remove(0, logDocument.getLength());
            placeholderVisible = false;
        } catch (BadLocationException e) {
            textPane.setText("");
            activityLineStart = -1;
            placeholderVisible = false;
        }
    }

    private void showPlaceholder() {
        clearVisibleDocument();
        try {
            logDocument.insertString(0, PLACEHOLDER, placeholderStyle);
            placeholderVisible = true;
        } catch (BadLocationException e) {
            textPane.setText(PLACEHOLDER);
            placeholderVisible = true;
        }
    }

    private void startActivityTimer() {
        if (!activityTimer.isRunning()) {
            activityTimer.start();
        }
        updateActivityLine();
    }

    private void stopActivityTimer() {
        if (activityTimer.isRunning()) {
            activityTimer.stop();
        }
        removeActivityLine();
    }

    private void updateActivityLine() {
        if (!trainingActive) {
            return;
        }
        boolean shouldFollow = followTail && isAtBottom();
        activityDots = activityDots % 3 + 1;
        withScrollTrackingSuppressed(() -> {
            removeActivityLine();
            appendActivityLine();
        });
        if (shouldFollow) {
            scrollToBottom();
        } else {
            followTail = false;
        }
    }

    private void appendActivityLine() {
        if (!trainingActive || activityLineStart >= 0) {
            return;
        }
        if (placeholderVisible) {
            clearVisibleDocument();
        }
        StringBuilder dots = new StringBuilder();
        int nDots = activityDots <= 0 ? 1 : activityDots;
        for (int i = 0; i < nDots; i++) {
            dots.append('.');
        }
        try {
            activityLineStart = logDocument.getLength();
            logDocument.insertString(logDocument.getLength(), "[" + LocalTime.now().format(TIME_FORMAT) + "] ",
                    timestampStyle);
            logDocument.insertString(logDocument.getLength(), ACTIVITY_TEXT + dots + System.lineSeparator(),
                    activityStyle);
        } catch (BadLocationException e) {
            activityLineStart = -1;
        }
    }

    private void removeActivityLine() {
        if (activityLineStart < 0) {
            return;
        }
        try {
            int length = logDocument.getLength() - activityLineStart;
            if (length > 0) {
                logDocument.remove(activityLineStart, length);
            }
        } catch (BadLocationException e) {
            // Ignore; the next append will reset the activity line.
        } finally {
            activityLineStart = -1;
        }
    }

    private boolean isAtBottom() {
        JScrollBar bar = scrollPane.getVerticalScrollBar();
        return bar.getValue() + bar.getVisibleAmount() >= bar.getMaximum() - 4;
    }

    private void withScrollTrackingSuppressed(Runnable runnable) {
        boolean previous = suppressScrollTracking;
        suppressScrollTracking = true;
        try {
            runnable.run();
        } finally {
            if (!previous) {
                SwingUtilities.invokeLater(() -> suppressScrollTracking = false);
            } else {
                suppressScrollTracking = true;
            }
        }
    }

    private void scrollToBottom() {
        adjustingScroll = true;
        SwingUtilities.invokeLater(() -> {
            JScrollBar bar = scrollPane.getVerticalScrollBar();
            bar.setValue(bar.getMaximum());
            followTail = true;
            adjustingScroll = false;
        });
    }

    private static void styleLogButton(JButton button) {
        button.setOpaque(true);
        button.setFocusPainted(false);
        button.setBorder(new LineBorder(new Color(163, 176, 199)));
        button.setBackground(new Color(242, 246, 252));
        button.setForeground(new Color(40, 67, 105));
        button.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 11));
    }

    private static final class NoWrapTextPane extends JTextPane {

        private static final long serialVersionUID = -7342380507971939122L;

        @Override
        public boolean getScrollableTracksViewportWidth() {
            return false;
        }
    }

    private static final class CopyIcon implements Icon {

        private static final int SIZE = 15;

        @Override
        public int getIconWidth() {
            return SIZE;
        }

        @Override
        public int getIconHeight() {
            return SIZE;
        }

        @Override
        public void paintIcon(Component c, Graphics g, int x, int y) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(new Color(120, 142, 174));
            g2.drawRoundRect(x + 2, y + 1, 8, 10, 2, 2);
            g2.setColor(new Color(40, 67, 105));
            g2.drawRoundRect(x + 5, y + 4, 8, 10, 2, 2);
            g2.dispose();
        }
    }

    private static final class EraserIcon implements Icon {

        private static final int SIZE = 15;

        @Override
        public int getIconWidth() {
            return SIZE;
        }

        @Override
        public int getIconHeight() {
            return SIZE;
        }

        @Override
        public void paintIcon(Component c, Graphics g, int x, int y) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.rotate(-0.55, x + SIZE / 2.0, y + SIZE / 2.0);
            g2.setColor(new Color(245, 171, 145));
            g2.fillRoundRect(x + 2, y + 5, 10, 6, 2, 2);
            g2.setColor(new Color(40, 67, 105));
            g2.drawRoundRect(x + 2, y + 5, 10, 6, 2, 2);
            g2.setColor(new Color(250, 250, 250));
            g2.fillRect(x + 8, y + 6, 3, 4);
            g2.dispose();
            Graphics2D line = (Graphics2D) g.create();
            line.setColor(new Color(120, 142, 174));
            line.drawLine(x + 3, y + 13, x + 12, y + 13);
            line.dispose();
        }
    }

    private static int deriveCurrentEpoch(int currentStep, int totalSteps, int totalEpochs) {
        if (currentStep <= 0 || totalSteps <= 0 || totalEpochs <= 0) {
            return 0;
        }
        int epoch = (int) Math.ceil((double) currentStep * totalEpochs / totalSteps);
        return Math.max(1, Math.min(totalEpochs, epoch));
    }

    private static String formatDuration(long millis) {
        long totalSeconds = Math.max(0L, millis / 1000L);
        long hours = totalSeconds / 3600L;
        long minutes = (totalSeconds % 3600L) / 60L;
        long seconds = totalSeconds % 60L;
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private final class StatusPanel extends JPanel {

        private static final long serialVersionUID = -5711641306826389429L;

        /**
         * Paints the status panel.
         *
         * @param g the graphics context.
         */
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
            drawCenteredXLabel(g2, "elapsed " + formatDuration(elapsedMillis), left, colW, singleRow);
            drawCenteredXLabel(g2, "iteration (it) " + currentStep + "/" + totalSteps, left + colW, colW, row1);
            drawCenteredXLabel(g2, "epoch (ep) " + currentEpoch + "/" + totalEpochs, left + colW, colW, row2);
            drawCenteredXLabel(g2, buildSecondsPerStepStatus(), left + 2 * colW, width - 2 * colW, row1);
            drawCenteredXLabel(g2, buildEpochRemainingStatus(), left + 2 * colW, width - 2 * colW, row2);
        }

        private void drawCenteredXLabel(Graphics2D g2, String label, int x, int width, int y) {
            g2.setColor(TEXT_COLOR);
            FontMetrics fm = g2.getFontMetrics();
            String visibleLabel = ellipsize(label, fm, Math.max(1, width - 4));
            int labelX = x + Math.max(0, (width - fm.stringWidth(visibleLabel)) / 2);
            g2.drawString(visibleLabel, labelX, y);
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
            return "epoch ETA " + formatDuration(Math.round(remainingSteps * secondsPerStep * 1000.0d));
        }

        private int remainingStepsInCurrentEpoch() {
            if (currentStep < 0 || totalSteps <= 0 || totalEpochs <= 0 || currentEpoch <= 0) {
                return -1;
            }
            int epochEndStep = (int) Math.ceil((double) currentEpoch * totalSteps / totalEpochs);
            return Math.max(0, epochEndStep - currentStep);
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
}
