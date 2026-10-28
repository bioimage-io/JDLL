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
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

import javax.swing.JEditorPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JScrollBar;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.border.LineBorder;
import javax.swing.border.TitledBorder;
import javax.swing.text.html.HTMLDocument;

import io.bioimage.modelrunner.model.python.envs.EnvironmentSetupLog;

public class YoloHtmlLogPanel extends JPanel {

    private static final long serialVersionUID = 2996134269810557663L;

    private static final int PAD = 2;
    private static final int MAX_LOG_LINES = 250;
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final String EMPTY_HTML = "<html><body style='font-family:sans-serif;font-size:11px;'>"
            + "<div id='log-lines'></div><div id='setup-progress'></div></body></html>";

    protected final JEditorPane editorPane = new JEditorPane() {
        private static final long serialVersionUID = 1L;
        @Override public boolean getScrollableTracksViewportWidth() { return false; }
    };
    protected final JScrollPane scrollPane = new JScrollPane(editorPane);
    private final TitledBorder elapsedBorder = new TitledBorder(new LineBorder(Color.GRAY), "0.0s",
            TitledBorder.LEFT, TitledBorder.TOP);
    private long nextLineId;
    private final Timer elapsedTimer;
    private long startNanos;
    private String installationLine;
    private String installationTimestamp;
    private boolean followTail = true;
    private boolean updating;
    private boolean scrollPending;
    private int previousValue, previousMaximum, previousExtent;
    private boolean previousAdjusting;
    private final InstallationLogController installationLog = new InstallationLogController(
            this::appendPlainLine, this::setInstallationLine);

    /** Bind environment reporting for this inference worker without changing model factory APIs. */
    public EnvironmentSetupLog.Scope openEnvironmentLog() {
        return installationLog.openScope();
    }

    /**
     * Creates a new YoloHtmlLogPanel instance.
     */
    public YoloHtmlLogPanel() {
        setLayout(null);
        setBorder(elapsedBorder);
        setOpaque(true);
        setBackground(YoloUiUtils.INPUT_BG);
        editorPane.setContentType("text/html");
        editorPane.setEditable(false);
        LogCaret caret = new LogCaret(() -> updating);
        caret.setBlinkRate(editorPane.getCaret().getBlinkRate());
        editorPane.setCaret(caret);
        editorPane.setOpaque(true);
        editorPane.setBackground(YoloUiUtils.INPUT_BG);
        scrollPane.setBackground(Color.WHITE);
        scrollPane.getViewport().setOpaque(true);
        scrollPane.getViewport().setBackground(Color.WHITE);
        editorPane.setText(EMPTY_HTML);
        elapsedTimer = new Timer(100, e -> updateElapsedTitle());
        scrollPane.getVerticalScrollBar().addAdjustmentListener(e -> {
            JScrollBar bar = scrollPane.getVerticalScrollBar();
            boolean rangeChanged = bar.getMaximum() != previousMaximum || bar.getVisibleAmount() != previousExtent;
            boolean moved = bar.getValue() != previousValue;
            boolean released = previousAdjusting && !bar.getValueIsAdjusting();
            boolean atBottom = bar.getValue() + bar.getVisibleAmount() >= bar.getMaximum() - 4;
            if (!updating && moved && (atBottom || LogScrollInput.isUserScroll(scrollPane))) {
                followTail = atBottom;
            }
            previousValue = bar.getValue();
            previousMaximum = bar.getMaximum();
            previousExtent = bar.getVisibleAmount();
            previousAdjusting = bar.getValueIsAdjusting();
            if (followTail && (rangeChanged || released || (!updating && moved))) scrollToBottom();
        });
        add(scrollPane);
        addComponentListener(new ComponentAdapter() {
            /**
             * Performs component resized.
             *
             * @param e the e.
             */
            @Override
            public void componentResized(ComponentEvent e) {
                doLayout();
            }
        });
    }

    /**
     * Performs do layout.
     */
    @Override
    public void doLayout() {
        java.awt.Insets insets = getInsets();
        int x = Math.max(PAD, insets.left);
        int y = Math.max(PAD, insets.top);
        int right = Math.max(PAD, insets.right);
        int bottom = Math.max(PAD, insets.bottom);
        scrollPane.setBounds(x, y, Math.max(0, getWidth() - x - right), Math.max(0, getHeight() - y - bottom));
        editorPane.setFont(editorPane.getFont().deriveFont((float) Math.max(YoloUiUtils.MIN_FONT_SIZE, (int) Math.floor(getHeight() * 0.08))));
    }

    /**
     * Performs start run timer.
     */
    public void startRunTimer() {
        clearLog();
        startNanos = System.nanoTime();
        updateElapsedTitle(0.0);
        elapsedTimer.restart();
    }

    /**
     * Performs stop run timer.
     */
    public void stopRunTimer() {
        installationLog.stop();
        if (elapsedTimer.isRunning()) {
            elapsedTimer.stop();
        }
        updateElapsedTitle();
    }

    /**
     * Performs clear log.
     */
    public void clearLog() {
        followTail = true;
        setHtml(EMPTY_HTML);
    }

    /**
     * Sets the HTML.
     *
     * @param html the HTML.
     */
    public void setHtml(String html) {
        nextLineId = 0;
        editorPane.setText(html);
        editorPane.setCaretPosition(0);
    }

    /**
     * Performs append HTML.
     *
     * @param htmlFragment the HTML fragment.
     */
    public void appendHtml(String htmlFragment) {
        if (!SwingUtilities.isEventDispatchThread()) {
            SwingUtilities.invokeLater(() -> appendHtml(htmlFragment));
            return;
        }
        updating = true;
        try {
            HTMLDocument document = logDocument();
            document.insertBeforeEnd(document.getElement("log-lines"),
                    "<pre id='log-" + nextLineId + "' style='margin:0;font-family:sans-serif;font-size:11px;'>"
                            + wrapLogLine(htmlFragment) + "</pre>");
            if (nextLineId >= MAX_LOG_LINES) {
                javax.swing.text.Element first = document.getElement("log-" + (nextLineId - MAX_LOG_LINES));
                int start = first.getStartOffset(), end = first.getEndOffset();
                java.awt.geom.Rectangle2D before = editorPane.modelToView2D(start);
                java.awt.geom.Rectangle2D after = editorPane.modelToView2D(end);
                int removedHeight = before == null || after == null ? 0 : (int) (after.getY() - before.getY());
                int oldScroll = scrollPane.getVerticalScrollBar().getValue();
                int dot = editorPane.getCaret().getDot(), mark = editorPane.getCaret().getMark();
                document.remove(start, end - start);
                if (!followTail) {
                    editorPane.setCaretPosition(Math.max(0, mark - (end - start)));
                    editorPane.moveCaretPosition(Math.max(0, dot - (end - start)));
                    scrollPane.getVerticalScrollBar().setValue(Math.max(0, oldScroll - removedHeight));
                }
            }
            nextLineId++;
        } catch (javax.swing.text.BadLocationException | java.io.IOException e) {
            // A UI logging failure must not interrupt inference.
        }
        finally { updating = false; }
        if (followTail) scrollToBottom();
    }

    private void appendPlainLine(String line) {
        appendHtml(escape(line));
    }

    private static String escape(String text) {
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    private void setInstallationLine(String line) {
        if (installationLine == null && line != null) installationTimestamp = LocalTime.now().format(TIME_FORMATTER);
        installationLine = line;
        updating = true;
        try {
            HTMLDocument document = logDocument();
            document.setInnerHTML(document.getElement("setup-progress"), installationHtml());
        } catch (javax.swing.text.BadLocationException | java.io.IOException e) {
            // The next progress event can refresh the live entry.
        } finally { updating = false; }
        if (followTail) scrollToBottom();
    }

    private String installationHtml() {
        // Swing's HTML parser does not reliably replace children with an empty fragment.
        return installationLine == null ? "<br>" : "<pre style='margin:0;'><span style='color:#666;'>[" + installationTimestamp
                + "]</span>&nbsp;<span style='font-family:monospace;color:#2c6fb3;'>"
                + escape(installationLine) + "</span></pre>";
    }

    private void scrollToBottom() {
        if (scrollPending) return;
        scrollPending = true;
        SwingUtilities.invokeLater(() -> {
            scrollPending = false;
            if (!followTail || scrollPane.getVerticalScrollBar().getValueIsAdjusting()) return;
            updating = true;
            try {
                int horizontalPosition = scrollPane.getHorizontalScrollBar().getValue();
                editorPane.setCaretPosition(editorPane.getDocument().getLength());
                scrollPane.getVerticalScrollBar().setValue(scrollPane.getVerticalScrollBar().getMaximum());
                scrollPane.getHorizontalScrollBar().setValue(horizontalPosition);
            }
            finally { updating = false; }
        });
    }

    private HTMLDocument logDocument() {
        HTMLDocument document = (HTMLDocument) editorPane.getDocument();
        if (document.getElement("log-lines") == null || document.getElement("setup-progress") == null) {
            editorPane.setText(EMPTY_HTML);
            nextLineId = 0;
            document = (HTMLDocument) editorPane.getDocument();
        }
        return document;
    }

    private static String wrapLogLine(String htmlFragment) {
        String message = htmlFragment == null ? "" : htmlFragment;
        return "<span style='color:#666;'>[" + LocalTime.now().format(TIME_FORMATTER) + "]</span>&nbsp;"
                + message;
    }

    private void updateElapsedTitle() {
        if (startNanos <= 0) {
            updateElapsedTitle(0.0);
            return;
        }
        updateElapsedTitle((System.nanoTime() - startNanos) / 1_000_000_000.0);
    }

    private void updateElapsedTitle(double seconds) {
        elapsedBorder.setTitle(String.format(java.util.Locale.US, "%.1fs", seconds));
        repaint();
    }

    /**
     * Returns the editor pane.
     *
     * @return the editor pane.
     */
    public JEditorPane getEditorPane() {
        return editorPane;
    }
}
