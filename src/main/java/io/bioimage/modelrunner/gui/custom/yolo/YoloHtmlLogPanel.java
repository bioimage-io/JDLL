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
import javax.swing.Timer;
import javax.swing.border.LineBorder;
import javax.swing.border.TitledBorder;

public class YoloHtmlLogPanel extends JPanel {

    private static final long serialVersionUID = 2996134269810557663L;

    private static final int PAD = 2;
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final String EMPTY_HTML = "<html><body style='font-family:sans-serif;font-size:11px;'></body></html>";

    protected final JEditorPane editorPane = new JEditorPane();
    protected final JScrollPane scrollPane = new JScrollPane(editorPane);
    private final TitledBorder elapsedBorder = new TitledBorder(new LineBorder(Color.GRAY), "0.0s",
            TitledBorder.LEFT, TitledBorder.TOP);
    private final Timer elapsedTimer;
    private long startNanos;

    protected YoloHtmlLogPanel() {
        setLayout(null);
        setBorder(elapsedBorder);
        setOpaque(true);
        setBackground(YoloUiUtils.INPUT_BG);
        editorPane.setContentType("text/html");
        editorPane.setEditable(false);
        editorPane.setBackground(YoloUiUtils.INPUT_BG);
        editorPane.setText(EMPTY_HTML);
        elapsedTimer = new Timer(100, e -> updateElapsedTitle());
        add(scrollPane);
        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                doLayout();
            }
        });
    }

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

    public void startRunTimer() {
        clearLog();
        startNanos = System.nanoTime();
        updateElapsedTitle(0.0);
        elapsedTimer.restart();
    }

    public void stopRunTimer() {
        if (elapsedTimer.isRunning()) {
            elapsedTimer.stop();
        }
        updateElapsedTitle();
    }

    public void clearLog() {
        setHtml(EMPTY_HTML);
    }

    public void setHtml(String html) {
        editorPane.setText(html);
        editorPane.setCaretPosition(0);
    }

    public void appendHtml(String htmlFragment) {
        String text = editorPane.getText();
        int idx = text.lastIndexOf("</body>");
        if (idx < 0) {
            setHtml(wrapLogLine(htmlFragment));
            return;
        }
        String updated = text.substring(0, idx) + wrapLogLine(htmlFragment) + text.substring(idx);
        editorPane.setText(updated);
        editorPane.setCaretPosition(editorPane.getDocument().getLength());
    }

    private static String wrapLogLine(String htmlFragment) {
        String message = htmlFragment == null ? "" : htmlFragment;
        return "<span style='color:#666;'>[" + LocalTime.now().format(TIME_FORMATTER) + "]</span> "
                + message + "<br/>";
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

    public JEditorPane getEditorPane() {
        return editorPane;
    }
}
