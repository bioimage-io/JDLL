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

import javax.swing.JEditorPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.border.LineBorder;

public class YoloHtmlLogPanel extends JPanel {

    private static final long serialVersionUID = 2996134269810557663L;

    private static final int PAD = 2;

    protected final JEditorPane editorPane = new JEditorPane();
    protected final JScrollPane scrollPane = new JScrollPane(editorPane);

    protected YoloHtmlLogPanel() {
        setLayout(null);
        setBorder(new LineBorder(Color.GRAY));
        setOpaque(true);
        setBackground(YoloUiUtils.INPUT_BG);
        editorPane.setContentType("text/html");
        editorPane.setEditable(false);
        editorPane.setBackground(YoloUiUtils.INPUT_BG);
        editorPane.setText("<html><body style='font-family:sans-serif;font-size:11px;'></body></html>");
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
        scrollPane.setBounds(PAD, PAD, Math.max(0, getWidth() - 2 * PAD), Math.max(0, getHeight() - 2 * PAD));
        editorPane.setFont(editorPane.getFont().deriveFont((float) Math.max(YoloUiUtils.MIN_FONT_SIZE, (int) Math.floor(getHeight() * 0.08))));
    }

    public void setHtml(String html) {
        editorPane.setText(html);
        editorPane.setCaretPosition(0);
    }

    public void appendHtml(String htmlFragment) {
        String text = editorPane.getText();
        int idx = text.lastIndexOf("</body>");
        if (idx < 0) {
            setHtml(htmlFragment);
            return;
        }
        String updated = text.substring(0, idx) + htmlFragment + text.substring(idx);
        editorPane.setText(updated);
        editorPane.setCaretPosition(editorPane.getDocument().getLength());
    }

    public JEditorPane getEditorPane() {
        return editorPane;
    }
}
