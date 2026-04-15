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
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import javax.swing.JPanel;
import javax.swing.border.LineBorder;

public class YoloValidationPreviewPanel extends JPanel {

    private static final long serialVersionUID = -4978636324362880905L;

    public YoloValidationPreviewPanel() {
        setOpaque(true);
        setBackground(Color.WHITE);
        setBorder(new LineBorder(new Color(205, 210, 221)));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int pad = 16;
        int titleH = Math.max(22, getHeight() / 7);
        int contentY = pad + titleH;
        int contentH = Math.max(40, getHeight() - contentY - pad);
        int gap = Math.max(10, getWidth() / 40);
        int boxW = Math.max(40, (getWidth() - 2 * pad - gap) / 2);
        int boxH = contentH;
        int leftX = pad;
        int rightX = leftX + boxW + gap;

        g2.setColor(new Color(70, 78, 98));
        YoloUiUtils.drawCenteredString(g2, "Validation preview", 0, pad / 2, getWidth(), titleH, new Color(70, 78, 98));

        drawPreviewBox(g2, leftX, contentY, boxW, boxH, "Validation image", false);
        drawPreviewBox(g2, rightX, contentY, boxW, boxH, "Prediction", true);
        g2.dispose();
    }

    private static void drawPreviewBox(Graphics2D g2, int x, int y, int w, int h, String title, boolean drawBoxes) {
        g2.setColor(new Color(241, 244, 250));
        g2.fillRoundRect(x, y, w, h, 12, 12);
        g2.setColor(new Color(205, 210, 221));
        g2.drawRoundRect(x, y, w, h, 12, 12);

        int innerPad = Math.max(10, Math.min(w, h) / 12);
        int imgX = x + innerPad;
        int imgY = y + innerPad + 14;
        int imgW = Math.max(20, w - 2 * innerPad);
        int imgH = Math.max(20, h - 2 * innerPad - 14);

        g2.setColor(new Color(220, 226, 237));
        g2.fillRoundRect(imgX, imgY, imgW, imgH, 10, 10);
        g2.setColor(new Color(160, 170, 188));
        g2.drawRoundRect(imgX, imgY, imgW, imgH, 10, 10);
        g2.setColor(new Color(106, 115, 134));
        YoloUiUtils.drawCenteredString(g2, title, x, y + 4, w, 18, new Color(90, 98, 115));

        if (drawBoxes) {
            g2.setColor(new Color(232, 72, 72));
            int bx1 = imgX + imgW / 6;
            int by1 = imgY + imgH / 5;
            int bw1 = Math.max(14, imgW / 3);
            int bh1 = Math.max(14, imgH / 4);
            int bx2 = imgX + imgW / 2;
            int by2 = imgY + imgH / 2;
            int bw2 = Math.max(14, imgW / 4);
            int bh2 = Math.max(14, imgH / 5);
            g2.drawRect(bx1, by1, bw1, bh1);
            g2.drawRect(bx2, by2, bw2, bh2);
        }
    }
}
