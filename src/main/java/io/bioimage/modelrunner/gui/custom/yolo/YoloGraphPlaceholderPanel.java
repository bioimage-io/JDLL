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

public class YoloGraphPlaceholderPanel extends JPanel {

    private static final long serialVersionUID = -5079977598122379171L;

    private final String title;

    public YoloGraphPlaceholderPanel(String title) {
        this.title = title;
        setOpaque(true);
        setBackground(Color.WHITE);
        setBorder(new LineBorder(new Color(205, 210, 221)));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setColor(new Color(235, 239, 247));
        int left = 16;
        int top = 16;
        int right = Math.max(left + 40, getWidth() - 16);
        int bottom = Math.max(top + 40, getHeight() - 24);
        g2.drawLine(left, bottom, right, bottom);
        g2.drawLine(left, top, left, bottom);
        g2.setColor(new Color(164, 201, 255));
        int[] xs = new int[] {left + 8, left + (right - left) / 3, left + (right - left) * 2 / 3, right - 8};
        int[] ys = new int[] {bottom - 12, bottom - 50, bottom - 34, top + 18};
        g2.drawPolyline(xs, ys, xs.length);
        g2.setColor(new Color(90, 98, 115));
        YoloUiUtils.drawCenteredString(g2, title, 0, 0, getWidth(), Math.max(20, getHeight() / 6), new Color(70, 78, 98));
        g2.dispose();
    }
}
