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
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.RenderingHints;

import javax.swing.JPanel;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;

public class YoloTitlePanel extends JPanel {

    private static final long serialVersionUID = -7112374297834023910L;

    private final String softwareName;
    private final Color softwareColor;

    private static final String RIGHT_TEXT = "YOLO";

    private static final Color RIGHT_COLOR = Color.BLACK;
    private static final Color BG_COLOR = Color.LIGHT_GRAY;

    private static final float MIN_FONT = 9f;
    private static final float MAX_FONT = 220f;
    private static final double GAP_RATIO = 0.035;
    private static final double HORIZONTAL_PADDING_RATIO = 0.04;
    private static final double VERTICAL_PADDING_RATIO = 0.14;

    protected YoloTitlePanel(GuiAdapter adapter) {
    	this.softwareName = adapter.getSoftwareName();
    	softwareColor = adapter.getTitleColor();
        setOpaque(true);
        setBackground(BG_COLOR);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        Insets in = getInsets();
        int availableW = Math.max(1, getWidth() - in.left - in.right);
        int availableH = Math.max(1, getHeight() - in.top - in.bottom);
        int padX = Math.max(2, (int) Math.round(availableW * HORIZONTAL_PADDING_RATIO));
        int padY = Math.max(1, (int) Math.round(availableH * VERTICAL_PADDING_RATIO));
        int drawW = Math.max(1, availableW - 2 * padX);
        int drawH = Math.max(1, availableH - 2 * padY);

        Font fittedFont = fitFont(g2, drawW, drawH);
        g2.setFont(fittedFont);
        FontMetrics fm = g2.getFontMetrics();

        int gap = Math.max(2, (int) Math.round(drawW * GAP_RATIO));
        int leftW = fm.stringWidth(softwareName);
        int rightW = fm.stringWidth(RIGHT_TEXT);
        int totalW = leftW + gap + rightW;

        int startX = in.left + (availableW - totalW) / 2;
        int baseline = in.top + (availableH - fm.getHeight()) / 2 + fm.getAscent();

        g2.setColor(softwareColor);
        g2.drawString(softwareName, startX, baseline);
        g2.setColor(RIGHT_COLOR);
        g2.drawString(RIGHT_TEXT, startX + leftW + gap, baseline);
        g2.dispose();
    }

    private Font fitFont(Graphics2D g2, int targetW, int targetH) {
        Font base = getFont();
        if (base == null) {
            base = new Font("SansSerif", Font.BOLD, 12);
        }

        float lo = MIN_FONT;
        float hi = MAX_FONT;
        float best = MIN_FONT;

        while (hi - lo > 0.5f) {
            float mid = (lo + hi) / 2f;
            Font testFont = base.deriveFont(Font.BOLD, mid);
            if (fits(g2, testFont, targetW, targetH)) {
                best = mid;
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return base.deriveFont(Font.BOLD, best);
    }

    private boolean fits(Graphics2D g2, Font font, int targetW, int targetH) {
        FontMetrics fm = g2.getFontMetrics(font);
        int gap = Math.max(2, (int) Math.round(targetW * GAP_RATIO));
        int totalW = fm.stringWidth(softwareName) + gap + fm.stringWidth(RIGHT_TEXT);
        return totalW <= targetW && fm.getHeight() <= targetH;
    }
}
