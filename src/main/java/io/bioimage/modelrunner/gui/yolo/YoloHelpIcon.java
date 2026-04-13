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
package io.bioimage.modelrunner.gui.yolo;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import javax.swing.JPanel;

public class YoloHelpIcon extends JPanel {

    private static final long serialVersionUID = 3831441879451326854L;

    private static final Color BG = new Color(230, 238, 250);
    private static final Color FG = new Color(58, 91, 160);

    protected YoloHelpIcon() {
        setOpaque(false);
        setToolTipText("Use Ctrl + mouse wheel to zoom the preview.");
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int size = Math.max(2, Math.min(getWidth(), getHeight()) - 2);
        int x = (getWidth() - size) / 2;
        int y = (getHeight() - size) / 2;
        g2.setColor(BG);
        g2.fillOval(x, y, size, size);
        g2.setColor(FG);
        g2.drawOval(x, y, size, size);
        YoloUiUtils.drawCenteredString(g2, "?", x, y, size, size, FG);
        g2.dispose();
    }
}
