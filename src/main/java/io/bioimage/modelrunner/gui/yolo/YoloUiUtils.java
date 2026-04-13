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
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;

import javax.swing.AbstractButton;
import javax.swing.BorderFactory;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.border.Border;

public final class YoloUiUtils {

    private static final String FULL_TEXT_KEY = "yolo.fullText";

    public static final int MIN_FONT_SIZE = 9;
    public static final Color PANEL_BG = new Color(248, 249, 252);
    public static final Color INPUT_BG = Color.WHITE;
    public static final Color INPUT_BORDER = new Color(205, 210, 221);
    public static final Color PRIMARY_BUTTON_BG = new Color(164, 201, 255);
    public static final Color PRIMARY_BUTTON_FG = Color.BLACK;
    public static final Color SECONDARY_BUTTON_BG = new Color(223, 228, 237);
    public static final Color SECONDARY_BUTTON_FG = Color.BLACK;
    public static final Border INPUT_LINE_BORDER = BorderFactory.createLineBorder(INPUT_BORDER);
    public static final Border BUTTON_BORDER = BorderFactory.createLineBorder(new Color(148, 155, 168));

    private YoloUiUtils() {}

    public static void styleFlatPrimaryButton(AbstractButton button) {
        styleFlatButton(button, PRIMARY_BUTTON_BG, PRIMARY_BUTTON_FG);
    }

    public static void styleFlatSecondaryButton(AbstractButton button) {
        styleFlatButton(button, SECONDARY_BUTTON_BG, SECONDARY_BUTTON_FG);
    }

    public static void styleFlatButton(AbstractButton button, Color bg, Color fg) {
        button.setOpaque(true);
        button.setContentAreaFilled(true);
        button.setBackground(bg);
        button.setForeground(fg);
        button.setFocusPainted(false);
        button.setBorder(BUTTON_BORDER);
        button.setBorderPainted(true);
    }

    public static void styleInput(JComponent component) {
        component.setOpaque(true);
        component.setBackground(INPUT_BG);
        component.setBorder(INPUT_LINE_BORDER);
    }

    public static void ensureFullText(AbstractButton button) {
        if (button.getClientProperty(FULL_TEXT_KEY) == null) {
            button.putClientProperty(FULL_TEXT_KEY, button.getText());
        }
    }

    public static void ensureFullText(JLabel label) {
        if (label.getClientProperty(FULL_TEXT_KEY) == null) {
            label.putClientProperty(FULL_TEXT_KEY, label.getText());
        }
    }

    public static void applyResponsiveText(AbstractButton button, int width, int height) {
        ensureFullText(button);
        String fullText = (String) button.getClientProperty(FULL_TEXT_KEY);
        Font fitted = buildFont(button.getFont(), height);
        button.setFont(fitted);
        String text = fitted.getSize() < MIN_FONT_SIZE ? ellipsize(button, fullText, width) : ellipsize(button, fullText, width);
        button.setText(text);
    }

    public static void applyResponsiveText(JLabel label, int width, int height) {
        ensureFullText(label);
        String fullText = (String) label.getClientProperty(FULL_TEXT_KEY);
        Font fitted = buildFont(label.getFont(), height);
        label.setFont(fitted);
        label.setText(ellipsize(label, fullText, width));
    }

    public static void applyResponsiveFont(JTextField field, int height) {
        field.setFont(buildFont(field.getFont(), height));
    }

    public static void applyResponsiveFont(JComboBox<?> comboBox, int height) {
        comboBox.setFont(buildFont(comboBox.getFont(), height));
    }

    public static void applyResponsiveFont(JTabbedPane tabs, int height) {
        tabs.setFont(buildFont(tabs.getFont(), height));
    }

    public static void applyResponsiveFont(AbstractButton button, int height) {
        button.setFont(buildFont(button.getFont(), height));
    }

    private static Font buildFont(Font baseFont, int height) {
        int size = Math.max(MIN_FONT_SIZE, (int) Math.floor(height * 0.56));
        return baseFont.deriveFont((float) size);
    }

    private static String ellipsize(JComponent component, String fullText, int width) {
        if (fullText == null) {
            return "";
        }
        FontMetrics fm = component.getFontMetrics(component.getFont());
        if (fm.stringWidth(fullText) <= width) {
            return fullText;
        }
        String ellipsis = "...";
        if (fm.stringWidth(ellipsis) > width) {
            return ellipsis;
        }
        String text = fullText;
        while (text.length() > 0 && fm.stringWidth(text + ellipsis) > width) {
            text = text.substring(0, text.length() - 1);
        }
        return text + ellipsis;
    }

    public static void drawCenteredString(Graphics2D g2, String text, int x, int y, int width, int height, Color color) {
        int fontSize = Math.max(MIN_FONT_SIZE, (int) Math.floor(height * 0.58));
        Font font = g2.getFont().deriveFont(Font.BOLD, (float) fontSize);
        g2.setFont(font);
        FontMetrics fm = g2.getFontMetrics();
        int tx = x + (width - fm.stringWidth(text)) / 2;
        int ty = y + ((height - fm.getHeight()) / 2) + fm.getAscent();
        g2.setColor(color);
        g2.drawString(text, tx, ty);
    }

    public static void alignLabel(JLabel label) {
        label.setHorizontalAlignment(SwingConstants.LEFT);
    }
}
