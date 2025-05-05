package io.bioimage.modelrunner.gui.custom;

import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;

import javax.swing.JTextField;

public class PlaceholderTextField extends JTextField {
    private static final long serialVersionUID = 5112778641734509160L;
	private final String placeholder;
    private final Color placeholderColor;

    protected PlaceholderTextField(String placeholder) {
        this.placeholder = placeholder;
        this.placeholderColor = Color.LIGHT_GRAY;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        if (getText().isEmpty()) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setColor(placeholderColor);
            Insets ins = getInsets();
            FontMetrics fm = g2.getFontMetrics();
            int x = ins.left;
            // Vertically center the text:
            int y = (getHeight() + fm.getAscent() - fm.getDescent()) / 2;
            g2.drawString(placeholder, x, y);
            g2.dispose();
        }
    }
}
