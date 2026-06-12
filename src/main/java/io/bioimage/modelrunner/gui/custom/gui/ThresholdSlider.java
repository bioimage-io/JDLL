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
package io.bioimage.modelrunner.gui.custom.gui;

import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;

import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

public class ThresholdSlider extends JPanel {
    private static final long serialVersionUID = 5613675572139011295L;
	private final JSlider slider;
    private final JFormattedTextField valueField;

    /**
     * Creates a new ThresholdSlider.
     */
    public ThresholdSlider() {
        super(null);

        // 1) Create a slider from 0–100 (we'll map that to 0.0–1.0)
        slider = new JSlider(0, 1000, 500);
        slider.setPaintTicks(false);
        slider.setPaintLabels(false);
        slider.setToolTipText("Lower values allow more detections; higher values keep only stronger detections.");

        // 2) Create a formatted text field for doubles 0.00–1.00
        NumberFormat fmt = DecimalFormat.getNumberInstance();
        fmt.setMinimumFractionDigits(2);
        fmt.setMaximumFractionDigits(2);
        valueField = new JFormattedTextField(fmt);
        valueField.setColumns(4);
        valueField.setHorizontalAlignment(SwingConstants.CENTER);
        valueField.setToolTipText("Probability threshold");
        // initialize with the slider’s value
        valueField.setValue(slider.getValue() / 1000.0);

        // When slider moves, update the text field
        slider.addChangeListener(e -> {
            double val = slider.getValue() / 1000.0;
            valueField.setValue(val);
        });

        // When the user edits the text field, update the slider
        valueField.addActionListener(e -> syncFieldToSlider());
        valueField.addFocusListener(new FocusAdapter() {
            /**
             * Performs focus lost.
             *
             * @param e the e.
             */
            @Override public void focusLost(FocusEvent e) {
                syncFieldToSlider();
            }
        });

        // 4) Layout
        add(slider);
        add(valueField);

        organiseComponents();
    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            /**
             * Executes component resized.
             *
             * @param e the e parameter.
             */
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = Math.max(2, Math.min(4, rawH / 18));
                int textFieldInset = Math.max(1, Math.min(3, rawH / 12));
                double percSlider = 0.84;
                
                int availableW = Math.max(1, rawW - inset * 3);
                int sliderW = Math.max(1, (int) Math.round(percSlider * availableW));
                int valueW = Math.max(1, availableW - sliderW);
                int sliderH = Math.max(1, rawH - 2 * inset);
                int valueH = Math.max(1, rawH - 2 * textFieldInset);
                slider.setBounds(inset, inset, sliderW, sliderH);
                valueField.setBounds(inset * 2 + sliderW, textFieldInset, valueW, valueH);

                float valueFontSize = Math.max(9f, Math.min(17f, Math.min(valueH * 0.52f, valueW / 3.8f)));
                Font valueFont = fittedValueFont(valueFontSize, Math.max(1, valueW - 4), Math.max(1, valueH - 4));
                valueField.setFont(valueFont);
            }
        });
    }

    private Font fittedValueFont(float initialSize, int width, int height) {
        float size = initialSize;
        while (size > 6f) {
            Font font = valueField.getFont().deriveFont(size);
            FontMetrics fm = valueField.getFontMetrics(font);
            if (fm.stringWidth("0.00") <= width && fm.getHeight() <= height)
                return font;
            size -= 0.5f;
        }
        return valueField.getFont().deriveFont(6f);
    }
    
    /**
     * 
     * @return the slider
     */
    public JSlider getSlider() {
    	return this.slider;
    }

    /**
     * Returns the threshold value.
     *
     * @return the threshold value.
     */
    public double getThreshold() {
        return slider.getValue() / 1000.0d;
    }

    /**
     * Sets the threshold value.
     *
     * @param threshold the threshold.
     */
    public void setThreshold(double threshold) {
        double value = Math.max(0.0d, Math.min(1.0d, threshold));
        slider.setValue((int) Math.round(value * 1000.0d));
    }

    private void syncFieldToSlider() {
        try {
            valueField.commitEdit();
            Number n = (Number) valueField.getValue();
            double d = n.doubleValue();
            // clamp between 0 and 1
            d = Math.max(0.0, Math.min(1.0, d));
            slider.setValue((int) Math.round(d * 1000));
        } catch (ParseException ex) {
            // if parse fails, just reset the field to the slider's current value
            valueField.setValue(slider.getValue() / 1000.0);
        }
    }

    private static void createAndShowGui() {
        JFrame frame = new JFrame("Detection Threshold");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ThresholdSlider());
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    /**
     * Executes main.
     *
     * @param args the args parameter.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(ThresholdSlider::createAndShowGui);
    }
}
