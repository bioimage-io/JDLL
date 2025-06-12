/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Dictionary;
import java.util.Hashtable;

import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingUtilities;

public class ThresholdSlider extends JPanel {
    private static final long serialVersionUID = 5613675572139011295L;
	private final JSlider slider;
    private final JFormattedTextField valueField;
    
    private JLabel leftLabel, centreLabel, rightLabel;

    public ThresholdSlider() {
        super(null);

        // 1) Create a slider from 0–100 (we'll map that to 0.0–1.0)
        slider = new JSlider(0, 1000, 500);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.setMajorTickSpacing(500);    // 0, 50, 100
        slider.setMinorTickSpacing(250);    // optional finer ticks

        // custom labels at 0 -> "0", 50 -> "0.5", 100 -> "1"
        Dictionary<Integer, JLabel> labels = new Hashtable<>();
        leftLabel = new JLabel("<html><center>0<br>(more detections)</center></html>");
        leftLabel = new JLabel("0 (more detections)");
        centreLabel = new JLabel("0.5");
        rightLabel = new JLabel("<html><center>1<br>(less detections)</center></html>");
        rightLabel = new JLabel("1 (less detections)");
        labels.put(0, leftLabel);
        labels.put(500, centreLabel);
        labels.put(1000, rightLabel);
        slider.setLabelTable(labels);

        // 2) Create a formatted text field for doubles 0.00–1.00
        NumberFormat fmt = DecimalFormat.getNumberInstance();
        fmt.setMinimumFractionDigits(2);
        fmt.setMaximumFractionDigits(2);
        valueField = new JFormattedTextField(fmt);
        valueField.setColumns(4);
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
            @Override public void focusLost(FocusEvent e) {
                syncFieldToSlider();
            }
        });

        // 4) Layout
        add(slider, BorderLayout.CENTER);
        add(valueField, BorderLayout.EAST);

        organiseComponents();
    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = 3;
                int textFieldInset = 5;
                double percSlider = 0.90;
                
                int x = inset;
                int sizeX = (int) Math.max((percSlider * (rawW - inset * 2 - inset)), 1);
                int sizeY = Math.max(1, rawH - 2 * inset);
                slider.setBounds(inset, inset, sizeX, sizeY);
                x += inset + sizeX;
                sizeX = (int) ( (1 - percSlider) * (rawW - inset * 2 - inset));
                sizeY = Math.max(1, rawH - 2 * textFieldInset);
                valueField.setBounds(x, textFieldInset, sizeX, sizeY);
                float vFontSize = (rawH - 2 * inset) / 6f;
                vFontSize = Math.min(Math.max(1, vFontSize), 18);
                float wFontSize = (float) (((1 - percSlider) * (rawW - inset * 2 - inset)) / 3.3f);
                wFontSize = Math.min(Math.max(1, wFontSize), 18);
                float fontSize = Math.min(vFontSize, wFontSize);
                Font font = slider.getFont().deriveFont(fontSize);
                while (true && fontSize > 0) {
                    int availableWidthLeft = leftLabel.getWidth()
                            - leftLabel.getInsets().left
                            - leftLabel.getInsets().right;
                    FontMetrics fm = leftLabel.getFontMetrics(font);
                    String text = leftLabel.getText();
                    int textWidth = fm.stringWidth(text);
                    if (availableWidthLeft > textWidth)
                    	break;
                    fontSize -= 0.5;
                    font = slider.getFont().deriveFont(fontSize);
                }
                leftLabel.setFont(font);
                centreLabel.setFont(font);
                rightLabel.setFont(font);
            }
        });
    }
    
    /**
     * 
     * @return the slider
     */
    public JSlider getSlider() {
    	return this.slider;
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

    public static void main(String[] args) {
        SwingUtilities.invokeLater(ThresholdSlider::createAndShowGui);
    }
}
