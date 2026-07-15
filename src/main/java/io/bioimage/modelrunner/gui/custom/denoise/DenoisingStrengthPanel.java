/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.denoise;

import java.text.DecimalFormat;
import java.text.ParseException;
import java.util.function.DoubleConsumer;

import javax.swing.JFormattedTextField;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;

import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;

public final class DenoisingStrengthPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    private final JLabel label = new JLabel("Denoising strength");
    private final JSlider slider = new JSlider(0, 100, 100);
    private final JFormattedTextField value = new JFormattedTextField(new DecimalFormat("0.00"));
    private DoubleConsumer consumer;
    private boolean updating;

    public DenoisingStrengthPanel() {
        setLayout(null);
        setOpaque(false);
        slider.setOpaque(false);
        value.setValue(1.0d);
        value.setHorizontalAlignment(JFormattedTextField.CENTER);
        value.setColumns(4);
        YoloUiUtils.styleInput(value);
        slider.setToolTipText("0.00 shows the original; 1.00 shows the complete denoised result.");
        slider.addChangeListener(e -> {
            if (updating) return;
            setStrength(slider.getValue() / 100.0d, false);
        });
        value.addActionListener(e -> commitText());
        value.addPropertyChangeListener("value", e -> {
            if (!updating) commitText();
        });
        add(label); add(slider); add(value);
    }

    private void commitText() {
        try {
            value.commitEdit();
            Object parsed = value.getValue();
            if (parsed instanceof Number) setStrength(((Number) parsed).doubleValue(), true);
        } catch (ParseException ignored) {
            setStrength(getStrength(), true);
        }
    }

    public void setStrength(double strength) { setStrength(strength, true); }

    private void setStrength(double strength, boolean updateSlider) {
        double clamped = Math.max(0.0d, Math.min(1.0d, strength));
        updating = true;
        try {
            if (updateSlider) slider.setValue((int) Math.round(clamped * 100.0d));
            value.setValue(clamped);
        } finally {
            updating = false;
        }
        if (consumer != null) consumer.accept(clamped);
    }

    public double getStrength() { return slider.getValue() / 100.0d; }
    public void setStrengthConsumer(DoubleConsumer valueConsumer) { consumer = valueConsumer; }
    public void setInteractionEnabled(boolean enabled) {
        label.setEnabled(enabled);
        slider.setEnabled(enabled);
        value.setEnabled(enabled);
    }

    @Override public void doLayout() {
        int h = Math.max(1, getHeight());
        int labelW = Math.max(110, getWidth() / 4);
        int valueW = Math.max(48, Math.min(72, getWidth() / 7));
        label.setBounds(0, 0, labelW, h);
        value.setBounds(getWidth() - valueW, 0, valueW, h);
        slider.setBounds(labelW + 6, 0, Math.max(1, getWidth() - labelW - valueW - 12), h);
        YoloUiUtils.applyResponsiveText(label, labelW, h);
        YoloUiUtils.applyResponsiveFont(value, h);
    }
}
