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

import java.awt.event.ActionListener;

import javax.swing.ButtonGroup;
import javax.swing.JComponent;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.SwingConstants;

import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;

/** Owns the three denoising option rows and their selected values. */
public final class DenoisingOptionsPanel {

    private final JLabel methodLabel = new JLabel("Method");
    private final JComboBox<DenoisingMethod> method =
            new JComboBox<DenoisingMethod>(DenoisingMethod.values());
    private final JPanel methodRow = new LabelComboRow(methodLabel, method, 0.18d);

    private final JLabel effortLabel = new JLabel("Effort");
    private final JRadioButton quick = new JRadioButton(DenoisingEffort.QUICK.toString());
    private final JRadioButton balanced = new JRadioButton(DenoisingEffort.BALANCED.toString());
    private final JRadioButton thorough = new JRadioButton(DenoisingEffort.THOROUGH.toString());
    private final JPanel effortRow = new EffortRow();

    private final JLabel noiseLabel = new JLabel("Noise structure");
    private final JComboBox<DenoisingNoiseStructure> noise =
            new JComboBox<DenoisingNoiseStructure>(DenoisingNoiseStructure.values());
    private final JPanel noiseRow;

    public DenoisingOptionsPanel(JComponent acceleration) {
        noiseRow = new NoiseRow(acceleration);
        method.setSelectedItem(DenoisingMethod.FAST);
        balanced.setSelected(true);
        noise.setSelectedItem(DenoisingNoiseStructure.AUTO);
        ButtonGroup effortGroup = new ButtonGroup();
        effortGroup.add(quick);
        effortGroup.add(balanced);
        effortGroup.add(thorough);
        method.addActionListener(e -> updateNoiseState());
        updateNoiseState();
    }

    public JPanel getMethodRow() { return methodRow; }
    public JPanel getEffortRow() { return effortRow; }
    public JPanel getNoiseRow() { return noiseRow; }
    public DenoisingMethod getMethod() { return (DenoisingMethod) method.getSelectedItem(); }
    public DenoisingNoiseStructure getNoiseStructure() {
        return (DenoisingNoiseStructure) noise.getSelectedItem();
    }
    public JComboBox<DenoisingMethod> getMethodComboBox() { return method; }
    public JComboBox<DenoisingNoiseStructure> getNoiseStructureComboBox() { return noise; }

    public DenoisingEffort getEffort() {
        if (quick.isSelected()) return DenoisingEffort.QUICK;
        if (thorough.isSelected()) return DenoisingEffort.THOROUGH;
        return DenoisingEffort.BALANCED;
    }

    public void addEffortActionListener(ActionListener listener) {
        quick.addActionListener(listener);
        balanced.addActionListener(listener);
        thorough.addActionListener(listener);
    }

    public void updateNoiseState() {
        boolean enabled = getMethod() == DenoisingMethod.CORRELATED;
        noiseLabel.setEnabled(enabled);
        noise.setEnabled(enabled);
    }

    public void setInteractionEnabled(boolean enabled) {
        methodLabel.setEnabled(enabled);
        method.setEnabled(enabled);
        effortLabel.setEnabled(enabled);
        quick.setEnabled(enabled);
        balanced.setEnabled(enabled);
        thorough.setEnabled(enabled);
        boolean noiseEnabled = enabled && getMethod() == DenoisingMethod.CORRELATED;
        noiseLabel.setEnabled(noiseEnabled);
        noise.setEnabled(noiseEnabled);
    }

    private static final class LabelComboRow extends JPanel {
        private static final long serialVersionUID = 1L;
        private static final int GAP = 6;
        private final JLabel label;
        private final JComboBox<?> combo;
        private final double labelRatio;

        private LabelComboRow(JLabel label, JComboBox<?> combo, double labelRatio) {
            this.label = label;
            this.combo = combo;
            this.labelRatio = labelRatio;
            setLayout(null);
            setOpaque(false);
            YoloUiUtils.alignLabel(label);
            YoloUiUtils.styleInput(combo);
            add(label);
            add(combo);
        }

        @Override
        public void doLayout() {
            int w = Math.max(0, getWidth());
            int h = Math.max(0, getHeight());
            int labelW = Math.max(1, (int) Math.round(w * labelRatio));
            label.setBounds(0, 0, labelW, h);
            combo.setBounds(labelW + GAP, 0, Math.max(1, w - labelW - GAP), h);
            YoloUiUtils.applyResponsiveText(label, labelW - 4, h);
            YoloUiUtils.applyResponsiveFont(combo, h);
        }
    }

    private final class EffortRow extends JPanel {
        private static final long serialVersionUID = 1L;
        private static final int OPTION_PAD = 8;

        private EffortRow() {
            setLayout(null);
            setOpaque(false);
            YoloUiUtils.alignLabel(effortLabel);
            quick.setOpaque(false);
            balanced.setOpaque(false);
            thorough.setOpaque(false);
            add(effortLabel);
            add(quick);
            add(balanced);
            add(thorough);
        }

        @Override
        public void doLayout() {
            int w = Math.max(0, getWidth());
            int h = Math.max(0, getHeight());
            int quarter = Math.max(1, w / 4);
            effortLabel.setBounds(0, 0, quarter, h);
            JRadioButton[] options = {quick, balanced, thorough};
            for (int i = 0; i < options.length; i++) {
                int start = quarter * (i + 1);
                int end = i == options.length - 1 ? w : quarter * (i + 2);
                options[i].setBounds(start + OPTION_PAD, 0,
                        Math.max(1, end - start - 2 * OPTION_PAD), h);
                YoloUiUtils.applyResponsiveText(options[i],
                        options[i].getWidth() - 4, h);
            }
            YoloUiUtils.applyResponsiveText(effortLabel, quarter - 4, h);
        }
    }

    private final class NoiseRow extends JPanel {
        private static final long serialVersionUID = 1L;
        private static final int COMBO_GAP = 6;
        private static final int ACCELERATION_GAP = 18;
        private static final double LABEL_RATIO = 0.25d;
        private final JComponent acceleration;

        private NoiseRow(JComponent acceleration) {
            this.acceleration = acceleration;
            setLayout(null);
            setOpaque(false);
            YoloUiUtils.alignLabel(noiseLabel);
            YoloUiUtils.styleInput(noise);
            acceleration.setOpaque(false);
            if (acceleration instanceof javax.swing.AbstractButton) {
                ((javax.swing.AbstractButton) acceleration).setHorizontalAlignment(SwingConstants.RIGHT);
            }
            add(noiseLabel);
            add(noise);
            add(acceleration);
        }

        @Override
        public void doLayout() {
            int w = Math.max(0, getWidth());
            int h = Math.max(0, getHeight());
            int labelW = Math.max(1, (int) Math.round(w * LABEL_RATIO));
            int preferredAccelerationW = acceleration.getPreferredSize().width + 32;
            int accelerationW = Math.min(Math.max(116, preferredAccelerationW), Math.max(1, w / 2));
            int comboX = labelW + COMBO_GAP;
            int accelerationX = Math.max(comboX + 1, w - accelerationW);
            noiseLabel.setBounds(0, 0, labelW, h);
            noise.setBounds(comboX, 0,
                    Math.max(1, accelerationX - comboX - ACCELERATION_GAP), h);
            acceleration.setBounds(accelerationX, 0, Math.max(1, w - accelerationX), h);
            YoloUiUtils.applyResponsiveText(noiseLabel, labelW - 4, h);
            YoloUiUtils.applyResponsiveFont(noise, h);
            if (acceleration instanceof javax.swing.AbstractButton) {
                YoloUiUtils.applyResponsiveText((javax.swing.AbstractButton) acceleration,
                        acceleration.getWidth(), Math.max(1, (int) Math.round(h * 0.8d)));
            }
        }
    }
}
