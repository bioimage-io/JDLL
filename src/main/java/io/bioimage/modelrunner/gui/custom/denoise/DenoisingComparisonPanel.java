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

import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

import io.bioimage.modelrunner.gui.custom.yolo.YoloImageDisplayPanel;

public final class DenoisingComparisonPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    private static final int GAP = 8;
    private final YoloImageDisplayPanel original = new YoloImageDisplayPanel();
    private final YoloImageDisplayPanel denoised = new YoloImageDisplayPanel();
    private final JLabel originalInfo = new JLabel("No source selected");
    private final JLabel denoisedInfo = new JLabel("No denoised preview");

    public DenoisingComparisonPanel() {
        setLayout(null);
        setOpaque(false);
        original.setDrawEnabled(false);
        denoised.setDrawEnabled(false);
        denoised.setHintOnHoverOnly(true);
        original.setEmptyMessage("Original preview will appear here");
        denoised.setEmptyMessage("Generate a preview to compare denoising");
        original.setViewportConsumer(denoised::setViewport);
        denoised.setViewportConsumer(original::setViewport);
        originalInfo.setForeground(new java.awt.Color(85, 90, 100));
        denoisedInfo.setForeground(new java.awt.Color(85, 90, 100));
        originalInfo.setHorizontalAlignment(SwingConstants.CENTER);
        denoisedInfo.setHorizontalAlignment(SwingConstants.CENTER);
        add(original); add(denoised); add(originalInfo); add(denoisedInfo);
    }

    @Override public void doLayout() {
        int left = Math.max(1, (getWidth() - GAP) / 2);
        int infoH = Math.max(18, Math.min(25, getHeight() / 12));
        int imageH = Math.max(1, getHeight() - infoH);
        original.setBounds(0, 0, left, imageH);
        denoised.setBounds(left + GAP, 0, Math.max(1, getWidth() - left - GAP), imageH);
        originalInfo.setBounds(2, imageH, Math.max(1, left - 4), infoH);
        denoisedInfo.setBounds(left + GAP + 2, imageH,
                Math.max(1, getWidth() - left - GAP - 4), infoH);
        io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils.applyResponsiveText(
                originalInfo, originalInfo.getWidth(), infoH);
        io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils.applyResponsiveText(
                denoisedInfo, denoisedInfo.getWidth(), infoH);
    }

    public YoloImageDisplayPanel getOriginalPanel() { return original; }
    public YoloImageDisplayPanel getDenoisedPanel() { return denoised; }
    public void setOriginalInfo(String value) {
        originalInfo.putClientProperty("yolo.fullText", null);
        originalInfo.setText(value == null ? "" : value);
        revalidate(); repaint();
    }
    public void setDenoisedInfo(String value) {
        denoisedInfo.putClientProperty("yolo.fullText", null);
        denoisedInfo.setText(value == null ? "" : value);
        revalidate(); repaint();
    }
}
