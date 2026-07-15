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

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloAccelerationCheckBox;
import io.bioimage.modelrunner.gui.custom.yolo.YoloHtmlLogPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloTitlePanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;

public class DenoisingGUI extends JPanel {
    private static final long serialVersionUID = 1L;
    private static final int TAB_PAD = 4;
    private static final int OUTER_PAD = 8;
    private static final int TITLE_GAP = 6;
    private static final int ROW_GAP = 8;
    private static final double TITLE_HEIGHT_RATIO = 0.065d;
    private static final double LOG_PREVIEW_DONATION_RATIO = 0.05d;
    private static final double LOG_WIDTH_RATIO = 0.95d;
    private static final double PROGRESS_HEIGHT_RATIO = 0.8d;
    protected final YoloTitlePanel titlePanel;
    protected final YoloImageSourcePanel sourcePanel = new YoloImageSourcePanel();
    protected final YoloAccelerationCheckBox acceleration = new YoloAccelerationCheckBox();
    protected final DenoisingOptionsPanel optionsPanel = new DenoisingOptionsPanel(acceleration);
    protected final DenoisingComparisonPanel comparisonPanel = new DenoisingComparisonPanel();
    protected final DenoisingStrengthPanel strengthPanel = new DenoisingStrengthPanel();
    protected final DenoisingStatusPanel statusPanel = new DenoisingStatusPanel();
    protected final YoloHtmlLogPanel logPanel = new YoloHtmlLogPanel();
    protected final DenoisingActionPanel actionPanel = new DenoisingActionPanel();

    protected DenoisingGUI(GuiAdapter adapter) {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        titlePanel = new YoloTitlePanel("Denoising", adapter);
        add(titlePanel); add(optionsPanel.getMethodRow()); add(sourcePanel);
        add(optionsPanel.getEffortRow()); add(optionsPanel.getNoiseRow());
        add(comparisonPanel); add(strengthPanel); add(statusPanel); add(logPanel); add(actionPanel);
    }

    @Override public void doLayout() {
        int width = Math.max(1, getWidth() - 2 * OUTER_PAD);
        int titleH = Math.max(40, (int) Math.round(getHeight() * TITLE_HEIGHT_RATIO));
        titlePanel.setBounds(TAB_PAD, TAB_PAD,
                Math.max(1, getWidth() - 2 * TAB_PAD), titleH);

        int removedAccelerationH = Math.max(22,
                YoloUiUtils.controlHeightForFontSize(YoloUiUtils.MAX_CONTROL_FONT_SIZE) + 4);
        int gap = Math.max(2, Math.min(ROW_GAP, getHeight() / 70));
        int y = TAB_PAD + titleH + TITLE_GAP;
        int availableH = Math.max(1, getHeight() - y - OUTER_PAD);
        int maxRowH = Math.max(1,
                YoloUiUtils.controlHeightForFontSize(YoloUiUtils.MAX_CONTROL_FONT_SIZE));
        int minimumPreviewH = Math.max(60, (int) Math.round(getHeight() * 0.32d));
        int rowH = Math.max(1, Math.min(maxRowH,
                (availableH - minimumPreviewH - 8 * gap - removedAccelerationH) / 8));
        int sourceH = rowH * 2;
        int progressH = Math.max(1, (int) Math.round(rowH * PROGRESS_HEIGHT_RATIO));
        int logH = removedAccelerationH + (rowH - progressH)
                + Math.max(1, (int) Math.round(getHeight() * LOG_PREVIEW_DONATION_RATIO));
        int fixedH = rowH * 7 + progressH + logH + 8 * gap;
        int comparisonH = Math.max(1, availableH - fixedH);

        optionsPanel.getMethodRow().setBounds(OUTER_PAD, y, width, rowH);
        y += rowH + gap;
        sourcePanel.setBounds(OUTER_PAD, y, width, sourceH);
        y += sourceH + gap;
        optionsPanel.getEffortRow().setBounds(OUTER_PAD, y, width, rowH);
        y += rowH + gap;
        optionsPanel.getNoiseRow().setBounds(OUTER_PAD, y, width, rowH);
        y += rowH + gap;
        comparisonPanel.setBounds(OUTER_PAD, y, width, comparisonH);
        y += comparisonH + gap;
        strengthPanel.setBounds(OUTER_PAD, y, width, rowH);
        y += rowH + gap;
        int logW = Math.max(1, (int) Math.round(width * LOG_WIDTH_RATIO));
        int logX = OUTER_PAD + (width - logW) / 2;
        logPanel.setBounds(logX, y, logW, logH);
        y += logH + gap;
        statusPanel.setBounds(OUTER_PAD, y, width, progressH);
        y += progressH + gap;
        actionPanel.setBounds(OUTER_PAD, y, width, rowH);
    }

    public YoloImageSourcePanel getImageSourcePanel() { return sourcePanel; }
    public DenoisingOptionsPanel getOptionsPanel() { return optionsPanel; }
    public YoloAccelerationCheckBox getAccelerationCheckBox() { return acceleration; }
    public DenoisingComparisonPanel getComparisonPanel() { return comparisonPanel; }
    public DenoisingStrengthPanel getStrengthPanel() { return strengthPanel; }
    public DenoisingStatusPanel getStatusPanel() { return statusPanel; }
    public YoloHtmlLogPanel getLogPanel() { return logPanel; }
    public DenoisingActionPanel getActionPanel() { return actionPanel; }
}
