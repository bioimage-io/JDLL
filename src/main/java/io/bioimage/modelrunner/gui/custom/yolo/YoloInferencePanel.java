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

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class YoloInferencePanel extends JPanel {

    private static final long serialVersionUID = 8213303922324227868L;

    protected static final int OUTER_PAD = 8;
    protected static final int ROW_GAP = 8;
    protected static final double BOTTOM_GAP_EXTRA_RATIO = 0.5;
    protected static final double DISPLAY_WIDTH_RATIO = 0.95;
    protected static final double DISPLAY_MIN_HEIGHT_RATIO = 0.4;
    protected static final double DISPLAY_MAX_HEIGHT_RATIO = 0.5;
    protected static final double LOG_WIDTH_RATIO = 0.95;
    protected static final double DISPLAY_BASE_HEIGHT_RATIO = 0.45;
    protected static final double HELP_WIDTH_RATIO = 0.045;
    protected static final double DRAW_LABEL_RATIO = 0.56;
    protected static final double DRAW_BUTTON_RATIO = 0.14;
    protected static final double ROW_UNIT_MODEL = 1.3;
    protected static final double ROW_UNIT_SOURCE = 2.0;
    protected static final double ROW_UNIT_DRAW = 1.2;
    protected static final double ROW_UNIT_ACTION = 1.0;
    protected static final double ROW_UNIT_WARNING = 1.3;
    protected static final double ROW_UNIT_LOG = 1.6;

    protected final YoloModelSelectionPanel modelSelectionPanel = new YoloModelSelectionPanel();
    protected final YoloImageSourcePanel imageSourcePanel = new YoloImageSourcePanel();
    protected final YoloImageDisplayPanel imageDisplayPanel = new YoloImageDisplayPanel();
    protected final JLabel drawLabel = new JLabel("Draw reference bounding box");
    protected final JButton drawButton = new JButton("Draw");
    protected final JButton refreshButton = new JButton("\u27f3");
    protected final YoloHelpIcon helpLabel = new YoloHelpIcon();
    protected final YoloHtmlLogPanel logPanel = new YoloHtmlLogPanel();
    protected final YoloActionPanel actionPanel = new YoloActionPanel();
    protected final JLabel warningLabel = new JLabel(
            "<html><div style='text-align:center;'>&#9888; YOLO is optional third-party software, installed separately, and governed by its own license terms. See documentation for details.</div></html>");

    protected YoloInferencePanel() {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        YoloUiUtils.alignLabel(drawLabel);
        YoloUiUtils.styleFlatSecondaryButton(drawButton);
        YoloUiUtils.styleFlatSecondaryButton(refreshButton);
        add(modelSelectionPanel);
        add(imageSourcePanel);
        add(imageDisplayPanel);
        add(drawLabel);
        add(drawButton);
        add(refreshButton);
        add(helpLabel);
        add(logPanel);
        add(actionPanel);
        warningLabel.setForeground(new java.awt.Color(170, 35, 35));
        add(warningLabel);
    }

    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int rowGap = Math.max(2, Math.min(ROW_GAP, h / 70));
        int innerW = Math.max(0, w - 2 * OUTER_PAD);
        int x = OUTER_PAD;
        int y = OUTER_PAD;

        int extraBottomGap = Math.max(0, (int) Math.round(rowGap * BOTTOM_GAP_EXTRA_RATIO));
        int totalAvailH = Math.max(8, h - 2 * OUTER_PAD - extraBottomGap - 6 * rowGap);
        int previewBaseH = Math.max(1, (int) Math.round(h * DISPLAY_BASE_HEIGHT_RATIO));
        int controlsAvailH = Math.max(4, totalAvailH - previewBaseH);
        double totalUnits = ROW_UNIT_MODEL + ROW_UNIT_SOURCE + ROW_UNIT_DRAW + ROW_UNIT_LOG + ROW_UNIT_ACTION + ROW_UNIT_WARNING;
        int rowUnitPx = Math.max(1, (int) Math.floor(controlsAvailH / totalUnits));

        int maxControlH = Math.max(1, YoloUiUtils.controlHeightForFontSize(YoloUiUtils.MAX_CONTROL_FONT_SIZE));
        int maxSourceH = maxControlH * 2;
        int maxLogH = maxControlH * 3;
        int maxWarningH = maxControlH * 2;

        int modelH = Math.max(1, Math.min(maxControlH, (int) Math.round(rowUnitPx * ROW_UNIT_MODEL)));
        int sourceH = Math.max(1, Math.min(maxSourceH, modelH * 2));
        int drawH = Math.max(1, Math.min(maxControlH, (int) Math.round(rowUnitPx * ROW_UNIT_DRAW)));
        int logH = Math.max(1, Math.min(maxLogH, (int) Math.round(rowUnitPx * ROW_UNIT_LOG)));
        int actionH = Math.max(1, Math.min(maxControlH, (int) Math.round(rowUnitPx * ROW_UNIT_ACTION)));
        int warningH = Math.max(18, Math.min(maxWarningH, (int) Math.round(rowUnitPx * ROW_UNIT_WARNING)));

        int previewH = Math.max(1, totalAvailH - modelH - sourceH - drawH - logH - actionH - warningH);
        int displayMinH = Math.max(1, (int) Math.round(h * DISPLAY_MIN_HEIGHT_RATIO));
        int displayMaxH = Math.max(displayMinH, (int) Math.round(h * DISPLAY_MAX_HEIGHT_RATIO));
        previewH = Math.max(displayMinH, previewH);
        previewH = Math.min(displayMaxH, previewH);

        modelSelectionPanel.setBounds(x, y, innerW, modelH);
        y += modelH + rowGap;

        imageSourcePanel.setBounds(x, y, innerW, sourceH);
        y += sourceH + rowGap;

        int logW = Math.max(1, (int) Math.round(innerW * LOG_WIDTH_RATIO));
        int logX = x + (innerW - logW) / 2;
        imageDisplayPanel.setBounds(logX, y, logW, previewH);
        y += previewH + rowGap;

        int drawLabelW = (int) Math.round(innerW * DRAW_LABEL_RATIO);
        int drawBtnW = (int) Math.round(innerW * DRAW_BUTTON_RATIO);
        int helpW = (int) Math.round(innerW * HELP_WIDTH_RATIO);
        int refreshW = Math.max(1, drawBtnW / 2);
        int totalDrawRowW = drawLabelW + drawBtnW + refreshW + helpW + 3 * rowGap;
        int rowX = x + Math.max(0, (innerW - totalDrawRowW) / 2);
        drawLabel.setBounds(rowX, y, drawLabelW, drawH);
        rowX += drawLabelW + rowGap;
        drawButton.setBounds(rowX, y, drawBtnW, drawH);
        rowX += drawBtnW + rowGap;
        refreshButton.setBounds(rowX, y, refreshW, drawH);
        rowX += refreshW + rowGap;
        int helpSize = Math.max(12, Math.min(helpW, drawH));
        helpLabel.setBounds(rowX + Math.max(0, (helpW - helpSize) / 2), y + Math.max(0, (drawH - helpSize) / 2), helpSize, helpSize);
        y += drawH + rowGap;

        logPanel.setBounds(logX, y, logW, logH);
        y += logH + rowGap;

        actionPanel.setBounds(logX, y, logW, actionH);
        y += actionH + rowGap;
        warningLabel.setBounds(logX, y, logW, warningH);

        YoloUiUtils.applyResponsiveText(drawLabel, drawLabelW - 4, drawH);
        YoloUiUtils.applyResponsiveText(drawButton, drawBtnW - 8, drawH);
        YoloUiUtils.applyResponsiveText(refreshButton, refreshW - 8, drawH);
        warningLabel.setFont(actionPanel.getRunButton().getFont().deriveFont(Math.max((float) YoloUiUtils.MIN_FONT_SIZE, actionPanel.getRunButton().getFont().getSize2D() * 0.78f)));
    }

    public YoloModelSelectionPanel getModelSelectionPanel() {
        return modelSelectionPanel;
    }

    public YoloImageSourcePanel getImageSourcePanel() {
        return imageSourcePanel;
    }

    public YoloImageDisplayPanel getImageDisplayPanel() {
        return imageDisplayPanel;
    }

    public JButton getDrawButton() {
        return drawButton;
    }

    public JButton getRefreshButton() {
        return refreshButton;
    }

    public YoloHtmlLogPanel getLogPanel() {
        return logPanel;
    }

    public YoloActionPanel getActionPanel() {
        return actionPanel;
    }
}
