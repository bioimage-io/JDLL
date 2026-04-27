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

import java.awt.CardLayout;
import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.swing.ButtonGroup;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.filechooser.FileNameExtensionFilter;

public class YoloTrainPanel extends JPanel {

    private static final long serialVersionUID = -1127872966672515154L;

    protected static final int OUTER_PAD = 8;
    protected static final int ROW_GAP = 6;
    protected static final double GRAPH_WIDTH_RATIO = 0.9;
    protected static final double GRAPH_HEIGHT_RATIO = 0.4;
    protected static final double LABEL_WIDTH_RATIO = 0.24;
    protected static final double BROWSE_WIDTH_RATIO = 0.14;
    protected static final double RADIO_WIDTH_RATIO = 0.16;
    protected static final double EPOCH_WIDTH_RATIO = 0.18;
    protected static final double SWITCH_BUTTON_WIDTH_RATIO = 0.16;
    protected static final double TOP_ROW_UNITS = 1.0;
    protected static final double DATASET_ROW_UNITS = 1.0;
    protected static final double TUNE_ROW_UNITS = 1.0;
    protected static final double SCRATCH_ROW_UNITS = 1.0;
    protected static final double EPOCH_ROW_UNITS = 1.0;
    protected static final double SWITCH_ROW_UNITS = 0.9;

    protected final JLabel modelNameLabel = new JLabel("Model name");
    protected final YoloPlaceholderTextField modelNameField = new YoloPlaceholderTextField("name of the model to train");

    protected final JLabel datasetLabel = new JLabel("Training dataset");
    protected final YoloPlaceholderTextField datasetField = new YoloPlaceholderTextField("path to the training dataset");
    protected final JButton datasetBrowseButton = new JButton("Browse");

    protected final JRadioButton fineTuneRadio = new JRadioButton("Fine tune", true);
    protected final JComboBox<YoloModelSelectionEntry> baseModelComboBox = new JComboBox<YoloModelSelectionEntry>();
    protected final JButton baseModelBrowseButton = new JButton("Browse");

    protected final JRadioButton scratchRadio = new JRadioButton("Train from scratch");

    protected final JLabel epochsLabel = new JLabel("Epochs");
    protected final YoloIntegerTextField epochsField = new YoloIntegerTextField("100");

    protected final JButton lossButton = new JButton("Loss");
    protected final JButton metricButton = new JButton("Metric");
    protected final JButton validationPreviewButton = new JButton("Validation preview");
    protected final JPanel graphCardPanel = new JPanel(new CardLayout());
    protected final YoloGraphPlaceholderPanel lossGraphPanel = new YoloGraphPlaceholderPanel("Loss");
    protected final YoloGraphPlaceholderPanel metricGraphPanel = new YoloGraphPlaceholderPanel("Metric");
    protected final YoloValidationPreviewPanel validationPreviewPanel = new YoloValidationPreviewPanel();

    protected YoloTrainPanel() {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);

        ButtonGroup group = new ButtonGroup();
        group.add(fineTuneRadio);
        group.add(scratchRadio);

        YoloUiUtils.alignLabel(modelNameLabel);
        YoloUiUtils.alignLabel(datasetLabel);
        YoloUiUtils.alignLabel(epochsLabel);

        YoloUiUtils.styleInput(modelNameField);
        YoloUiUtils.styleInput(datasetField);
        baseModelComboBox.setEditable(true);
        YoloUiUtils.styleInput(baseModelComboBox);
        YoloUiUtils.styleInput(epochsField);

        YoloUiUtils.styleFlatSecondaryButton(datasetBrowseButton);
        YoloUiUtils.styleFlatSecondaryButton(baseModelBrowseButton);
        YoloUiUtils.styleFlatSecondaryButton(lossButton);
        YoloUiUtils.styleFlatSecondaryButton(metricButton);
        YoloUiUtils.styleFlatSecondaryButton(validationPreviewButton);

        graphCardPanel.setOpaque(false);
        graphCardPanel.add(lossGraphPanel, "loss");
        graphCardPanel.add(metricGraphPanel, "metric");
        graphCardPanel.add(validationPreviewPanel, "validationPreview");

        lossButton.addActionListener(e -> showGraph("loss"));
        metricButton.addActionListener(e -> showGraph("metric"));
        validationPreviewButton.addActionListener(e -> showGraph("validationPreview"));
        baseModelBrowseButton.addActionListener(e -> browseBaseModel());

        add(modelNameLabel);
        add(modelNameField);
        add(datasetLabel);
        add(datasetField);
        add(datasetBrowseButton);
        add(fineTuneRadio);
        add(baseModelComboBox);
        add(baseModelBrowseButton);
        add(scratchRadio);
        add(epochsLabel);
        add(epochsField);
        add(lossButton);
        add(metricButton);
        add(validationPreviewButton);
        add(graphCardPanel);

        updateMode();
        fineTuneRadio.addActionListener(e -> updateMode());
        scratchRadio.addActionListener(e -> updateMode());
    }

    private void showGraph(String name) {
        CardLayout layout = (CardLayout) graphCardPanel.getLayout();
        layout.show(graphCardPanel, name);
    }

    private void updateMode() {
        boolean fineTune = fineTuneRadio.isSelected();
        baseModelComboBox.setEnabled(fineTune);
        baseModelBrowseButton.setEnabled(fineTune);
    }

    private void browseBaseModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        chooser.setFileFilter(new FileNameExtensionFilter("YOLO weights (*.pt)", "pt"));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        if (selected != null) {
            setSelectedBaseModelValue(selected.getAbsolutePath());
        }
    }

    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int gap = Math.max(2, Math.min(ROW_GAP, h / 90));
        int innerW = Math.max(0, w - 2 * OUTER_PAD);
        int graphW = Math.max(1, (int) Math.round(innerW * GRAPH_WIDTH_RATIO));
        int graphBaseH = Math.max(1, (int) Math.round(h * GRAPH_HEIGHT_RATIO));

        int topAreaH = Math.max(1, h - 2 * OUTER_PAD - graphBaseH - gap);
        double rowUnits = TOP_ROW_UNITS + DATASET_ROW_UNITS + TUNE_ROW_UNITS + SCRATCH_ROW_UNITS + EPOCH_ROW_UNITS + SWITCH_ROW_UNITS;
        int rowUnitPx = Math.max(1, (int) Math.floor((topAreaH - 5 * gap) / rowUnits));

        int maxControlH = Math.max(1, YoloUiUtils.controlHeightForFontSize(YoloUiUtils.MAX_CONTROL_FONT_SIZE));
        int compactControlH = Math.max(1, maxControlH - 2);
        int row1H = Math.max(1, Math.min(compactControlH, (int) Math.round(rowUnitPx * TOP_ROW_UNITS)));
        int row2H = Math.max(1, Math.min(compactControlH, (int) Math.round(rowUnitPx * DATASET_ROW_UNITS)));
        int row3H = Math.max(1, Math.min(compactControlH, (int) Math.round(rowUnitPx * TUNE_ROW_UNITS)));
        int row4H = Math.max(1, Math.min(compactControlH, (int) Math.round(rowUnitPx * SCRATCH_ROW_UNITS)));
        int row5H = Math.max(1, Math.min(compactControlH, (int) Math.round(rowUnitPx * EPOCH_ROW_UNITS)));
        int row6H = Math.max(1, Math.min(maxControlH, Math.max(
                topAreaH - row1H - row2H - row3H - row4H - row5H - 5 * gap,
                (int) Math.round(rowUnitPx * SWITCH_ROW_UNITS))));
        int usedControlsH = row1H + row2H + row3H + row4H + row5H + row6H + 5 * gap;
        int graphH = Math.max(graphBaseH, h - 2 * OUTER_PAD - usedControlsH);

        int labelW = (int) Math.round(innerW * LABEL_WIDTH_RATIO);
        int browseW = (int) Math.round(innerW * BROWSE_WIDTH_RATIO);
        int radioW = (int) Math.round(innerW * RADIO_WIDTH_RATIO);
        int epochsW = (int) Math.round(innerW * EPOCH_WIDTH_RATIO);
        int switchBtnW = (int) Math.round(innerW * SWITCH_BUTTON_WIDTH_RATIO);
        int fieldW = Math.max(1, innerW - labelW - gap);

        int x = OUTER_PAD;
        int y = OUTER_PAD;

        modelNameLabel.setBounds(x, y, labelW, row1H);
        modelNameField.setBounds(x + labelW + gap, y, fieldW - labelW, row1H);
        y += row1H + gap;

        int datasetFieldW = Math.max(1, innerW - labelW - browseW - 2 * gap);
        datasetLabel.setBounds(x, y, labelW, row2H);
        datasetField.setBounds(x + labelW + gap, y, datasetFieldW, row2H);
        datasetBrowseButton.setBounds(x + labelW + datasetFieldW + 2 * gap, y, browseW, row2H);
        y += row2H + gap;

        int baseFieldW = Math.max(1, innerW - radioW - browseW - 2 * gap);
        fineTuneRadio.setBounds(x, y, radioW, row3H);
        baseModelComboBox.setBounds(x + radioW + gap, y, baseFieldW, row3H);
        baseModelBrowseButton.setBounds(x + radioW + baseFieldW + 2 * gap, y, browseW, row3H);
        y += row3H + gap;

        scratchRadio.setBounds(x, y, innerW, row4H);
        y += row4H + gap;

        epochsLabel.setBounds(x, y, labelW, row5H);
        epochsField.setBounds(x + labelW + gap, y, epochsW, row5H);
        y += row5H + gap;

        int totalSwitchW = 3 * switchBtnW + 2 * gap;
        int switchX = x + Math.max(0, (innerW - totalSwitchW) / 2);
        lossButton.setBounds(switchX, y, switchBtnW, row6H);
        metricButton.setBounds(lossButton.getX() + switchBtnW + gap, y, switchBtnW, row6H);
        validationPreviewButton.setBounds(metricButton.getX() + switchBtnW + gap, y, switchBtnW, row6H);

        int graphX = x + (innerW - graphW) / 2;
        graphCardPanel.setBounds(graphX, h - OUTER_PAD - graphH, graphW, graphH);

        YoloUiUtils.applyResponsiveText(modelNameLabel, labelW - 4, row1H);
        YoloUiUtils.applyResponsiveFont(modelNameField, row1H);
        YoloUiUtils.applyResponsiveText(datasetLabel, labelW - 4, row2H);
        YoloUiUtils.applyResponsiveFont(datasetField, row2H);
        YoloUiUtils.applyResponsiveText(datasetBrowseButton, browseW - 8, row2H);
        YoloUiUtils.applyResponsiveFont(fineTuneRadio, row3H);
        YoloUiUtils.applyResponsiveFont(baseModelComboBox, row3H);
        YoloUiUtils.applyResponsiveText(baseModelBrowseButton, browseW - 8, row3H);
        YoloUiUtils.applyResponsiveFont(scratchRadio, row4H);
        YoloUiUtils.applyResponsiveText(epochsLabel, labelW - 4, row5H);
        YoloUiUtils.applyResponsiveFont(epochsField, row5H);
        YoloUiUtils.applyResponsiveText(lossButton, switchBtnW - 8, row6H);
        YoloUiUtils.applyResponsiveText(metricButton, switchBtnW - 8, row6H);
        YoloUiUtils.applyResponsiveText(validationPreviewButton, switchBtnW - 8, row6H);
    }

    public YoloPlaceholderTextField getModelNameField() {
        return modelNameField;
    }

    public YoloPlaceholderTextField getDatasetField() {
        return datasetField;
    }

    public JButton getDatasetBrowseButton() {
        return datasetBrowseButton;
    }

    public JRadioButton getFineTuneRadio() {
        return fineTuneRadio;
    }

    public JComboBox<YoloModelSelectionEntry> getBaseModelComboBox() {
        return baseModelComboBox;
    }

    public void setBaseModels(LinkedHashMap<String, String> models) {
        DefaultComboBoxModel<YoloModelSelectionEntry> comboModel =
                new DefaultComboBoxModel<YoloModelSelectionEntry>();
        if (models != null) {
            for (Map.Entry<String, String> entry : models.entrySet()) {
                comboModel.addElement(new YoloModelSelectionEntry(entry.getKey(), entry.getValue()));
            }
        }
        baseModelComboBox.setModel(comboModel);
    }

    public String getSelectedBaseModelValue() {
        Object selected = baseModelComboBox.getSelectedItem();
        if (selected instanceof YoloModelSelectionEntry) {
            return ((YoloModelSelectionEntry) selected).getValue();
        }
        Object editorItem = baseModelComboBox.getEditor().getItem();
        return editorItem == null ? null : editorItem.toString().trim();
    }

    public void setSelectedBaseModelValue(String value) {
        if (value == null) {
            baseModelComboBox.setSelectedItem(null);
            return;
        }
        DefaultComboBoxModel<YoloModelSelectionEntry> model =
                (DefaultComboBoxModel<YoloModelSelectionEntry>) baseModelComboBox.getModel();
        for (int i = 0; i < model.getSize(); i++) {
            if (value.equals(model.getElementAt(i).getValue())) {
                baseModelComboBox.setSelectedIndex(i);
                return;
            }
        }
        baseModelComboBox.setSelectedItem(value);
    }

    public JButton getBaseModelBrowseButton() {
        return baseModelBrowseButton;
    }

    public JRadioButton getScratchRadio() {
        return scratchRadio;
    }

    public YoloIntegerTextField getEpochsField() {
        return epochsField;
    }

    public JPanel getGraphCardPanel() {
        return graphCardPanel;
    }

    public JButton getLossButton() {
        return lossButton;
    }

    public JButton getMetricButton() {
        return metricButton;
    }

    public JButton getValidationPreviewButton() {
        return validationPreviewButton;
    }
}
