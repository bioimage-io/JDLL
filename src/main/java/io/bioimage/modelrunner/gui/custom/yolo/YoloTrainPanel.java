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
import java.awt.Color;
import java.awt.datatransfer.DataFlavor;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import javax.swing.ButtonGroup;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;
import javax.swing.TransferHandler;
import javax.swing.filechooser.FileNameExtensionFilter;

public class YoloTrainPanel extends JPanel {

    private static final long serialVersionUID = -1127872966672515154L;

    protected static final int OUTER_PAD = 8;
    protected static final int ROW_GAP = 6;
    protected static final double GRAPH_WIDTH_RATIO = 0.9;
    protected static final double GRAPH_HEIGHT_RATIO = 0.4;
    protected static final double TRAIN_ACTION_HEIGHT_RATIO = 0.1;
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
    protected final JLabel epochsErrorLabel = new JLabel();

    protected final JButton lossButton = new JButton("Loss");
    protected final JButton metricButton = new JButton("Metric");
    protected final JButton validationPreviewButton = new JButton("Validation preview");
    protected final JPanel graphCardPanel = new JPanel(new CardLayout());
    protected final YoloGraphPlaceholderPanel lossGraphPanel = new YoloGraphPlaceholderPanel("Loss");
    protected final YoloGraphPlaceholderPanel metricGraphPanel = new YoloGraphPlaceholderPanel("Metric");
    protected final YoloValidationPreviewPanel validationPreviewPanel = new YoloValidationPreviewPanel();
    protected final YoloActionPanel trainActionPanel = new YoloActionPanel();
    private boolean trainingRunning;

    private static final Color ERROR_FG = new Color(255, 0, 0, 170);
    private static final Pattern INVALID_MODEL_NAME_CHARS = Pattern.compile("[\\\\/:*?\"<>|]");

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
        styleErrorLabel(epochsErrorLabel);

        YoloUiUtils.styleInput(modelNameField);
        YoloUiUtils.styleInput(datasetField);
        baseModelComboBox.setEditable(true);
        YoloUiUtils.styleInput(baseModelComboBox);
        YoloUiUtils.styleInput(epochsField);
        datasetField.setTransferHandler(new PathDropHandler(path -> datasetField.setText(path)));
        TransferHandler baseModelDropHandler = new PathDropHandler(this::setSelectedBaseModelValue);
        baseModelComboBox.setTransferHandler(baseModelDropHandler);
        ((JComponent) baseModelComboBox.getEditor().getEditorComponent()).setTransferHandler(baseModelDropHandler);

        YoloUiUtils.styleFlatSecondaryButton(datasetBrowseButton);
        YoloUiUtils.styleFlatSecondaryButton(baseModelBrowseButton);
        YoloUiUtils.styleFlatSecondaryButton(lossButton);
        YoloUiUtils.styleFlatSecondaryButton(metricButton);
        YoloUiUtils.styleFlatSecondaryButton(validationPreviewButton);
        trainActionPanel.getRunButton().setText("Train");
        trainActionPanel.getCancelButton().setEnabled(false);

        graphCardPanel.setOpaque(false);
        graphCardPanel.add(lossGraphPanel, "loss");
        graphCardPanel.add(metricGraphPanel, "metric");
        graphCardPanel.add(validationPreviewPanel, "validationPreview");

        lossButton.addActionListener(e -> showGraph("loss"));
        metricButton.addActionListener(e -> showGraph("metric"));
        validationPreviewButton.addActionListener(e -> showGraph("validationPreview"));
        datasetBrowseButton.addActionListener(e -> browseDatasetYaml());
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
        add(epochsErrorLabel);
        add(lossButton);
        add(metricButton);
        add(validationPreviewButton);
        add(graphCardPanel);
        add(trainActionPanel);

        updateMode();
        fineTuneRadio.addActionListener(e -> updateMode());
        scratchRadio.addActionListener(e -> updateMode());
    }

    private void showGraph(String name) {
        CardLayout layout = (CardLayout) graphCardPanel.getLayout();
        layout.show(graphCardPanel, name);
    }

    private static void styleErrorLabel(JLabel label) {
        label.setForeground(ERROR_FG);
        label.setText("");
    }

    private void updateMode() {
        boolean fineTune = fineTuneRadio.isSelected();
        baseModelComboBox.setEnabled(!trainingRunning && fineTune);
        baseModelBrowseButton.setEnabled(!trainingRunning && fineTune);
    }

    public void setTrainingRunning(boolean running) {
        this.trainingRunning = running;
        modelNameField.setEnabled(!running);
        datasetField.setEnabled(!running);
        datasetBrowseButton.setEnabled(!running);
        fineTuneRadio.setEnabled(!running);
        scratchRadio.setEnabled(!running);
        epochsField.setEnabled(!running);
        trainActionPanel.getRunButton().setEnabled(!running);
        trainActionPanel.getCancelButton().setEnabled(running);
        updateMode();
    }

    private void browseDatasetYaml() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        chooser.setFileFilter(new FileNameExtensionFilter("YOLO dataset YAML (*.yaml, *.yml)", "yaml", "yml"));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        if (selected != null) {
            datasetField.setText(selected.getAbsolutePath());
        }
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
        int maxControlH = Math.max(1, YoloUiUtils.controlHeightForFontSize(YoloUiUtils.MAX_CONTROL_FONT_SIZE));
        int actionH = Math.max(1, Math.min(maxControlH, (int) Math.round(h * TRAIN_ACTION_HEIGHT_RATIO)));
        int actionGap = gap;
        int graphBaseH = Math.max(1, (int) Math.round(h * GRAPH_HEIGHT_RATIO) - actionH - actionGap);

        int topAreaH = Math.max(1, h - 2 * OUTER_PAD - graphBaseH - actionH - actionGap - gap);
        double rowUnits = TOP_ROW_UNITS + DATASET_ROW_UNITS + TUNE_ROW_UNITS + SCRATCH_ROW_UNITS + EPOCH_ROW_UNITS + SWITCH_ROW_UNITS;
        int rowUnitPx = Math.max(1, (int) Math.floor((topAreaH - 5 * gap) / rowUnits));

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
        int graphH = Math.max(1, Math.max(graphBaseH, h - 2 * OUTER_PAD - usedControlsH - actionH - actionGap));

        int labelW = (int) Math.round(innerW * LABEL_WIDTH_RATIO);
        int browseW = (int) Math.round(innerW * BROWSE_WIDTH_RATIO);
        int radioW = (int) Math.round(innerW * RADIO_WIDTH_RATIO);
        int epochsW = (int) Math.round(innerW * EPOCH_WIDTH_RATIO);
        int switchBtnW = (int) Math.round(innerW * SWITCH_BUTTON_WIDTH_RATIO);
        int fieldW = Math.max(1, innerW - labelW - gap);

        int x = OUTER_PAD;
        int y = OUTER_PAD;

        modelNameLabel.setBounds(x, y, labelW, row1H);
        modelNameField.setBounds(x + labelW + gap, y, fieldW, row1H);
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
        int epochsErrorX = x + labelW + epochsW + 2 * gap;
        epochsErrorLabel.setBounds(epochsErrorX, y, Math.max(1, x + innerW - epochsErrorX), row5H);
        y += row5H + gap;

        int totalSwitchW = 3 * switchBtnW + 2 * gap;
        int switchX = x + Math.max(0, (innerW - totalSwitchW) / 2);
        lossButton.setBounds(switchX, y, switchBtnW, row6H);
        metricButton.setBounds(lossButton.getX() + switchBtnW + gap, y, switchBtnW, row6H);
        validationPreviewButton.setBounds(metricButton.getX() + switchBtnW + gap, y, switchBtnW, row6H);

        int graphX = x + (innerW - graphW) / 2;
        int actionW = graphW;
        int actionX = graphX;
        int actionY = h - OUTER_PAD - actionH;
        graphCardPanel.setBounds(graphX, actionY - actionGap - graphH, graphW, graphH);
        trainActionPanel.setBounds(actionX, actionY, actionW, actionH);

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
        epochsErrorLabel.setFont(epochsLabel.getFont());
        YoloUiUtils.applyResponsiveText(lossButton, switchBtnW - 8, row6H);
        YoloUiUtils.applyResponsiveText(metricButton, switchBtnW - 8, row6H);
        YoloUiUtils.applyResponsiveText(validationPreviewButton, switchBtnW - 8, row6H);
        trainActionPanel.doLayout();
    }

    public YoloPlaceholderTextField getModelNameField() {
        return modelNameField;
    }

    public boolean validateTrainingFields() {
        clearTrainingErrors();
        boolean valid = true;

        String modelName = modelNameField.getText() == null ? "" : modelNameField.getText().trim();
        if (!isValidModelFileName(modelName)) {
            modelNameField.setText("");
            modelNameField.setErrorPlaceholder("Please enter a valid model name");
            valid = false;
        }

        File datasetYaml = resolveDatasetYaml(datasetField.getText());
        if (datasetYaml == null || !isYoloDatasetYaml(datasetYaml)) {
            datasetField.setText("");
            datasetField.setErrorPlaceholder("Please select a valid YOLO dataset");
            valid = false;
        } else {
            datasetField.setText(datasetYaml.getAbsolutePath());
        }

        if (!isValidEpochs()) {
            epochsErrorLabel.setText("Please enter epochs > 0");
            valid = false;
        }

        if (fineTuneRadio.isSelected() && !isValidFineTuneBaseModel()) {
            setComboBoxErrorPlaceholder(baseModelComboBox, "Please select a valid .pt model");
            valid = false;
        }

        revalidate();
        repaint();
        return valid;
    }

    public void clearTrainingErrors() {
        modelNameField.clearErrorPlaceholder();
        datasetField.clearErrorPlaceholder();
        setComboBoxErrorPlaceholder(baseModelComboBox, null);
        epochsErrorLabel.setText("");
    }

    private static void setComboBoxErrorPlaceholder(JComboBox<?> comboBox, String message) {
        JComponent editor = (JComponent) comboBox.getEditor().getEditorComponent();
        if (editor instanceof JTextField) {
            JTextField field = (JTextField) editor;
            if (message != null) {
                comboBox.setSelectedItem(null);
                field.setText(message);
            }
            field.setForeground(message == null ? Color.BLACK : ERROR_FG);
            field.repaint();
        }
        comboBox.repaint();
    }

    private boolean isValidModelFileName(String modelName) {
        if (modelName == null || modelName.trim().isEmpty()) {
            return false;
        }
        String cleanName = modelName.trim();
        if (cleanName.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)) {
            cleanName = cleanName.substring(0, cleanName.length() - YoloModelRegistry.YOLO_WEIGHTS_EXTENSION.length());
        }
        return !cleanName.isEmpty()
                && !cleanName.equals(".")
                && !cleanName.equals("..")
                && !cleanName.contains("..")
                && !INVALID_MODEL_NAME_CHARS.matcher(cleanName).find();
    }

    private boolean isValidEpochs() {
        try {
            return Integer.parseInt(epochsField.getText().trim()) > 0;
        } catch (Exception e) {
            return false;
        }
    }

    private boolean isValidFineTuneBaseModel() {
        String baseModel = getSelectedBaseModelValue();
        if (baseModelComboBox.getSelectedItem() instanceof YoloModelSelectionEntry) {
            return baseModel != null && baseModel.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
        }
        return baseModel != null
                && baseModel.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)
                && new File(baseModel).isFile();
    }

    private static File resolveDatasetYaml(String datasetPath) {
        if (datasetPath == null || datasetPath.trim().isEmpty()) {
            return null;
        }
        File path = new File(datasetPath.trim());
        if (path.isFile() && isYamlFile(path)) {
            return path;
        }
        if (!path.isDirectory()) {
            return null;
        }
        File directYaml = firstExisting(path, "data.yaml", "data.yml", "dataset.yaml", "dataset.yml");
        if (directYaml != null) {
            return directYaml;
        }
        try (Stream<java.nio.file.Path> stream = Files.walk(path.toPath(), 3)) {
            return stream
                    .filter(p -> Files.isRegularFile(p) && isYamlFile(p.toFile()))
                    .map(java.nio.file.Path::toFile)
                    .filter(YoloTrainPanel::isYoloDatasetYaml)
                    .findFirst()
                    .orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    private static File firstExisting(File dir, String... names) {
        for (String name : names) {
            File candidate = new File(dir, name);
            if (candidate.isFile()) {
                return candidate;
            }
        }
        return null;
    }

    private static boolean isYamlFile(File file) {
        String name = file.getName().toLowerCase();
        return name.endsWith(".yaml") || name.endsWith(".yml");
    }

    private static boolean isYoloDatasetYaml(File file) {
        try {
            String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8).toLowerCase();
            return hasYamlKey(content, "train")
                    && (hasYamlKey(content, "val") || hasYamlKey(content, "validation"))
                    && hasYamlKey(content, "names");
        } catch (IOException e) {
            return false;
        }
    }

    private static boolean hasYamlKey(String content, String key) {
        return Pattern.compile("(?m)^\\s*" + Pattern.quote(key) + "\\s*:").matcher(content).find();
    }

    private static class PathDropHandler extends TransferHandler {
        private static final long serialVersionUID = 2448864258216265787L;

        private final java.util.function.Consumer<String> pathConsumer;

        private PathDropHandler(java.util.function.Consumer<String> pathConsumer) {
            this.pathConsumer = pathConsumer;
        }

        @Override
        public boolean canImport(TransferSupport support) {
            return support.isDataFlavorSupported(DataFlavor.javaFileListFlavor);
        }

        @Override
        public boolean importData(TransferSupport support) {
            try {
                @SuppressWarnings("unchecked")
                List<File> files = (List<File>) support.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
                if (files == null || files.isEmpty()) {
                    return false;
                }
                pathConsumer.accept(files.get(0).getAbsolutePath());
                return true;
            } catch (Exception e) {
                return false;
            }
        }
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

    public YoloGraphPlaceholderPanel getLossGraphPanel() {
        return lossGraphPanel;
    }

    public YoloGraphPlaceholderPanel getMetricGraphPanel() {
        return metricGraphPanel;
    }

    public YoloValidationPreviewPanel getValidationPreviewPanel() {
        return validationPreviewPanel;
    }

    public YoloActionPanel getTrainActionPanel() {
        return trainActionPanel;
    }
}
