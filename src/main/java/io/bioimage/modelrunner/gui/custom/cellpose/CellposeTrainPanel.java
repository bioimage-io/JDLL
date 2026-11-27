package io.bioimage.modelrunner.gui.custom.cellpose;

import io.bioimage.modelrunner.model.special.cellpose.CellposeModels;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.io.File;
import java.util.LinkedHashMap;

import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import io.bioimage.modelrunner.gui.custom.unet.UnetTrainPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloModelSelectionEntry;

public final class CellposeTrainPanel extends UnetTrainPanel {
    private static final long serialVersionUID = 1L;
    private final JPanel channelRow = new JPanel(new GridLayout(1, 4, 3, 0));
    private final JSpinner primary = channel("Cell intensity channel (1-based). Zero uses the mean of all channels.");
    private final JSpinner nuclei = channel("Optional nuclear channel (1-based). Zero leaves it empty.");

    public CellposeTrainPanel() {
        LinkedHashMap<String,String> architectures = new LinkedHashMap<>();
        architectures.put("Cyto3 (2D CPnet)", CellposeModels.NATIVE);
        configureFixedArchitectures(architectures, CellposeModels.NATIVE, CellposeModels::isSelectable,
                CellposeModels.NATIVE::equals, "Native Cellpose weights", "pt", "pth");
        scratchArchitectureComboBox.setToolTipText("Cellpose 3 cyto3 architecture with freshly initialized weights.");
        datasetField.setToolTipText("Paired images/ and masks/ folders, optionally inside train/ and val/. Volumes train as XY planes.");
        channelRow.add(new JLabel("Cell ch."));
        channelRow.add(primary);
        channelRow.add(new JLabel("Nuclei ch."));
        channelRow.add(nuclei);
        add(channelRow);
        getFineTuneRadio().doClick();
        baseModelComboBox.addActionListener(e -> {
            String value = getSelectedBaseModelValue();
            if (value == null || !new File(value).exists()) return;
            try {
                int[] channels = CellposeModels.channels(new File(value));
                primary.setValue(channels[0]);
                nuclei.setValue(channels[1]);
            } catch (java.io.IOException ignored) { /* The chooser validates the selected model. */ }
        });
    }

    private static JSpinner channel(String tooltip) {
        JSpinner spinner = new JSpinner(new SpinnerNumberModel(0, 0, Integer.MAX_VALUE, 1));
        spinner.setToolTipText(tooltip);
        return spinner;
    }

    public int[] channels() throws java.text.ParseException {
        primary.commitEdit();
        nuclei.commitEdit();
        return new int[] {(int) primary.getValue(), (int) nuclei.getValue()};
    }

    @Override public void doLayout() {
        super.doLayout();
        Rectangle bounds = epochsErrorLabel.getBounds();
        channelRow.setBounds(bounds);
        channelRow.setVisible(epochsErrorLabel.getText() == null || epochsErrorLabel.getText().isEmpty());
    }

    @Override public void setTrainingRunning(boolean active) {
        super.setTrainingRunning(active);
        primary.setEnabled(!active);
        nuclei.setEnabled(!active);
    }

    @Override public boolean validateTrainingFields() {
        boolean valid = super.validateTrainingFields();
        if ("1".equals(epochsField.getText().trim()) && (!fineTuneRadio.isSelected() ||
                io.bioimage.modelrunner.gui.custom.training.TrainingContinuation.find(getSelectedBaseModelValue()) == null)) {
            epochsErrorLabel.setText("Use at least 2 epochs (first is warm-up)");
            revalidate();
            return false;
        }
        return valid;
    }

    @Override protected boolean isValidDatasetPath(File file) { return file.isDirectory(); }

    @Override protected void browseBaseModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = chooser.getSelectedFile();
            if (!CellposeModels.isSelectable(file.getAbsolutePath())) {
                javax.swing.JOptionPane.showMessageDialog(this, "Select native Cellpose weights or a trained model folder.");
                return;
            }
            baseModelComboBox.addItem(new YoloModelSelectionEntry(file.getName(), file.getAbsolutePath()));
            setSelectedBaseModelValue(file.getAbsolutePath());
        }
    }
}
