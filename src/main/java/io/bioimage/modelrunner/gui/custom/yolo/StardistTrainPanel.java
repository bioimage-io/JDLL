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

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JFileChooser;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileNameExtensionFilter;

import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry;
import io.bioimage.modelrunner.gui.custom.stardist.StardistValidationPreviewPanel;
import io.bioimage.modelrunner.gui.custom.unet.UnetDatasetInspector;

public class StardistTrainPanel extends BaseTrainPanel {
    
    private static final long serialVersionUID = 3944729402784309789L;
    private final Timer datasetReviewTimer;
    private LinkedHashMap<String, String> availableBaseModels = new LinkedHashMap<String, String>();
    private UnetDatasetInspector.Dimensionality datasetDimensionality = UnetDatasetInspector.Dimensionality.UNKNOWN;
    private String modelsDir;
    private long datasetReviewVersion;

    /**
     * Creates a new StardistTrainPanel instance.
     */
    protected StardistTrainPanel() {
    	super(new StardistValidationPreviewPanel());
        setScratchArchitectures(StardistModelRegistry.buildScratchArchitectureEntries());
        scratchRadio.setSelected(true);
        fineTuneRadio.setText("Fine tune");
        baseModelComboBox.setEnabled(false);
        baseModelBrowseButton.setEnabled(false);
        scratchArchitectureComboBox.setEnabled(true);
        datasetReviewTimer = new Timer(350, e -> reviewDatasetDimensionality());
        datasetReviewTimer.setRepeats(false);
        datasetField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                datasetReviewTimer.restart();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                datasetReviewTimer.restart();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                datasetReviewTimer.restart();
            }
        });
    }

    /**
     * Performs browse base model.
     */
    protected void browseBaseModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.setFileFilter(new FileNameExtensionFilter("StarDist weights (*.h5)", "h5"));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        if (selected != null) {
            setSelectedBaseModelValue(selected.getAbsolutePath());
        }
    }

    /**
     * Returns whether valid model file name.
     *
     * @param modelName the model name.
     * @return true if valid model file name; false otherwise.
     */
    protected boolean isValidModelFileName(String modelName) {
        if (modelName == null || modelName.trim().isEmpty()) {
            return false;
        }
        String cleanName = modelName.trim();
        if (cleanName.toLowerCase().endsWith(StardistModelRegistry.STARDIST_KERAS_WEIGHTS_EXTENSION)) {
            cleanName = cleanName.substring(0, cleanName.length() - StardistModelRegistry.STARDIST_KERAS_WEIGHTS_EXTENSION.length());
        }
        return !cleanName.isEmpty()
                && !cleanName.equals(".")
                && !cleanName.equals("..")
                && !cleanName.contains("..")
                && !INVALID_MODEL_NAME_CHARS.matcher(cleanName).find();
    }

    /**
     * Returns whether valid fine tune base model.
     *
     * @return true if valid fine tune base model; false otherwise.
     */
    protected boolean isValidFineTuneBaseModel() {
        String baseModel = getSelectedBaseModelValue();
        int dimensions = datasetDimensionality == UnetDatasetInspector.Dimensionality.THREE_D ? 3 : 2;
        return StardistModelRegistry.isSelectableFineTuneSource(baseModel, dimensions);
    }

    /**
     * Returns whether valid scratch architecture.
     *
     * @return true if valid scratch architecture; false otherwise.
     */
    protected boolean isValidScratchArchitecture() {
        return StardistModelRegistry.isKnownScratchArchitecture(getSelectedScratchArchitectureValue());
    }

    /**
     * Refreshes scratch architectures, adding a matching custom config when available.
     *
     * @param modelsDir the models directory.
     */
    public void refreshScratchArchitectures(String modelsDir) {
        this.modelsDir = modelsDir;
        String modelName = getModelNameField().getText();
        boolean reviewed = datasetDimensionality != UnetDatasetInspector.Dimensionality.UNKNOWN;
        boolean volume = datasetDimensionality == UnetDatasetInspector.Dimensionality.THREE_D;
        String custom = reviewed
                ? StardistModelRegistry.customScratchConfigValue(modelsDir, modelName, volume) : null;
        setScratchArchitectures(
                StardistModelRegistry.buildScratchArchitectureEntries(modelsDir, modelName, volume, reviewed), custom);
    }

    /**
     * Sets the training running.
     *
     * @param running the running.
     */
    @Override
    public void setTrainingRunning(boolean running) {
        super.setTrainingRunning(running);
    }

    /**
     * Sets the base models.
     *
     * @param models the models.
     */
    public void setBaseModels(LinkedHashMap<String, String> models) {
        availableBaseModels = models == null
                ? new LinkedHashMap<String, String>() : new LinkedHashMap<String, String>(models);
        refreshBaseModels();
    }

    private void refreshBaseModels() {
        DefaultComboBoxModel<YoloModelSelectionEntry> comboModel =
                new DefaultComboBoxModel<YoloModelSelectionEntry>();
        int dimensions = datasetDimensionality == UnetDatasetInspector.Dimensionality.THREE_D ? 3 : 2;
        if (availableBaseModels != null) {
            for (Map.Entry<String, String> entry : availableBaseModels.entrySet()) {
                if (datasetDimensionality != UnetDatasetInspector.Dimensionality.UNKNOWN
                        && !StardistModelRegistry.isSelectableFineTuneSource(entry.getValue(), dimensions)) {
                    continue;
                }
                comboModel.addElement(new YoloModelSelectionEntry(entry.getKey(), entry.getValue()));
            }
        }
        baseModelComboBox.setModel(comboModel);
    }

    private void reviewDatasetDimensionality() {
        String path = datasetField.getText() == null ? "" : datasetField.getText().trim();
        long version = ++datasetReviewVersion;
        if (path.isEmpty()) {
            applyDatasetDimensionality(version, UnetDatasetInspector.Dimensionality.UNKNOWN);
            return;
        }
        Thread review = new Thread(() -> {
            UnetDatasetInspector.Dimensionality dimensionality =
                    UnetDatasetInspector.inspect(new File(path));
            SwingUtilities.invokeLater(() -> applyDatasetDimensionality(version, dimensionality));
        }, "stardist-dataset-dimensionality");
        review.setDaemon(true);
        review.start();
    }

    private void applyDatasetDimensionality(long version, UnetDatasetInspector.Dimensionality dimensionality) {
        if (version != datasetReviewVersion) {
            return;
        }
        boolean changed = datasetDimensionality != dimensionality;
        datasetDimensionality = dimensionality;
        refreshScratchArchitectures(modelsDir);
        refreshBaseModels();
        if (changed) {
            scratchArchitectureComboBox.setToolTipText(
                    dimensionality == UnetDatasetInspector.Dimensionality.THREE_D
                            ? "The selected dataset contains volumes; only StarDist3D configurations are shown."
                            : "Choose StarDist model capacity for 2D training.");
        }
    }

    public int getDatasetDimensions() {
        return datasetDimensionality == UnetDatasetInspector.Dimensionality.THREE_D ? 3 : 2;
    }

    /**
     * Returns the validation preview panel.
     *
     * @return the validation preview panel.
     */
    @Override
    public StardistValidationPreviewPanel getValidationPreviewPanel() {
        return (StardistValidationPreviewPanel) super.getValidationPreviewPanel();
    }
}
