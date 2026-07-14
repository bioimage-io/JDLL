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
package io.bioimage.modelrunner.gui.custom.unet;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

import io.bioimage.modelrunner.gui.custom.yolo.BaseTrainPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloModelSelectionEntry;

public class UnetTrainPanel extends BaseTrainPanel {

    private static final long serialVersionUID = 519461882197793765L;
    private boolean volumeArchitectureMode;
    private String modelsDir;
    private UnetDatasetInspector.Dimensionality datasetDimensionality = UnetDatasetInspector.Dimensionality.UNKNOWN;

    /**
     * Creates a new UnetTrainPanel instance.
     */
    protected UnetTrainPanel() {
        super();
        setScratchArchitectures(UnetModelRegistry.buildPlanarScratchArchitectureEntries());
        selectScratchArchitectureValue(UnetModelRegistry.defaultScratchArchitecture(false));
        scratchRadio.setSelected(true);
        baseModelComboBox.setEnabled(false);
        baseModelBrowseButton.setEnabled(false);
        scratchArchitectureComboBox.setEnabled(true);
        scratchArchitectureComboBox.setToolTipText("Choose model capacity for standard 2D training.");
    }

    /**
     * Performs browse base model.
     */
    @Override
    protected void browseBaseModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.setFileFilter(new FileNameExtensionFilter("UNet weights (*.pt, *.pth)", "pt", "pth"));
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
    @Override
    protected boolean isValidModelFileName(String modelName) {
        if (modelName == null || modelName.trim().isEmpty()) {
            return false;
        }
        String cleanName = UnetModelRegistry.removeWeightsExtension(modelName.trim());
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
    @Override
    protected boolean isValidFineTuneBaseModel() {
        return UnetModelRegistry.isModelPath(getSelectedBaseModelValue());
    }

    /**
     * Returns whether valid scratch architecture.
     *
     * @return true if valid scratch architecture; false otherwise.
     */
    @Override
    protected boolean isValidScratchArchitecture() {
        return UnetModelRegistry.isKnownScratchArchitecture(getSelectedScratchArchitectureValue());
    }

    /**
     * Updates scratch architectures according to the reviewed dataset dimensionality.
     *
     * @param dimensionality the inferred dataset dimensionality.
     */
    public void setDatasetDimensionality(UnetDatasetInspector.Dimensionality dimensionality) {
        boolean previousVolume = volumeArchitectureMode;
        datasetDimensionality = dimensionality == null ? UnetDatasetInspector.Dimensionality.UNKNOWN : dimensionality;
        boolean volume = dimensionality == UnetDatasetInspector.Dimensionality.THREE_D;
        volumeArchitectureMode = volume;
        refreshScratchArchitectures(volume != previousVolume);
        scratchArchitectureComboBox.setToolTipText(volume
                ? "Fast 3D trains with neighboring planes. True 3D trains on full volumetric patches."
                : "Choose model capacity for standard 2D training.");
    }

    /**
     * Sets the models directory used to discover custom scratch configs.
     *
     * @param modelsDir the models directory.
     */
    public void setModelsDir(String modelsDir) {
        this.modelsDir = modelsDir;
        refreshScratchArchitectures();
    }

    /**
     * Refreshes the scratch architectures.
     */
    public void refreshScratchArchitectures() {
        refreshScratchArchitectures(false);
    }

    private void refreshScratchArchitectures(boolean selectDefault) {
        boolean volume = datasetDimensionality == UnetDatasetInspector.Dimensionality.THREE_D;
        boolean reviewed = datasetDimensionality != UnetDatasetInspector.Dimensionality.UNKNOWN;
        String modelName = getModelNameField().getText();
        String custom = reviewed ? UnetModelRegistry.customScratchConfigValue(modelsDir, modelName, volume) : null;
        setScratchArchitectures(volume
                ? UnetModelRegistry.buildVolumeScratchArchitectureEntries(modelsDir, modelName, reviewed)
                : UnetModelRegistry.buildPlanarScratchArchitectureEntries(modelsDir, modelName, reviewed), custom);
        if (custom == null && selectDefault) {
            selectScratchArchitectureValue(UnetModelRegistry.defaultScratchArchitecture(volume));
        }
    }

    /**
     * Sets the base models.
     *
     * @param models the models.
     */
    @Override
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

}
