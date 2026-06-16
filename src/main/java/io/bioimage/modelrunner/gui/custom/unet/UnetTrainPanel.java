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

    /**
     * Creates a new UnetTrainPanel instance.
     */
    protected UnetTrainPanel() {
        super();
        setScratchArchitectures(UnetModelRegistry.buildScratchArchitectureEntries());
        scratchRadio.setSelected(true);
        baseModelComboBox.setEnabled(false);
        baseModelBrowseButton.setEnabled(false);
        scratchArchitectureComboBox.setEnabled(true);
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
