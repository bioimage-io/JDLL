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
import javax.swing.filechooser.FileNameExtensionFilter;

public class YoloTrainPanel extends BaseTrainPanel {

    private static final long serialVersionUID = -655892851236294330L;
    
    /**
     * Creates a new YoloTrainPanel instance.
     */
    protected YoloTrainPanel() {
    	super();
        setScratchArchitectures(YoloModelRegistry.buildScratchArchitectureEntries());
    }

    /**
     * Performs browse base model.
     */
    protected void browseBaseModel() {
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
        if (cleanName.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)) {
            cleanName = cleanName.substring(0, cleanName.length() - YoloModelRegistry.YOLO_WEIGHTS_EXTENSION.length());
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
        if (baseModelComboBox.getSelectedItem() instanceof YoloModelSelectionEntry) {
            return baseModel != null && baseModel.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
        }
        return baseModel != null
                && baseModel.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)
                && new File(baseModel).isFile();
    }

    /**
     * Returns whether valid scratch architecture.
     *
     * @return true if valid scratch architecture; false otherwise.
     */
    protected boolean isValidScratchArchitecture() {
        return YoloModelRegistry.isKnownScratchArchitecture(getSelectedScratchArchitectureValue());
    }

    /**
     * Sets the base models.
     *
     * @param models the models.
     */
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
