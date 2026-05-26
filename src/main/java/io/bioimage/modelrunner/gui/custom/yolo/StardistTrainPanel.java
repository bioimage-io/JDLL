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

import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry;

public class StardistTrainPanel extends BaseTrainPanel {
    
    private static final long serialVersionUID = 3944729402784309789L;

	protected StardistTrainPanel() {
    	super();
        setScratchArchitectures(StardistModelRegistry.buildScratchArchitectureEntries());
        scratchRadio.setSelected(true);
        fineTuneRadio.setText("Fine tune (soon)");
        fineTuneRadio.setEnabled(false);
        baseModelComboBox.setEnabled(false);
        baseModelBrowseButton.setEnabled(false);
        scratchArchitectureComboBox.setEnabled(true);
    }

    protected void browseBaseModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        chooser.setFileFilter(new FileNameExtensionFilter("StarDist weights (*.mpk)", "mpk"));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        if (selected != null) {
            setSelectedBaseModelValue(selected.getAbsolutePath());
        }
    }

    protected boolean isValidModelFileName(String modelName) {
        if (modelName == null || modelName.trim().isEmpty()) {
            return false;
        }
        String cleanName = modelName.trim();
        if (cleanName.toLowerCase().endsWith(StardistModelRegistry.STARDIST_WEIGHTS_EXTENSION)) {
            cleanName = cleanName.substring(0, cleanName.length() - StardistModelRegistry.STARDIST_WEIGHTS_EXTENSION.length());
        }
        return !cleanName.isEmpty()
                && !cleanName.equals(".")
                && !cleanName.equals("..")
                && !cleanName.contains("..")
                && !INVALID_MODEL_NAME_CHARS.matcher(cleanName).find();
    }

    protected boolean isValidFineTuneBaseModel() {
        String baseModel = getSelectedBaseModelValue();
        if (baseModelComboBox.getSelectedItem() instanceof YoloModelSelectionEntry) {
            return baseModel != null
                    && (baseModel.toLowerCase().endsWith(StardistModelRegistry.STARDIST_WEIGHTS_EXTENSION)
                    || new File(baseModel).isDirectory());
        }
        return baseModel != null
                && (baseModel.toLowerCase().endsWith(StardistModelRegistry.STARDIST_WEIGHTS_EXTENSION)
                || new File(baseModel).isDirectory())
                && new File(baseModel).exists();
    }

    protected boolean isValidScratchArchitecture() {
        return StardistModelRegistry.isKnownScratchArchitecture(getSelectedScratchArchitectureValue());
    }

    @Override
    public void setTrainingRunning(boolean running) {
        super.setTrainingRunning(running);
        fineTuneRadio.setEnabled(false);
        baseModelComboBox.setEnabled(false);
        baseModelBrowseButton.setEnabled(false);
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
}
