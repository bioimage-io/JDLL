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

import java.awt.datatransfer.DataFlavor;
import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.TransferHandler;

public class YoloModelSelectionPanel extends JPanel {

    private static final long serialVersionUID = 1487902993174756571L;

    private static final int GAP = 6;
    private static final double LABEL_WIDTH_RATIO = 0.18;
    private static final double BUTTON_WIDTH_RATIO = 0.14;

    protected final JLabel modelLabel = new JLabel("YOLO model");
    protected final JComboBox<YoloModelSelectionEntry> modelComboBox = new JComboBox<YoloModelSelectionEntry>();
    protected final JButton browseButton = new JButton("Browse");

    /**
     * Creates a new YoloModelSelectionPanel instance.
     */
    protected YoloModelSelectionPanel() {
        setLayout(null);
        setOpaque(false);
        modelComboBox.setEditable(false);
        modelComboBox.setTransferHandler(new ModelDropHandler());
        YoloUiUtils.alignLabel(modelLabel);
        YoloUiUtils.styleInput(modelComboBox);
        YoloUiUtils.styleFlatSecondaryButton(browseButton);
        add(modelLabel);
        add(modelComboBox);
        add(browseButton);
    }

    /**
     * Performs do layout.
     */
    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int labelW = (int) Math.round(w * LABEL_WIDTH_RATIO);
        int buttonW = (int) Math.round(w * BUTTON_WIDTH_RATIO);
        int comboW = Math.max(1, w - labelW - buttonW - 2 * GAP);
        modelLabel.setBounds(0, 0, labelW, h);
        modelComboBox.setBounds(labelW + GAP, 0, comboW, h);
        browseButton.setBounds(labelW + GAP + comboW + GAP, 0, buttonW, h);
        YoloUiUtils.applyResponsiveText(modelLabel, labelW - 4, h);
        YoloUiUtils.applyResponsiveFont(modelComboBox, h);
        YoloUiUtils.applyResponsiveText(browseButton, buttonW - 8, h);
    }

    /**
     * Sets the models.
     *
     * @param models the models.
     */
    public void setModels(LinkedHashMap<String, String> models) {
        DefaultComboBoxModel<YoloModelSelectionEntry> comboModel =
                new DefaultComboBoxModel<YoloModelSelectionEntry>();
        if (models != null) {
            for (Map.Entry<String, String> entry : models.entrySet()) {
                comboModel.addElement(new YoloModelSelectionEntry(entry.getKey(), entry.getValue()));
            }
        }
        modelComboBox.setModel(comboModel);
    }

    /**
     * Sets the model label text.
     *
     * @param text the text.
     */
    public void setModelLabelText(String text) {
        modelLabel.setText(text == null ? "" : text);
        revalidate();
        repaint();
    }

    /**
     * Adds or selects a model entry.
     *
     * @param key the display key.
     * @param value the model path.
     */
    public void addOrSelectModel(String key, String value) {
        if (value == null || value.trim().isEmpty()) {
            return;
        }
        String cleanValue = value.trim();
        DefaultComboBoxModel<YoloModelSelectionEntry> model =
                (DefaultComboBoxModel<YoloModelSelectionEntry>) modelComboBox.getModel();
        for (int i = 0; i < model.getSize(); i++) {
            if (cleanValue.equals(model.getElementAt(i).getValue())) {
                modelComboBox.setSelectedIndex(i);
                return;
            }
        }
        model.addElement(new YoloModelSelectionEntry(key == null || key.trim().isEmpty() ? cleanValue : key,
                cleanValue));
        modelComboBox.setSelectedIndex(model.getSize() - 1);
    }

    /**
     * Returns the selected model key.
     *
     * @return the selected model key.
     */
    public String getSelectedModelKey() {
        YoloModelSelectionEntry selected = (YoloModelSelectionEntry) modelComboBox.getSelectedItem();
        return selected == null ? null : selected.getKey();
    }

    /**
     * Returns the selected model value.
     *
     * @return the selected model value.
     */
    public String getSelectedModelValue() {
        YoloModelSelectionEntry selected = (YoloModelSelectionEntry) modelComboBox.getSelectedItem();
        return selected == null ? null : selected.getValue();
    }

    /**
     * Returns the model combo box.
     *
     * @return the model combo box.
     */
    public JComboBox<YoloModelSelectionEntry> getModelComboBox() {
        return modelComboBox;
    }

    /**
     * Returns the browse button.
     *
     * @return the browse button.
     */
    public JButton getBrowseButton() {
        return browseButton;
    }

    private class ModelDropHandler extends TransferHandler {
        private static final long serialVersionUID = 3472435032725300446L;

        /**
         * Returns whether can import.
         *
         * @param support the support.
         * @return true if can import; false otherwise.
         */
        @Override
        public boolean canImport(TransferSupport support) {
            return support.isDataFlavorSupported(DataFlavor.javaFileListFlavor);
        }

        /**
         * Returns whether import data.
         *
         * @param support the support.
         * @return true if import data; false otherwise.
         */
        @Override
        public boolean importData(TransferSupport support) {
            try {
                @SuppressWarnings("unchecked")
                List<File> files = (List<File>) support.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
                if (files == null || files.isEmpty()) {
                    return false;
                }
                String path = files.get(0).getAbsolutePath();
                addOrSelectModel(path, path);
                return true;
            } catch (Exception e) {
                return false;
            }
        }
    }
}
