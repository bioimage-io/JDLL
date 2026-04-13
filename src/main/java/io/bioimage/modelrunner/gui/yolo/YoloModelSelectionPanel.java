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
package io.bioimage.modelrunner.gui.yolo;

import java.awt.datatransfer.DataFlavor;
import java.io.File;
import java.util.List;

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
    protected final JComboBox<String> modelComboBox = new JComboBox<String>();
    protected final JButton browseButton = new JButton("Browse");

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

    public void setModels(List<String> models) {
        modelComboBox.setModel(new DefaultComboBoxModel<String>(models.toArray(new String[0])));
    }

    public JComboBox<String> getModelComboBox() {
        return modelComboBox;
    }

    public JButton getBrowseButton() {
        return browseButton;
    }

    private class ModelDropHandler extends TransferHandler {
        private static final long serialVersionUID = 3472435032725300446L;

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
                String path = files.get(0).getAbsolutePath();
                DefaultComboBoxModel<String> model = (DefaultComboBoxModel<String>) modelComboBox.getModel();
                boolean found = false;
                for (int i = 0; i < model.getSize(); i++) {
                    if (path.equals(model.getElementAt(i))) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    model.addElement(path);
                }
                modelComboBox.setSelectedItem(path);
                return true;
            } catch (Exception e) {
                return false;
            }
        }
    }
}
