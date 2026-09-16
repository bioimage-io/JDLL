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
package io.bioimage.modelrunner.gui.custom.cellpose;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;

/**
 * Cellpose-specific inference controls.
 */
public class CellposeOptionsPanel extends JPanel {

    private static final long serialVersionUID = -1274330030238886907L;
    private static final int GAP = 6;
    private static final String[] CHANNELS = {"gray", "red", "green", "blue"};

    private final JLabel cytoplasmLabel = new JLabel("Cytoplasm color");
    private final JComboBox<String> cytoplasmComboBox = new JComboBox<String>(CHANNELS);
    private final JLabel nucleiLabel = new JLabel("Nuclei color");
    private final JComboBox<String> nucleiComboBox = new JComboBox<String>(CHANNELS);
    private final JCheckBox displayIntermediateOutputs =
            new JCheckBox("Display intermediate outputs");

    /**
     * Creates the Cellpose options panel.
     */
    public CellposeOptionsPanel() {
        setLayout(null);
        setOpaque(false);
        YoloUiUtils.alignLabel(cytoplasmLabel);
        YoloUiUtils.alignLabel(nucleiLabel);
        YoloUiUtils.styleInput(cytoplasmComboBox);
        YoloUiUtils.styleInput(nucleiComboBox);
        displayIntermediateOutputs.setOpaque(false);
        add(cytoplasmLabel);
        add(cytoplasmComboBox);
        add(nucleiLabel);
        add(nucleiComboBox);
        add(displayIntermediateOutputs);
    }

    /**
     * Lays out the two channel selectors and output option.
     */
    @Override
    public void doLayout() {
        int width = Math.max(0, getWidth());
        int height = Math.max(0, getHeight());
        int rowHeight = Math.max(1, (height - GAP) / 2);
        int columnWidth = Math.max(1, (width - 3 * GAP) / 4);
        int remainder = Math.max(1, width - 3 * columnWidth - 3 * GAP);

        cytoplasmLabel.setBounds(0, 0, columnWidth, rowHeight);
        cytoplasmComboBox.setBounds(columnWidth + GAP, 0, columnWidth, rowHeight);
        nucleiLabel.setBounds(2 * (columnWidth + GAP), 0, columnWidth, rowHeight);
        nucleiComboBox.setBounds(3 * (columnWidth + GAP), 0, remainder, rowHeight);
        displayIntermediateOutputs.setBounds(0, rowHeight + GAP, width, rowHeight);

        YoloUiUtils.applyResponsiveText(cytoplasmLabel, columnWidth - 4, rowHeight);
        YoloUiUtils.applyResponsiveFont(cytoplasmComboBox, rowHeight);
        YoloUiUtils.applyResponsiveText(nucleiLabel, columnWidth - 4, rowHeight);
        YoloUiUtils.applyResponsiveFont(nucleiComboBox, rowHeight);
        YoloUiUtils.applyResponsiveText(displayIntermediateOutputs, width - 8, rowHeight);
    }

    public JComboBox<String> getCytoplasmComboBox() {
        return cytoplasmComboBox;
    }

    public JComboBox<String> getNucleiComboBox() {
        return nucleiComboBox;
    }

    public JCheckBox getDisplayIntermediateOutputsCheckBox() {
        return displayIntermediateOutputs;
    }
}
