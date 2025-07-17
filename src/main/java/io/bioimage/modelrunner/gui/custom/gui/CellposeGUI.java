/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.gui.custom.gui;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import java.awt.Dimension;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class CellposeGUI extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;
    
    protected JLabel modelLabel, customLabel, cytoplasmLabel, nucleiLabel, diameterLabel;
	protected JComboBox<String> modelComboBox;
	protected PlaceholderTextField customModelPathField;
    protected JButton browseButton;
    protected IntegerTextField diameterField;
    protected JComboBox<String> cytoCbox, nucleiCbox;
    protected JCheckBox check;
    protected FooterPanel footer;
    
    protected final String CUSTOM_STR = "your custom model";
    protected static final List<String> VAR_NAMES = Arrays.asList(new String[] {
    		"Select a model:", "Custom Model Path:", "Cytoplasm Color:", "Nuclei Color:", "Diameter:", "Display all outputs"
    });

    public static final String[] RGB_LIST = new String[] {"red", "blue", "green"};
    public static final String[] GRAYSCALE_LIST = new String[] {"gray"};
    public static final String[] ALL_LIST = new String[] {"gray", "red", "blue", "green"};
    public static final HashMap<String, Integer> CHANNEL_MAP;
    static {
    	CHANNEL_MAP = new HashMap<String, Integer>();
    	CHANNEL_MAP.put("red", 1);
    	CHANNEL_MAP.put("blue", 2);
    	CHANNEL_MAP.put("green", 3);
    	CHANNEL_MAP.put("gray", 0);
    }
    private static final Dimension MIN_D = new Dimension(20, 40);

    protected CellposeGUI() {
        setLayout(null);

        // --- Model Selection Panel ---
        modelLabel = new JLabel(VAR_NAMES.get(0));
        String[] models = {"cyto3", "cyto2", "cyto", "nuclei", CUSTOM_STR};
        modelComboBox = new JComboBox<String>(models);

        // Panel for custom model file path
        customLabel = new JLabel(VAR_NAMES.get(1));
        customLabel.setEnabled(false);
        customModelPathField = new PlaceholderTextField("path to your custom model");
        customModelPathField.setEnabled(false);
        browseButton = new JButton("Browse");
        browseButton.setEnabled(false);

        // Channel selection
        cytoplasmLabel = new JLabel(VAR_NAMES.get(2));
        nucleiLabel = new JLabel(VAR_NAMES.get(3));
        cytoCbox = new JComboBox<String>(RGB_LIST);
        nucleiCbox = new JComboBox<String>(RGB_LIST);
        diameterLabel = new JLabel(VAR_NAMES.get(4));
        diameterField = new IntegerTextField("optional");
        check = new JCheckBox(VAR_NAMES.get(5));
        check.setSelected(false);

        // --- Buttons Panel ---
        footer = new FooterPanel();
        add(footer);

		add(modelLabel);
        add(modelComboBox);
        add(customLabel);
        add(customModelPathField);
        add(browseButton);
        add(cytoplasmLabel);
        add(cytoCbox);
        add(nucleiLabel);
        add(nucleiCbox);
        add(diameterLabel);
        add(diameterField);
        add(check);
        
        this.setMinimumSize(MIN_D);
        
        organiseComponents();

        // Enable when custom selected
        modelComboBox.addPopupMenuListener(new PopupMenuListener() {
            @Override
            public void popupMenuWillBecomeVisible(PopupMenuEvent e) {}
            @Override
            public void popupMenuCanceled(PopupMenuEvent e) {}

            @Override
            public void popupMenuWillBecomeInvisible(PopupMenuEvent e) {
            	boolean enabled = modelComboBox.getSelectedItem().equals(CUSTOM_STR);
                customLabel.setEnabled(enabled);
                customModelPathField.setEnabled(enabled);
                browseButton.setEnabled(enabled);
            }

        });

    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = 5;
                int nParams = VAR_NAMES.size();
                int nRows = nParams + 1;
                int rowH = (rawH - (inset * nRows)) / nRows;
                
                int y = inset;
                int modelLabelW = (rawW - inset * 3) / 5;
                modelLabel.setBounds(inset, y, modelLabelW, rowH);
                int cboxW = rawW - inset * 3 - modelLabelW;
                modelComboBox.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                int browseButtonW = modelLabelW / 1;
                int textFieldW = rawW - inset * 4 - modelLabelW - browseButtonW;
                customLabel.setBounds(inset, y, modelLabelW, rowH);
                customModelPathField.setBounds(inset * 2 + modelLabelW, y, textFieldW, rowH);
                browseButton.setBounds(inset * 3 + modelLabelW + textFieldW, y, browseButtonW, rowH);
                y += (inset + rowH);
                cytoplasmLabel.setBounds(inset, y, modelLabelW, rowH);
                cytoCbox.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                nucleiLabel.setBounds(inset, y, modelLabelW, rowH);
                nucleiCbox.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                diameterLabel.setBounds(inset, y, modelLabelW, rowH);
                diameterField.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                check.setBounds(inset, y, rawW - 2 * inset, rowH);
                y += (inset + rowH);
                footer.setBounds(inset, y, rawW - 2 * inset, rowH);
            }
        });
    }
    
    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("Cellpose Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new CellposeGUI());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setSize(60, 100);
            }
        });
    }
}
