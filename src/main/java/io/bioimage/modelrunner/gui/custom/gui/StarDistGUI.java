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

import javax.swing.ComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.awt.Dimension;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class StarDistGUI extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;
    
    protected JLabel modelLabel, customLabel, thresLabel;
	protected JComboBox<String> modelComboBox;
	protected PlaceholderTextField customModelPathField;
    protected JButton browseButton;
    protected ThresholdSlider thresholdSlider;
    protected StarDistOptionalParams optionalParams;
    protected FooterPanel footer;

    protected Double thresh1D = 0.479071463157368d;
    protected Double thresh3D = 0.6924782541382084d;
    protected HashMap<String, Double> threshMap = new HashMap<String, Double>();
    
    private static final ObjectMapper MAPPER = new ObjectMapper();
    
    protected final String CUSTOM_STR = "your custom model";
    protected static List<String> VAR_NAMES = Arrays.asList(new String[] {
    		"Select a model:", "Custom Model Path:", "Probability Threshold:", "Normalization low percentile:", "Normalization high percentile:"
    });
    
    protected static final double ROW_RATIO = 2d / 3d;
    
    protected static final int MIN_HEIGHT_ROW = 5;
    
    private static final Dimension MIN_D = new Dimension(20, 40);
    
    public static final String DEFAULT_1_CHANNEL_MODEL = "StarDist Fluorescence Nuclei Segmentation";
    
    public static final String DEFAULT_3_CHANNEL_MODEL = "StarDist H&E Nuclei Segmentation";

    protected StarDistGUI() {
        setLayout(null);

        // --- Model Selection Panel ---
        modelLabel = new JLabel(VAR_NAMES.get(0));
        String[] models = {DEFAULT_1_CHANNEL_MODEL, DEFAULT_3_CHANNEL_MODEL, CUSTOM_STR};
        modelComboBox = new JComboBox<String>(models){
    	    private static final long serialVersionUID = -1253338659158602375L;

			@Override
    	    public void setModel(ComboBoxModel<String> aModel) {
				if (this.dataModel == null) {
					super.setModel(aModel);
					return;
				}
				Object prevSelected = this.getSelectedItem();
				super.setModel(aModel);
				if (!prevSelected.equals(this.getSelectedItem()))
					this.setSelectedItem(this.getSelectedItem());
			}

			@Override
    	    public void setSelectedItem(Object anItem) {
    	        super.setSelectedItem(anItem);
    	        if (anItem.equals("StarDist Fluorescence Nuclei Segmentation") && thresh1D != null) {
    	        	thresholdSlider.getSlider().setValue((int) (thresh1D * 1000));
    	        } else if (anItem.equals("StarDist H&E Nuclei Segmentation") && thresh3D != null) {
    	        	StarDistGUI.this.thresholdSlider.getSlider().setValue((int) (thresh3D * 1000));
    	        } else if (anItem.equals(CUSTOM_STR) && threshMap.get(customModelPathField.getText().trim()) != null) {
    	        	thresholdSlider.getSlider().setValue((int) (threshMap.get(customModelPathField.getText().trim()) * 1000));
    	        } else if (anItem.equals(CUSTOM_STR) && new File(customModelPathField.getText()).isDirectory()
    	        		&& new File(customModelPathField.getText(), "thresholds.json").isFile()) {
    	        	try {
						Double prob = MAPPER.readTree(new File(customModelPathField.getText(), "thresholds.json")).get("prob").asDouble();
						threshMap.put(customModelPathField.getText().trim(), prob);
    	        	} catch (IOException e) {
        	        	thresholdSlider.getSlider().setValue(500);
					}
    	        } else {
    	        	thresholdSlider.getSlider().setValue(500);
    	        }
    	    }
    	};;

        // Panel for custom model file path
        customLabel = new JLabel(VAR_NAMES.get(1));
        customLabel.setEnabled(false);
        customModelPathField = new PlaceholderTextField("path to your custom model");
        customModelPathField.setEnabled(false);
        browseButton = new JButton("Browse");
        browseButton.setEnabled(false);

        thresLabel = new JLabel(VAR_NAMES.get(2));
        thresholdSlider = new ThresholdSlider();
        thresholdSlider.getSlider().setValue((int) (this.thresh1D * 1000));
        
        optionalParams = new StarDistOptionalParams();

        // --- Buttons Panel ---
        footer = new FooterPanel();
        add(footer);

		add(modelLabel);
        add(modelComboBox);
        add(customLabel);
        add(customModelPathField);
        add(browseButton);
        add(thresLabel);
        add(thresholdSlider);
        add(optionalParams);
        
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
                double nRows = nParams - 0.5 + 2.5;
                int rowH = (int) ((rawH - (inset * nRows)) / nRows);
                
                int y = inset;
                int modelLabelW = (rawW - inset * 3) / 5;
                
                int nRowH = (int) Math.max(ROW_RATIO * rowH, MIN_HEIGHT_ROW);
                int offsetH = (int) Math.max(Math.floor((rowH - nRowH) / 2), 0);
                modelLabel.setBounds(inset, y + offsetH, modelLabelW, nRowH);
                int cboxW = rawW - inset * 3 - modelLabelW;
                modelComboBox.setBounds(inset * 2 + modelLabelW, y + offsetH, cboxW, nRowH);
                y += (inset + rowH);
                int browseButtonW = modelLabelW / 1;
                int textFieldW = rawW - inset * 4 - modelLabelW - browseButtonW;
                customLabel.setBounds(inset, y + offsetH, modelLabelW, nRowH);
                customModelPathField.setBounds(inset * 2 + modelLabelW, y + offsetH, textFieldW, nRowH);
                browseButton.setBounds(inset * 3 + modelLabelW + textFieldW, y + offsetH, browseButtonW, nRowH);
                y += (inset + rowH);
                int offsetH2 = (int) Math.max(Math.floor((rowH * 2 - nRowH) / 2), 0);
                thresLabel.setBounds(inset, y + offsetH2, modelLabelW, nRowH);
                thresholdSlider.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH * 2);
                y += (inset + rowH * 2);
                optionalParams.setBounds(inset, y, rawW - 2 * inset, (int) (rowH * 2.5));
                y += (inset + rowH * 2.5);
                int sizeY = rawH - y - inset;
                footer.setBounds(inset, y, rawW - 2 * inset, sizeY);
            }
        });
    }
    
    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("StarDist Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new StarDistGUI());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setSize(60, 100);
            }
        });
    }
}
