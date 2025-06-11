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

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

public class StarDistOptionalParams extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;

    private JSpinner minPercField, maxPercField;
    private JLabel minLabel, maxLabel;

    protected StarDistOptionalParams() {
        setLayout(null);
        SpinnerNumberModel modelL = new SpinnerNumberModel(1., 0., 100., 0.01);
        minPercField= new JSpinner(modelL);
        minLabel = new JLabel(StarDistGUI.VAR_NAMES.get(3));
        // Channel selection
        SpinnerNumberModel modelH = new SpinnerNumberModel(99.8, 0., 100., 0.01);
        maxPercField= new JSpinner(modelH);
        maxLabel = new JLabel(StarDistGUI.VAR_NAMES.get(4));
        add(minLabel);
        add(minPercField);
        add(maxLabel);
        add(maxPercField);
        setBorder(BorderFactory.createTitledBorder("Optional Parameters"));
        organiseComponents();
    }
    
    /**
     * 
     * @return the label for the minimum percentile
     */
    public JLabel getMinLabel() {
    	return this.minLabel;
    }
    
    /**
     * 
     * @return the label for the maximum percentile
     */
    public JLabel getMaxLabel() {
    	return this.maxLabel;
    }
    
    /**
     * 
     * @return the spinner for the minimum percentile
     */
    public JSpinner getMinPercField() {
    	return this.minPercField;
    }
    
    /**
     * 
     * @return the spinner for the maximum percentile
     */
    public JSpinner getMaxPercField() {
    	return this.maxPercField;
    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = 5;
                int nRows = 2;
                int rowH = (int) ((rawH - (inset * nRows)) / nRows);
                
                int y = inset;
                int modelLabelW = (rawW - inset * 3) * 4 / 5;
                
                int nRowH = (int) Math.max(StarDistGUI.ROW_RATIO * rowH, StarDistGUI.MIN_HEIGHT_ROW);
                int offsetH = (int) Math.max(Math.floor((rowH - nRowH) / 2), 0);
                minLabel.setBounds(inset, y + offsetH, modelLabelW, nRowH);
                int cboxW = rawW - inset * 3 - modelLabelW;
                minPercField.setBounds(inset * 2 + modelLabelW, y + offsetH, cboxW, nRowH);
                y += (inset + rowH);
                maxLabel.setBounds(inset, y + offsetH, modelLabelW, nRowH);
                maxPercField.setBounds(inset * 2 + modelLabelW, y + offsetH, cboxW, nRowH);
            }
        });
    }
}
