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
package io.bioimage.modelrunner.gui.custom;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingUtilities;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

public class FooterPanel extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;
    
    protected ButtonsPanel buttons;
    
    protected JProgressBar bar;
    
    private static final double BUTTON_WRATIO = 1d / 2d;
    
    private static final Dimension MIN_D = new Dimension(20, 10);

    protected FooterPanel() {
        setLayout(null);
        

        buttons = new ButtonsPanel();
        bar = new JProgressBar();
		bar.setStringPainted(true);
		bar.setString("");
        add(buttons);
        add(bar);
        
        this.setMinimumSize(MIN_D);
        
        organiseComponents();

    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = 2;
                int w = (int) ((rawW - inset) * BUTTON_WRATIO);
                bar.setBounds(0, inset, w, rawH - 2 * inset);
                buttons.setBounds(w + inset, 0, w, rawH);
            }
        });
    }
    
    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("Buttons pannel");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new FooterPanel());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setSize(200, 200);
            }
        });
    }
}
