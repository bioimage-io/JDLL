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

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

public class ButtonsPanel extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;
    
    protected JButton cancelButton, installButton, runButton;
    
    private static final double BUTTON_WRATIO = 1d / 3d;

    protected ButtonsPanel() {
        setLayout(null);

        // --- Buttons Panel ---
        cancelButton = new JButton("Cancel");
        installButton = new JButton("Install");
        runButton = new JButton("Run");
        add(cancelButton);
        add(installButton);
        add(runButton);
        
        organiseComponents();

    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = 2;
                int w = (int) ((rawW - 2 * inset) * BUTTON_WRATIO);
                cancelButton.setBounds(0, 0, w, rawH);
                installButton.setBounds(w + inset, 0, w, rawH);
                runButton.setBounds((w + inset) * 2, 0, w, rawH);
            }
        });
    }
    
    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("Buttons pannel");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new ButtonsPanel());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setSize(200, 200);
            }
        });
    }
}
