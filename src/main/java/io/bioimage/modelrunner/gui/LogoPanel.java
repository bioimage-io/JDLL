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
package io.bioimage.modelrunner.gui;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.swing.JPanel;

public class LogoPanel extends JPanel{

	private static final long serialVersionUID = -8109832428317782274L;
	private BufferedImage image;
	private AtomicBoolean isDefault = new AtomicBoolean(true);
    private final PropertyChangeSupport pcs = new PropertyChangeSupport(this);

	
    LogoPanel(BufferedImage img) { 
    	this.image = img;
    	setOpaque(false); 
    }
	
    LogoPanel() { 
    	setOpaque(false); 
    }
    
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int side = Math.min(getWidth(), getHeight());
        if (image != null) {
            g.drawImage(image, 0, 0, getWidth(), getHeight(), this);
        } else {
            g.fillRect(0, 0, side, side);
        }
    }
    
    public void setImage(BufferedImage image, boolean defaultIm) {
    	if (!this.isDefault.get() && defaultIm)
    		return;
    	isDefault.set(false);
    	BufferedImage oldIm = this.image;
    	this.image = image;
    	pcs.firePropertyChange("image", oldIm, image);
        repaint();
    }
    
    public BufferedImage getImage() {
    	return this.image;
    }

    // listener registration
    public void addPropertyChangeListener(PropertyChangeListener l) {
    	if (pcs != null) {
            pcs.addPropertyChangeListener(l);
        } else {
            super.addPropertyChangeListener(l);
        }
    }
    
    public void removePropertyChangeListener(PropertyChangeListener l) {
        pcs.removePropertyChangeListener(l);
    }
}
