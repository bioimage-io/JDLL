package io.bioimage.modelrunner.gui;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.swing.JPanel;

public class LogoPanel extends JPanel{

	private static final long serialVersionUID = -8109832428317782274L;
	private BufferedImage image;
	private AtomicBoolean isDefault = new AtomicBoolean(true);

	
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
            g.drawImage(image, 0, 0, side, side, this);
        } else {
            g.fillRect(0, 0, side, side);
        }
    }
    
    public void setImage(BufferedImage image, boolean defaultIm) {
    	if (!this.isDefault.get() && defaultIm)
    		return;
    	isDefault.set(false);
    	this.image = image;
        repaint();
    }
    
    public BufferedImage getImage() {
    	return this.image;
    }
}
