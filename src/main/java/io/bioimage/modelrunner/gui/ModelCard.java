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

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.net.URL;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;

public class ModelCard extends JPanel {

    private static final long serialVersionUID = -5625832740571130175L;

	private final URL defaultLogoURL;
    private final double scale;
    private final String cardID;
    
    private JLabel nameLabel;
    private JLabel nicknameLabel;
    private JLabel unsupportedLabel; // The overlay label
    protected LogoPanel logoIcon; // Panel holding image and overlay
    private boolean isUnsupported = false; // Flag to track state

    protected static final String UNSUPPORTED_TEXT = "UNSUPPORTED";
    protected static final Color UNSUPPORTED_BG_COLOR = Color.red;
    protected static final Color UNSUPPORTED_FG_COLOR = Color.black;


    protected ModelCard(GuiAdapter adapter) {
    	this(adapter, ModelSelectionPanelGui.MAIN_CARD_ID, 1d);
    }


    protected ModelCard(GuiAdapter adapter, String cardID, double scale) {
    	super(null);
		if (adapter == null)
			defaultLogoURL = null;
		else
			defaultLogoURL = ContentPanel.class.getClassLoader().getResource(adapter.getIconPath());
    	this.cardID = cardID;
    	this.scale = scale;
        this.setBackground(Color.WHITE);
        this.setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));


        this.logoIcon = new LogoPanel();

        this.unsupportedLabel = new JLabel(UNSUPPORTED_TEXT, JLabel.CENTER);
        this.unsupportedLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (24)));
        this.unsupportedLabel.setForeground(UNSUPPORTED_FG_COLOR);
        this.unsupportedLabel.setBackground(UNSUPPORTED_BG_COLOR);
        this.unsupportedLabel.setOpaque(true);
        this.unsupportedLabel.setBorder(BorderFactory.createEtchedBorder());
        this.unsupportedLabel.setVisible(true);


        this.nameLabel = new JLabel(Gui.LOADING_STR, JLabel.CENTER);
        this.nameLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (16 * scale)));
        this.nicknameLabel = new JLabel(Gui.LOADING_STR, JLabel.CENTER);
        this.nicknameLabel.setFont(new Font("SansSerif", Font.ITALIC, (int) (14 * scale)));

        this.add(this.nameLabel);
        this.add(this.unsupportedLabel);
        this.add(this.logoIcon);
        this.add(this.nicknameLabel);
        
        organiseComponents();
        hookImageListener();
    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                layoutAll();
            }
        });
    }
    
    private void hookImageListener() {
        logoIcon.addPropertyChangeListener(evt -> {
            if ("image".equals(evt.getPropertyName())) {
                layoutAll();
            }
        });
    }
    
    private void layoutAll() {
        Insets in = getInsets();
        int W = getWidth()  - in.left - in.right;
        int H = getHeight() - in.top  - in.bottom;
        
        int topInset = 2;
        int bottomInset = 2;
        int imTopInset = 2;
        int imBottomInset = 2;
        int sideInset = 2;
        

        nameLabel.setFont(nameLabel.getFont().deriveFont(Font.BOLD, (float) (16 * scale)));
        nicknameLabel.setFont(nicknameLabel.getFont().deriveFont(Font.PLAIN, (float) (14 * scale)));

        Dimension topSize = nameLabel.getPreferredSize();
        Dimension bottomSize = nicknameLabel.getPreferredSize();

        BufferedImage im = logoIcon.getImage();
        int imH, imW;
        if (im == null) {
        	imH = logoIcon.getPreferredSize().height;
        	imW = logoIcon.getPreferredSize().width;
        } else {
            imH = im.getHeight();
            imW = im.getWidth();
        }

        double newW, newH;
        int posx, posY;
        double ratio = imH / (double) imW;
        if (ratio > 1) {
        	newH = H - topInset - bottomInset - imTopInset - imBottomInset - topSize.height - bottomSize.height;
        	newW = newH /ratio;
            posY = imTopInset + topInset + topSize.height;
            posx = (int) (W / 2 - newW / 2);
        	if (newW > W + 2 * sideInset) {
        		newW = W - sideInset * 2;
                newH = newW * ratio;
                posx = sideInset;
                posY = (int) (H / 2 - newH / 2);
        	}
        } else {
            newW = W - sideInset * 2;
            newH = newW * ratio;
            posx = sideInset;
            posY = (int) (H / 2 - newH / 2);
        	if (newH > H - topInset - bottomInset - imTopInset - imBottomInset - topSize.height - bottomSize.height) {
            	newH = H - topInset - bottomInset - imTopInset - imBottomInset - topSize.height - bottomSize.height;
            	newW = newH /ratio;
                posY = imTopInset + topInset + topSize.height;
                posx = (int) (W / 2 - newW / 2);
        	}
        }
        newH = Math.max(1, newH);
        newW = Math.max(1, newW);
        int nameX = Math.max(1, Math.min(topSize.width, W - sideInset * 2));
        int nameY = Math.max(1, topSize.height);
        int nicknameNameX = Math.max(1, Math.min(bottomSize.width, W - sideInset * 2));
        int nicknameNameY = Math.max(1, bottomSize.height);

        int sideInsetName = Math.max(sideInset, W/ 2 - topSize.width / 2);
        int sideInsetNickname = Math.max(sideInset, W/ 2 - bottomSize.width / 2);
        
        int posYNick = Math.max(0, H - bottomInset - bottomSize.height);

        nameLabel  .setBounds(sideInsetName, topInset, nameX, nameY);
        nicknameLabel.setBounds(sideInsetNickname, posYNick, nicknameNameX, nicknameNameY);
        logoIcon.setBounds(posx, posY, (int) newW, (int) newH);
        
        int labelW = (int) newW;
        if (newW > 5) {
        	labelW = labelW - 4;
        }
        int labelY = posY;
        int labelH = (int) newH;
        if (newH > 15) {
        	labelH = (int) (newH / 3);
        	labelY += ((int) (newH / 2)) - (int) (labelH / 2);
        }

        int leftH = (int) (posYNick - imBottomInset - posY - newH);
        if (leftH > 30 && leftH > labelH) {
        	labelH = leftH;
        	labelY = (int) (posY + newH);
        }
        unsupportedLabel.setBounds(sideInset, labelY, getWidth() - sideInset * 2, labelH);
        unsupportedLabel.setVisible(isUnsupported);
        unsupportedLabel.setFont(unsupportedLabel.getFont().deriveFont(Font.BOLD, (float) (labelH / 3.5)));
    }

    /**
	 * Sets the visibility of the "Unsupported" overlay.
	 * @param unsupported true to show the overlay, false to hide it.
	 */
	public void setUnsupported(boolean unsupported) {
	    if (this.isUnsupported != unsupported) {
	        this.isUnsupported = unsupported;
	        this.unsupportedLabel.setVisible(unsupported);
	    }
	}

	/**
	 * Checks if the card is marked as unsupported.
	 * @return true if the unsupported overlay is set to be visible.
	 */
	public boolean isUnsupported() {
	    return this.isUnsupported;
	}

    protected static ModelCard createModelCard(GuiAdapter adapter) {
        ModelCard modelCardPanel = new ModelCard(adapter);
        return modelCardPanel;
    }

    protected static ModelCard createModelCard(GuiAdapter adapter, String id, double scale) {
        ModelCard modelCardPanel = new ModelCard(adapter, id, scale);
        return modelCardPanel;
    }

    protected void updateCard(String name, String nickname, URL imagePath) {
        this.nameLabel.setText(name);
        this.nicknameLabel.setText(nickname);
        if (defaultLogoURL != null)
        	DefaultIcon.drawImOrLogo(imagePath, defaultLogoURL, logoIcon, cardID);
        else
        	DefaultIcon.drawImOrLogo(imagePath, logoIcon, cardID);
    }

    protected void updateCard(String name, String nickname, URL imagePath, boolean supported) {
        this.nameLabel.setText(name);
        this.nicknameLabel.setText(nickname);
        this.setUnsupported(!supported);
        if (defaultLogoURL != null)
        	DefaultIcon.drawImOrLogo(imagePath, defaultLogoURL, logoIcon, cardID);
        else
        	DefaultIcon.drawImOrLogo(imagePath, logoIcon, cardID);
    }
}