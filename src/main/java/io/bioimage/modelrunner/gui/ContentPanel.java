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

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.net.URL;

import javax.swing.BorderFactory;
import javax.swing.JEditorPane;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;
import javax.swing.event.HyperlinkEvent;
import javax.swing.event.HyperlinkListener;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.workers.ModelInfoWorker;
import io.bioimage.modelrunner.gui.workers.ModelInfoWorker.TextLoadCallback;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;

public class ContentPanel extends JPanel {
	
	private final URL defaultLogoURL;
	private boolean isUnsupported = false;
	
	private LogoPanel exampleImageLabel;
	private JLabel exampleTitleLabel;
	private JLabel unsupportedLabel;
	protected JLabel infoTitleLabel;
    private JEditorPane modelInfoArea;
    private JProgressBar progressBar;
	private JLabel progressInfoLabel;
	private JScrollPane infoScrollPane;

	private final static double BAR_RATIO = 0.05;
	private final static double LABEL_RATIO = 0.05;
	
	private static final long serialVersionUID = -7691139174208436363L;

	protected ContentPanel() {
		this(null);
	}

	protected ContentPanel(GuiAdapter adapter) {
		super(null);
		
		if (adapter == null) {
			defaultLogoURL = null;
		} else {
			defaultLogoURL = ContentPanel.class.getClassLoader().getResource(adapter.getIconPath());
		}
		

        this.unsupportedLabel = new JLabel(ModelCard.UNSUPPORTED_TEXT, JLabel.CENTER);
        this.unsupportedLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (16)));
        this.unsupportedLabel.setForeground(ModelCard.UNSUPPORTED_FG_COLOR);
        this.unsupportedLabel.setBackground(ModelCard.UNSUPPORTED_BG_COLOR);
        this.unsupportedLabel.setOpaque(true);
        this.unsupportedLabel.setBorder(BorderFactory.createEtchedBorder());
        this.unsupportedLabel.setVisible(true);

        exampleTitleLabel = new JLabel("Cover Image");
        exampleTitleLabel.setFont(new Font("SansSerif", Font.BOLD, 24));

        // Calculate dimensions for the logo based on the main interface size
        exampleImageLabel = new LogoPanel();

        infoTitleLabel = new JLabel("Model Information");
        infoTitleLabel.setFont(new Font("SansSerif", Font.BOLD, 18));

        modelInfoArea = new JEditorPane("text/html", "Detailed model description...");
        modelInfoArea.setEditable(false);
        modelInfoArea.addHyperlinkListener(new HyperlinkListener() {
            public void hyperlinkUpdate(HyperlinkEvent e) {
                if (e.getEventType() == HyperlinkEvent.EventType.ACTIVATED) {
                    try {
                        Desktop.getDesktop().browse(e.getURL().toURI());
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            }
        });
        infoScrollPane = new JScrollPane(modelInfoArea);

        createProgressBar();

        add(unsupportedLabel);
        add(exampleImageLabel);
        add(exampleTitleLabel);
        add(infoTitleLabel);
        add(progressBar);
        add(progressInfoLabel);
        add(infoScrollPane);
        hookImageListener();
        organiseComponents();
	}
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                
                int inset = 4;
                
                int xRight = inset + rawW / 2;
                
                int spaceX = rawW / 2 - inset * 2;
                
                leftSideGUI(rawH, rawW, spaceX, inset);
                rightSideGUI(rawH, rawW, inset, spaceX, xRight);
            }
        });
    }
    
    private void rightSideGUI(int rawH, int rawW, int inset, int spaceX, int xRight) {
        Dimension rightLabelSize = infoTitleLabel.getPreferredSize();
        int labelPosX = Math.max(xRight, 3 * rawW / 4 - rightLabelSize.width / 2);
        int labelW = Math.max(1, Math.min(rightLabelSize.width, rawW / 2 + spaceX +  2 * inset - labelPosX));
        infoTitleLabel.setBounds(labelPosX, inset, labelW, rightLabelSize.height);

        double barHeight = Math.max(1, rawH * BAR_RATIO);
        double strHeight = Math.max(1, rawH * LABEL_RATIO);
        
        int barInset = 2;
        
        double hPanel = rawH - barInset - 4 * inset - barHeight - strHeight - rightLabelSize.height;
        double wPanel = rawW / 2 - inset * 2;
        int posY = 2 * inset + rightLabelSize.height;
        

        wPanel = Math.max(1, wPanel);
        hPanel = Math.max(1, hPanel);
        barHeight = Math.max(5, barHeight);
        strHeight = Math.max(1, strHeight);
        
        infoScrollPane.setBounds(xRight, posY, (int) wPanel, (int) hPanel);
        posY += hPanel + inset;
        progressBar.setBounds(xRight, posY, (int) wPanel, (int) barHeight);
        posY += barHeight + barInset;
        progressInfoLabel.setBounds(xRight, posY, (int) wPanel, (int) strHeight);        
    }
    
    private void leftSideGUI(int H, int W, int spaceX, int inset) {

        int xLeft = inset;
        Dimension leftLabelSize = exampleTitleLabel.getPreferredSize();
        int labelPosX = Math.max(xLeft, W / 4 - leftLabelSize.width / 2);
        int titleW = Math.min(leftLabelSize.width, spaceX + inset - labelPosX);
        titleW = Math.max(1, titleW);
        exampleTitleLabel.setBounds(labelPosX, inset, titleW, leftLabelSize.height);
        
        BufferedImage im = exampleImageLabel.getImage();
        int imH, imW;
        if (im == null) {
        	imH = exampleImageLabel.getPreferredSize().height;
        	imW = exampleImageLabel.getPreferredSize().width;
        } else {
            imH = im.getHeight();
            imW = im.getWidth();
        }
        
        double newW, newH;
        int posx, posY;
        double ratio = imH / (double) imW;
        if (ratio > 1) {
        	newH = H - inset * 3 - leftLabelSize.height;
        	newW = newH / ratio;
            posY = leftLabelSize.height + inset * 2;
            posx = (int) (inset + spaceX / 2 - newW / 2);
            if (posx < 0) {
        		newW = spaceX;
                newH = newW * ratio;
                posx = inset;
                posY = (int) ((H - 3 * inset - leftLabelSize.height) / 2 + 2 * inset + leftLabelSize.height - newH / 2);
            }
        } else {
            newW = spaceX;
            newH = newW * ratio;
            posx = inset;
            posY = (int) ((H - 3 * inset - leftLabelSize.height) / 2 + 2 * inset + leftLabelSize.height - newH / 2);
        	if (posY < 2 * inset + leftLabelSize.height) {
            	newH = H - inset * 3 - leftLabelSize.height;
            	newW = newH / ratio;
                posY = leftLabelSize.height + inset * 2;
                posx = (int) (inset + spaceX / 2 - newW / 2);
        	}
        }
        newW = Math.max(1, newW);
        newH = Math.max(1, newH);
        exampleImageLabel.setBounds(posx, posY, (int) newW, (int) newH);
        


        
        unsupportedLabel.setFont(unsupportedLabel.getFont().deriveFont(Font.BOLD, (float) (16)));
        
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
        unsupportedLabel.setBounds(xLeft, labelY, spaceX, labelH);
        unsupportedLabel.setVisible(isUnsupported);
    }
    
    private void hookImageListener() {
    	exampleImageLabel.addPropertyChangeListener(evt -> {
            if ("image".equals(evt.getPropertyName())) {
                int rawW = getWidth();
                int rawH = getHeight();
                
                int inset = 4;
                
                int spaceX = rawW / 2 - inset * 2;
                
                leftSideGUI(rawH, rawW, spaceX, inset);
            }
        });
    }
    
    protected boolean isUnsupported() {
    	return this.isUnsupported;
    }
    
    protected void setUnsupported(boolean isUnsupported) {
    	this.isUnsupported = isUnsupported;
    }
	
	protected void setIcon(BufferedImage im) {
		this.exampleImageLabel.setImage(im, false);
	}
	
	protected void setInfo(String text) {
		SwingUtilities.invokeLater(() -> {
			this.modelInfoArea.setText(text);
			modelInfoArea.setCaretPosition(0);
		});
	}
	
	private void createProgressBar() {
        // Create progress bar
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(false);
        progressBar.setBackground(Color.LIGHT_GRAY);
        progressBar.setVisible(true);
        progressBar.setForeground(new Color(46, 204, 113)); // Modern green color

        // Create progress label
        progressInfoLabel = new JLabel("Example text");
        progressInfoLabel.setVisible(false);
        progressInfoLabel.setForeground(Color.black);
        progressInfoLabel.setFont(new Font("SansSerif", Font.ITALIC, 14));

	}
	
	protected void setDeterminatePorgress(int progress) {
		if (this.progressBar.isIndeterminate())
			this.progressBar.setIndeterminate(false);
		this.progressBar.setValue(progress);
		progressBar.setStringPainted(true);
		progressBar.setString(progress + "%");
	}
	
	protected void setProgressIndeterminate(boolean indeterminate) {
		this.progressBar.setIndeterminate(indeterminate);
	}
	
	protected void setProgressBarText(String text) {
		this.progressBar.setString(text);
	}
	
	protected int getProgress() {
		return this.progressBar.getValue();
	}
	
	protected String getProgressBarText() {
		return this.progressBar.getString();
	}
	
	protected void setProgressLabelText(String text) {
		text = text == null ? "" : text;
		this.progressInfoLabel.setText(text);
		if (!this.progressInfoLabel.isVisible())
			this.progressInfoLabel.setVisible(true);
	}

	protected void update(ModelDescriptor modelDescriptor, URL path, int logoWidth, int logoHeight) {
    	DefaultIcon.drawImOrLogo(path, defaultLogoURL, exampleImageLabel, ModelSelectionPanelGui.MAIN_CARD_ID);
    	TextLoadCallback callback = new TextLoadCallback() {
    	    @Override
    	    public void onTextLoaded(String infoText) {
                setInfo(infoText);
    	        revalidate();
    	        repaint();
    	    }
    	};
        ModelInfoWorker worker = new ModelInfoWorker(modelDescriptor, callback);
        worker.execute();
	}
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // 1) Create the frame
            JFrame frame = new JFrame("Content Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(300, 400);  // or whatever size you need
            frame.setLocationRelativeTo(null);

            // 2) Create and configure your card
            ContentPanel card = new ContentPanel();

            // 3) Add to frame (since ModelCardGui uses null layout internally,
            //    we’ll use BorderLayout here to have it fill the window)
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(card, BorderLayout.CENTER);

            // 4) Show it
            frame.setVisible(true);
        });
    }
}
