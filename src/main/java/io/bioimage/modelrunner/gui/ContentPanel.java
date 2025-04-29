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
		

        this.unsupportedLabel = new JLabel(ModelCard.UNSUPPORTED_TEXT);
        this.unsupportedLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (16)));
        this.unsupportedLabel.setForeground(ModelCard.UNSUPPORTED_FG_COLOR);
        this.unsupportedLabel.setBackground(ModelCard.UNSUPPORTED_BG_COLOR);
        this.unsupportedLabel.setOpaque(true);
        this.unsupportedLabel.setBorder(BorderFactory.createEtchedBorder());
        this.unsupportedLabel.setVisible(false);

        exampleTitleLabel = new JLabel("Cover Image");
        exampleTitleLabel.setFont(new Font("SansSerif", Font.BOLD, 18));

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

        add(exampleImageLabel);
        add(exampleTitleLabel);
        add(infoTitleLabel);
        add(modelInfoArea);
        add(progressBar);
        add(progressInfoLabel);
        add(infoScrollPane);
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
        
        
        int imH = exampleImageLabel.getPreferredSize().height;
        int imW = exampleImageLabel.getPreferredSize().width;
        
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
        
        int labelX = posx;
        int labelW = (int) newW;
        if (newW > 5) {
        	labelX = posx + 2;
        	labelW = labelW - 4;
        }
        int labelY = posY;
        int labelH = (int) newH;
        if (newH > 15) {
        	labelH = (int) (newH / 3);
        	labelY += ((int) newH / 2) - ((int) labelH);
        }
        unsupportedLabel.setBounds(labelX, labelY, labelW, labelH);
        unsupportedLabel.setVisible(isUnsupported);
    }
    
    protected void setUnsupported(boolean isUnsupported) {
    	this.isUnsupported = isUnsupported;
    }
	
	protected void setIcon(BufferedImage im) {
		this.exampleImageLabel.setImage(im, false);
	}
	
	protected void setInfo(String text) {
		this.modelInfoArea.setText(text);
		modelInfoArea.setCaretPosition(0);
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
        progressInfoLabel.setForeground(Color.black);
        progressInfoLabel.setFont(new Font("SansSerif", Font.ITALIC, 14));

	}
	
	protected void setDeterminatePorgress(int progress) {
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
	}

	protected void update(ModelDescriptor modelDescriptor, URL path, int logoWidth, int logoHeight) {
    	DefaultIcon.drawImOrLogo(path, defaultLogoURL, exampleImageLabel);
    	TextLoadCallback callback = new TextLoadCallback() {
    	    @Override
    	    public void onTextLoaded(String infoText) {
    	        /**
    	         * TODO
    	    	if (!ModelSelectionPanel.ICONS_DISPLAYED.get("main").equals(path)) {
    	            return;
    	        }
    	         */
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
