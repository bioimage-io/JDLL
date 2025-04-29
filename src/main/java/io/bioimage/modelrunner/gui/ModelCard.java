package io.bioimage.modelrunner.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

public class ModelCard extends JPanel {

    private static final long serialVersionUID = -5625832740571130175L;

    private final double scale;
    
    private JLabel nameLabel;
    private JLabel nicknameLabel;
    private JLabel unsupportedLabel; // The overlay label
    protected LogoPanel logoIcon; // Panel holding image and overlay
    private boolean isUnsupported = true; // Flag to track state

    protected static final String UNSUPPORTED_TEXT = "UNSUPPORTED";
    protected static final Color UNSUPPORTED_BG_COLOR = Color.red;
    protected static final Color UNSUPPORTED_FG_COLOR = Color.black;


    protected ModelCard() {
    	this(1d);
    }


    protected ModelCard(double scale) {
    	super(null);
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
    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
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
                	newH = H - topInset - bottomInset - imTopInset - imBottomInset - topSize.height;
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
                	labelY += ((int) (newH / 2)) - (int) (labelH / 2);
                }
                unsupportedLabel.setBounds(labelX, labelY, labelW, labelH);
                unsupportedLabel.setVisible(isUnsupported);
                unsupportedLabel.setFont(unsupportedLabel.getFont().deriveFont(Font.BOLD, (float) (labelH / 3.5)));
            }
        });
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

    protected static ModelCard createModelCard() {
        ModelCard modelCardPanel = new ModelCard();
        return modelCardPanel;
    }

    protected static ModelCard createModelCard(double scale) {
        ModelCard modelCardPanel = new ModelCard(scale);
        return modelCardPanel;
    }

    protected void updateCard(String name, String nickname, URL imagePath) {
        this.nameLabel.setText(name);
        this.nicknameLabel.setText(nickname);
        
        DefaultIcon.drawImOrLogo(imagePath, imagePath, logoIcon);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // 1) Create the frame
            JFrame frame = new JFrame("Model Card Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(300, 400);  // or whatever size you need
            frame.setLocationRelativeTo(null);

            // 2) Create and configure your card
            ModelCard card = ModelCard.createModelCard();
            try {
				card.updateCard(
				    "My Model Name",
				    "Friendly Nickname",
				    // try to load an image from resources; fallback to null if missing
				    new File("/home/carlos/git/deep-icy/src/main/resources/deepicy_imgs/icy_logo.png").toURL()
				);
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            // optionally mark unsupported to see the overlay:
            card.setUnsupported(true);

            // 3) Add to frame (since ModelCardGui uses null layout internally,
            //    we’ll use BorderLayout here to have it fill the window)
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(card, BorderLayout.CENTER);

            // 4) Show it
            frame.setVisible(true);
        });
    }
}