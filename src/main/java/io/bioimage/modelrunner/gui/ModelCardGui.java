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
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.OverlayLayout; // Import OverlayLayout
import javax.swing.SwingUtilities;

import io.bioimage.modelrunner.gui.workers.ImageLoaderWorker;
import io.bioimage.modelrunner.gui.workers.ImageLoaderWorker.ImageLoadCallback;

public class ModelCardGui extends JPanel {

    private static final long serialVersionUID = -5625832740571130175L;

    private JLabel nameLabel;
    private JLabel nicknameLabel;
    private JLabel unsupportedLabel; // The overlay label
    private LogoPanel logoIcon; // Panel holding image and overlay
    private boolean isUnsupported = false; // Flag to track state

    private static final String UNSUPPORTED_TEXT = "Unsupported";
    private static final Color UNSUPPORTED_BG_COLOR = Color.red;
    private static final Color UNSUPPORTED_FG_COLOR = Color.black;


    private ModelCardGui() {
    	super(null);
        this.setBackground(Color.WHITE);
        this.setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));


        this.logoIcon = new LogoPanel();

        this.unsupportedLabel = new JLabel(UNSUPPORTED_TEXT);
        this.unsupportedLabel.setForeground(UNSUPPORTED_FG_COLOR);
        this.unsupportedLabel.setBackground(UNSUPPORTED_BG_COLOR);
        this.unsupportedLabel.setOpaque(true);
        this.unsupportedLabel.setBorder(BorderFactory.createEtchedBorder());
        this.unsupportedLabel.setVisible(true);


        this.nameLabel = new JLabel(Gui.LOADING_STR, JLabel.CENTER);
        this.nameLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (16)));
        this.nicknameLabel = new JLabel(Gui.LOADING_STR, JLabel.CENTER);
        this.nicknameLabel.setFont(new Font("SansSerif", Font.ITALIC, (int) (14)));

        this.add(this.nameLabel);
        this.add(this.logoIcon);
        this.add(this.nicknameLabel);
        this.add(this.unsupportedLabel);
        
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
                

                nameLabel.setFont(nameLabel.getFont().deriveFont(Font.BOLD, 16));
                nicknameLabel.setFont(nicknameLabel.getFont().deriveFont(Font.PLAIN, 16));

                Dimension topSize = nameLabel.getPreferredSize();
                Dimension bottomSize = nicknameLabel.getPreferredSize();

                int imH = logoIcon.getImage().getHeight();
                int imW = logoIcon.getImage().getWidth();

                double newW = W - sideInset * 2;
                double newH = newW * imH / (double) imW;
                int posx = sideInset;
                int posY = (int) (H / 2 - newH / 2);
                if (imH > imW) {
                	newH = H - topInset - bottomInset - imTopInset - imBottomInset - topSize.height;
                	newW = newH * imW / (double) imH;
                    posY = imTopInset + topInset + topSize.height;
                    posx = (int) (W / 2 - newW / 2);
                }

                nameLabel  .setBounds(sideInset, topInset, topSize.width, topSize.height);
                nicknameLabel.setBounds(sideInset, H - bottomInset - bottomSize.height, bottomSize.width, bottomSize.height);
                logoIcon.setBounds(posx, posY, (int) newW, (int) newH);
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

    protected static ModelCardGui createModelCard() {
        ModelCardGui modelCardPanel = new ModelCardGui();
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
            ModelCardGui card = ModelCardGui.createModelCard();
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