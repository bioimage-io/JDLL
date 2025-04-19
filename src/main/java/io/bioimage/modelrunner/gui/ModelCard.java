package io.bioimage.modelrunner.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.net.URL;

import javax.swing.BorderFactory;
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.OverlayLayout; // Import OverlayLayout

import io.bioimage.modelrunner.gui.workers.ImageLoaderWorker;
import io.bioimage.modelrunner.gui.workers.ImageLoaderWorker.ImageLoadCallback;

public class ModelCard extends JPanel {

    private static final long serialVersionUID = -5625832740571130175L;

    private JLabel nameLabel;
    private JLabel imageLabel; // The actual image label
    private JLabel nicknameLabel;
    private JLabel unsupportedLabel; // The overlay label
    private JPanel imagePanel; // Panel holding image and overlay
    private boolean isUnsupported = false; // Flag to track state

    private long cardWidth;
    private long cardHeight;
    private String id;
    private ImageLoaderWorker worker;
    private final double scale;

    private static double CARD_ICON_VRATIO = 0.8;
    private static double CARD_ICON_HRATIO = 0.9;
    private static final String UNSUPPORTED_TEXT = "Unsupported";
    private static final Color UNSUPPORTED_BG_COLOR = Color.red;
    private static final Color UNSUPPORTED_FG_COLOR = Color.black;


    private ModelCard(long cardWidth, long cardHeight, double scale) {
        super(new BorderLayout());
        this.scale = scale;
        this.cardWidth = cardWidth;
        this.cardHeight = cardHeight;
        this.setPreferredSize(new Dimension((int) (cardWidth * scale), (int) (cardHeight * scale)));
        this.setBackground(Color.WHITE);
        this.setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));

        int iconW = (int) (CARD_ICON_HRATIO * this.cardWidth * scale);
        int iconH = (int) (this.cardHeight * CARD_ICON_VRATIO * scale);

        Icon logoIcon = createEmptyIcon(iconW, iconH);
        this.imageLabel = new JLabel(logoIcon, JLabel.CENTER);

        this.unsupportedLabel = new JLabel(UNSUPPORTED_TEXT);
        this.unsupportedLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (24 * scale)));
        this.unsupportedLabel.setForeground(UNSUPPORTED_FG_COLOR);
        this.unsupportedLabel.setBackground(UNSUPPORTED_BG_COLOR);
        this.unsupportedLabel.setOpaque(true); // Necessary for background color to show
        this.unsupportedLabel.setBorder(BorderFactory.createEtchedBorder());
        this.unsupportedLabel.setVisible(true); // Initially hidden

        this.imagePanel = new JPanel();
        imagePanel.setBorder(BorderFactory.createEtchedBorder());
        this.imagePanel.setLayout(new OverlayLayout(imagePanel));
        this.imagePanel.setOpaque(false); // Make panel transparent

        this.unsupportedLabel.setAlignmentX(JLabel.CENTER_ALIGNMENT);
        this.unsupportedLabel.setAlignmentY(JLabel.CENTER_ALIGNMENT);
        this.imagePanel.add(this.unsupportedLabel); // Align left
        this.imageLabel.setAlignmentX(JLabel.CENTER_ALIGNMENT);
        this.imageLabel.setAlignmentY(JLabel.CENTER_ALIGNMENT);
        this.imagePanel.add(this.imageLabel);

        this.nameLabel = new JLabel(Gui.LOADING_STR, JLabel.CENTER);
        this.nameLabel.setFont(new Font("SansSerif", Font.BOLD, (int) (16 * scale)));
        this.nicknameLabel = new JLabel(Gui.LOADING_STR, JLabel.CENTER);
        this.nicknameLabel.setFont(new Font("SansSerif", Font.ITALIC, (int) (14 * scale)));

        this.add(this.nameLabel, BorderLayout.NORTH);
        // Add the imagePanel (containing image and overlay) instead of just imageLabel
        this.add(this.imagePanel, BorderLayout.CENTER);
        this.add(this.nicknameLabel, BorderLayout.SOUTH);
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

    /**
     * Set an optional id
     * @param id
     *  the identifier of the card
     */
    public void setOptionalID(String id) {
        this.id = id;
    }

    protected static ModelCard createModelCard(long cardWidth, long cardHeight, double scale) {
        ModelCard modelCardPanel = new ModelCard(cardWidth, cardHeight, scale);
        return modelCardPanel;
    }

    protected void updateCard(String name, String nickname, URL imagePath) {
        this.nameLabel.setText(name);
        this.nicknameLabel.setText(nickname);
        int iconW = (int) (CARD_ICON_HRATIO * this.cardWidth * scale);
        int iconH = (int) (this.cardHeight * CARD_ICON_VRATIO * scale);

        // Set loading icon on the main imageLabel
        DefaultIcon.getLoadingIconWithCallback(iconW, iconH, icon -> {
            imageLabel.setIcon(icon);
            // No need to revalidate/repaint here if loading icon has same size
            // as the final image icon. The callback below will handle it.
        });

        if (worker != null && !worker.isDone())
            worker.cancelBackground();

        ImageLoadCallback callback = new ImageLoadCallback() {
            @Override
            public void onImageLoaded(ImageIcon icon) {
                // Check if this image is still the one expected for this card ID
                if (ModelSelectionPanel.ICONS_DISPLAYED.get(id) != imagePath)
                    return;
                // Update the icon on the main imageLabel
                imageLabel.setIcon(icon);
                // Revalidate and repaint the container panel to ensure layout updates
                imagePanel.revalidate();
                imagePanel.repaint();
                // Optionally revalidate/repaint the whole card if necessary
                // ModelCard.this.revalidate();
                // ModelCard.this.repaint();
            }
        };
        worker = ImageLoaderWorker.create(imagePath, iconW, iconH, callback);
        worker.execute();
    }

    private static ImageIcon createEmptyIcon(int width, int height) {
        // Create a transparent BufferedImage of the specified size
        BufferedImage emptyImage = new BufferedImage(Math.max(1, width), Math.max(1, height), BufferedImage.TYPE_INT_ARGB); // Ensure non-zero dimensions
        // Create an ImageIcon from the empty image
        return new ImageIcon(emptyImage);
    }
}