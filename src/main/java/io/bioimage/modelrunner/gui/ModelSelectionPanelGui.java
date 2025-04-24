package io.bioimage.modelrunner.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;
import javax.swing.border.TitledBorder;

public class ModelSelectionPanelGui extends JPanel {

	private static final long serialVersionUID = 6264134076603842497L;

	protected ModelCard prevModelPanel;
    protected ModelCard selectedModelPanel;
    protected ModelCard nextModelPanel;
    
    protected JButton nextButton;
    protected JButton prevButton;
    protected TitledBorder lineBorder;
    
    protected String defaultString = Gui.LOADING_STR;

    protected static final double MAIN_CARD_RT = 1;
    protected static final double SECOND_CARD_RT = 0.6;

    protected static final double BTN_HEIGHT_RATIO = 0.07;
    protected static final double MAX_BTN_HEIGHT = 33;


	protected ModelSelectionPanelGui() {
        super(null);
        this.setBackground(new Color(236, 240, 241));
        lineBorder = BorderFactory.createTitledBorder(BorderFactory.createLineBorder(Color.gray, 2, true), 
        		Gui.LOCAL_STR);
        Border paddingBorder = BorderFactory.createEmptyBorder(2, 2, 2, 2);
        this.setBorder(BorderFactory.createCompoundBorder(paddingBorder,lineBorder));

        prevModelPanel = ModelCard.createModelCard(SECOND_CARD_RT);
        selectedModelPanel = ModelCard.createModelCard(MAIN_CARD_RT);
        nextModelPanel = ModelCard.createModelCard(SECOND_CARD_RT);


        prevButton = new JButton("◀");
        prevButton.setFont(new Font("SansSerif", Font.BOLD, 10));

        nextButton = new JButton("▶");
        nextButton.setFont(new Font("SansSerif", Font.BOLD, 10));
        
        add(prevModelPanel);
        add(selectedModelPanel);
        add(nextModelPanel);
        add(prevButton);
        add(nextButton);
        
        organiseComponents();
	}
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                Insets in = getInsets();
                int W = getWidth()  - in.left - in.right;
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = in.left;
                int insetTop = in.top;
                
                double btnH = Math.min(MAX_BTN_HEIGHT, rawH * BTN_HEIGHT_RATIO);
                prevButton.setBounds(in.left, rawH - in.left - (int) btnH, (int) (W * 0.5), (int) btnH);
                nextButton.setBounds(in.left + (int) (W * 0.5), rawH - in.left - (int) btnH, (int) (W * 0.5), (int) btnH);
                
                int hSide = (int) (rawH - inset - insetTop - btnH);
                int wSide = (int) ((W - 4 * inset) / (SECOND_CARD_RT * 2 + 1));
                
                int side = Math.min(wSide, hSide);

                int mediumX = (int) (rawW / 2 - side / 2);
                int leftX = (int) (mediumX - inset - side * SECOND_CARD_RT);
                int rightX = (int) (mediumX + side + inset);
                
                int topY = (int) ((rawH - btnH) / 2 - side / 2);
                int bottomY = (int) (topY + side * (1 - SECOND_CARD_RT) / 2);
                
                prevModelPanel.setBounds(leftX, bottomY, (int) (side * SECOND_CARD_RT), (int) (side * SECOND_CARD_RT));
                selectedModelPanel.setBounds(mediumX, topY, side, side);
                nextModelPanel.setBounds(rightX, bottomY, (int) (side * SECOND_CARD_RT), (int) (side * SECOND_CARD_RT));
            }
        });
    }
    
    protected void setBorderLabel(String text) {
    	lineBorder.setTitle(text);
    	this.validate();
    	this.repaint();
    }
    
    protected void setArrowsEnabled(boolean enabled) {
    	nextButton.setEnabled(enabled);
    	prevButton.setEnabled(enabled);
    }
    
    protected void setLocalBorder() {
    	setBorderLabel(Gui.LOCAL_STR);
    }

    
    protected void setBMZBorder() {
    	setBorderLabel(Gui.BIOIMAGEIO_STR);
    }
    
    protected void setLoading() {
    	defaultString = Gui.LOADING_STR;
    }
    
    protected void setNotFound() {
    	defaultString = Gui.NOT_FOUND_STR;
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // 1) Create the frame
            JFrame frame = new JFrame("CArrousel Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(300, 400);  // or whatever size you need
            frame.setLocationRelativeTo(null);

            // 2) Create and configure your card
            ModelSelectionPanelGui card = new ModelSelectionPanelGui();

            // 3) Add to frame (since ModelCardGui uses null layout internally,
            //    we’ll use BorderLayout here to have it fill the window)
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(card, BorderLayout.CENTER);

            // 4) Show it
            frame.setVisible(true);
        });
    }
}
