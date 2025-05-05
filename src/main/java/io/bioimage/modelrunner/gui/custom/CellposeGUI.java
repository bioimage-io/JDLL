package io.bioimage.modelrunner.gui.custom;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import java.awt.Dimension;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.Arrays;
import java.util.List;

public class CellposeGUI extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;
    
    protected JLabel modelLabel, customLabel, cytoplasmLabel, nucleiLabel, diameterLabel;
	protected JComboBox<String> modelComboBox;
	protected JTextField customModelPathField;
    protected JButton browseButton;
    protected JTextField diameterField;
    protected JComboBox<String> cytoCbox, nucleiCbox;
    protected JCheckBox check;
    protected FooterPanel footer;
    
    protected final String CUSOTM_STR = "your custom model";
    protected static List<String> VAR_NAMES = Arrays.asList(new String[] {
    		"Select a model:", "Custom Model Path:", "Cytoplasm color", "Nuclei Color", "Diameter:", "Display all outputs"
    });

    protected static String[] RGB_LIST = new String[] {"red", "blue", "green"};
    protected static String[] GRAYSCALE_LIST = new String[] {"gray"};
    private static final Dimension MIN_D = new Dimension(20, 40);

    protected CellposeGUI() {
        setLayout(null);

        // --- Model Selection Panel ---
        modelLabel = new JLabel(VAR_NAMES.get(0));
        String[] models = {"cyto3", "cyto2", "cyto", "nuclei", CUSOTM_STR};
        modelComboBox = new JComboBox<String>(models);

        // Panel for custom model file path
        customLabel = new JLabel(VAR_NAMES.get(1));
        customLabel.setEnabled(false);
        customModelPathField = new JTextField(20);
        customModelPathField.setEnabled(false);
        browseButton = new JButton("Browse");
        browseButton.setEnabled(false);

        // Channel selection
        cytoplasmLabel = new JLabel(VAR_NAMES.get(2));
        nucleiLabel = new JLabel(VAR_NAMES.get(3));
        cytoCbox = new JComboBox<String>(RGB_LIST);
        nucleiCbox = new JComboBox<String>(RGB_LIST);
        diameterLabel = new JLabel(VAR_NAMES.get(4));
        diameterField = new JTextField();
        check = new JCheckBox(VAR_NAMES.get(5));
        check.setSelected(false);

        // --- Buttons Panel ---
        footer = new FooterPanel();
        add(footer);

		add(modelLabel);
        add(modelComboBox);
        add(customLabel);
        add(customModelPathField);
        add(browseButton);
        add(cytoplasmLabel);
        add(cytoCbox);
        add(nucleiLabel);
        add(nucleiCbox);
        add(diameterLabel);
        add(diameterField);
        add(check);
        
        this.setMinimumSize(MIN_D);
        
        organiseComponents();

        // Enable when custom selected
        modelComboBox.addPopupMenuListener(new PopupMenuListener() {
            @Override
            public void popupMenuWillBecomeVisible(PopupMenuEvent e) {}
            @Override
            public void popupMenuCanceled(PopupMenuEvent e) {}

            @Override
            public void popupMenuWillBecomeInvisible(PopupMenuEvent e) {
            	boolean enabled = modelComboBox.getSelectedItem().equals(CUSOTM_STR);
                customLabel.setEnabled(enabled);
                customModelPathField.setEnabled(enabled);
                browseButton.setEnabled(enabled);
            }

        });

    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = 5;
                int nParams = VAR_NAMES.size();
                int nRows = nParams + 1;
                int rowH = (rawH - (inset * nRows)) / nRows;
                
                int y = inset;
                int modelLabelW = (rawW - inset * 3) / 5;
                modelLabel.setBounds(inset, y, modelLabelW, rowH);
                int cboxW = rawW - inset * 3 - modelLabelW;
                modelComboBox.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                int browseButtonW = modelLabelW / 1;
                int textFieldW = rawW - inset * 4 - modelLabelW - browseButtonW;
                customLabel.setBounds(inset, y, modelLabelW, rowH);
                customModelPathField.setBounds(inset * 2 + modelLabelW, y, textFieldW, rowH);
                browseButton.setBounds(inset * 3 + modelLabelW + textFieldW, y, browseButtonW, rowH);
                y += (inset + rowH);
                cytoplasmLabel.setBounds(inset, y, modelLabelW, rowH);
                cytoCbox.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                nucleiLabel.setBounds(inset, y, modelLabelW, rowH);
                nucleiCbox.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                diameterLabel.setBounds(inset, y, modelLabelW, rowH);
                diameterField.setBounds(inset * 2 + modelLabelW, y, cboxW, rowH);
                y += (inset + rowH);
                check.setBounds(inset, y, rawW - 2 * inset, rowH);
                y += (inset + rowH);
                footer.setBounds(inset, y, rawW - 2 * inset, rowH);
            }
        });
    }
    
    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("Cellpose Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new CellposeGUI());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setSize(60, 100);
            }
        });
    }
}
