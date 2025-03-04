package io.bioimage.modelrunner.gui;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class CellposeGUI extends JPanel implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
	private JComboBox<String> modelComboBox;
	private JLabel customLabel;
    private JTextField customModelPathField;
    private JButton browseButton;
    private JTextField diameterField;
    private JComboBox<String> channelComboBox;
    private JButton cancelButton, installButton, runButton;
    
    private final String CUSOTM_STR = "your custom model";

    public CellposeGUI() {
    	this(null);
    }

    public CellposeGUI(Integer nChannels) {
        // Set a modern-looking border layout with padding
        setLayout(new BorderLayout());
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));

        // --- Model Selection Panel ---
        JPanel modelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelPanel.add(new JLabel("Select a model:"));
        String[] models = {"cyto3", "cyto2", "cyto", "nuclei", CUSOTM_STR};
        modelComboBox = new JComboBox<>(models);
        modelPanel.add(modelComboBox);

        // Panel for custom model file path
        JPanel customModelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        customLabel = new JLabel("Custom Model Path:");
        customLabel.setEnabled(false);
        customModelPanel.add(customLabel);
        customModelPathField = new JTextField(20);
        customModelPathField.setEnabled(false);
        customModelPanel.add(customModelPathField);
        browseButton = new JButton("Browse");
        browseButton.setEnabled(false);
        customModelPanel.add(browseButton);

        // --- Optional Parameters Panel ---
        JPanel parametersPanel = new JPanel(new GridLayout(2, 2, 10, 10));
        parametersPanel.setBorder(BorderFactory.createTitledBorder("Optional Parameters"));
        // Diameter input
        parametersPanel.add(new JLabel("Diameter:"));
        diameterField = new JTextField();
        parametersPanel.add(diameterField);
        // Channel selection
        parametersPanel.add(new JLabel("Channel:"));
        String[] channels;
        if (nChannels != null && nChannels == 1)
        	channels = new String[] {"[0,0]"};
        else if (nChannels != null && nChannels == 3)
        	channels = new String[] {"[2,3]", "[2,1]"};
        else
        	channels = new String[] {"[0,0]", "[2,3]", "[2,1]"};
        channelComboBox = new JComboBox<>(channels);
        parametersPanel.add(channelComboBox);

        // --- Buttons Panel ---
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        cancelButton = new JButton("Cancel");
        installButton = new JButton("Install");
        runButton = new JButton("Run");
        buttonPanel.add(cancelButton);
        buttonPanel.add(installButton);
        buttonPanel.add(runButton);

        // Add components to main panel
        mainPanel.add(modelPanel);
        mainPanel.add(customModelPanel);
        mainPanel.add(Box.createVerticalStrut(10)); // spacing
        mainPanel.add(parametersPanel);
        mainPanel.add(Box.createVerticalStrut(10)); // spacing
        mainPanel.add(buttonPanel);

        // Add main panel to the current panel
        add(mainPanel, BorderLayout.CENTER);

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

        // You can add additional listeners for the Cancel, Install, and Run buttons here.
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
            }
        });
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
    	if (e.getSource() == browseButton) {
    		browseFiles();
    	} else if (e.getSource() == this.runButton) {
    		runCellpose();
    	} else if (e.getSource() == this.installButton) {
    		installCellpose();
    	}
    }
    
    private void runCellpose() {
    	
    }
    
    private void installCellpose() {
    	
    }
    
    private void browseFiles() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int option = fileChooser.showOpenDialog(CellposeGUI.this);
        if (option == JFileChooser.APPROVE_OPTION) {
            customModelPathField.setText(fileChooser.getSelectedFile().getAbsolutePath());
        }
    }
}
