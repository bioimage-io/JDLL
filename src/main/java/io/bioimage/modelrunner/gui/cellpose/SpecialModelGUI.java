package io.bioimage.modelrunner.gui.cellpose;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class SpecialModelGUI extends JPanel {

    private static final long serialVersionUID = 5381352117710530216L;
	private JComboBox<String> modelComboBox;
    private JTextField customModelPathField;
    private JButton browseButton;
    private JTextField diameterField;
    private JComboBox<String> channelComboBox;
    private JButton cancelButton, installButton, runButton;

    public SpecialModelGUI() {
        // Set a modern-looking border layout with padding
        setLayout(new BorderLayout());
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));

        // --- Model Selection Panel ---
        JPanel modelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelPanel.add(new JLabel("Select Pretrained Model:"));
        String[] models = {"Default Model 1", "Default Model 2", "Default Model 3", "Custom Model"};
        modelComboBox = new JComboBox<>(models);
        modelPanel.add(modelComboBox);

        // Panel for custom model file path
        JPanel customModelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        customModelPanel.add(new JLabel("Custom Model Path:"));
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
        String[] channels = {"[0,0]", "[2,3]", "[2,1]"};
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

        // --- Event Listeners ---
        // Enable custom model file fields if "Custom Model" is selected
        modelComboBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                boolean isCustom = modelComboBox.getSelectedItem().toString().equals("Custom Model");
                customModelPathField.setEnabled(isCustom);
                browseButton.setEnabled(isCustom);
            }
        });

        // Browse button action to open a file chooser dialog
        browseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser();
                int option = fileChooser.showOpenDialog(SpecialModelGUI.this);
                if (option == JFileChooser.APPROVE_OPTION) {
                    customModelPathField.setText(fileChooser.getSelectedFile().getAbsolutePath());
                }
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
                frame.getContentPane().add(new SpecialModelGUI());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }
}
