package io.bioimage.modelrunner.gui;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.view.Views;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CellposeGUI extends JPanel implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
    private final ConsumerInterface consumer;
    private Cellpose model;
    
	private JComboBox<String> modelComboBox;
	private JLabel customLabel;
    private JTextField customModelPathField;
    private JButton browseButton;
    private JTextField diameterField;
    private JComboBox<String> channelComboBox;
    private JCheckBox check;
    private JProgressBar bar;
    private JButton cancelButton, installButton, runButton;
    
    private final String CUSOTM_STR = "your custom model";

    public CellposeGUI(ConsumerInterface consumer) {
        // Set a modern-looking border layout with padding
    	this.consumer = consumer;
        setLayout(new BorderLayout());
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));

        // --- Model Selection Panel ---
        JPanel modelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelPanel.add(new JLabel("Select a model:"));
        String[] models = {"cyto3", "cyto2", "cyto", "nuclei", CUSOTM_STR};
        modelComboBox = new JComboBox<String>(models);
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
        JPanel parametersPanel = new JPanel(new GridLayout(3, 2, 10, 10));
        parametersPanel.setBorder(BorderFactory.createTitledBorder("Optional Parameters"));
        // Diameter input
        parametersPanel.add(new JLabel("Diameter:"));
        diameterField = new JTextField();
        parametersPanel.add(diameterField);
        // Channel selection
        parametersPanel.add(new JLabel("Channel:"));
        String[] channels;
        if (consumer.getFocusedImageChannels() != null && consumer.getFocusedImageChannels() == 1)
        	channels = new String[] {"[0,0]"};
        else if (consumer.getFocusedImageChannels() != null && consumer.getFocusedImageChannels() == 3)
        	channels = new String[] {"[2,3]", "[2,1]"};
        else
        	channels = new String[] {"[0,0]", "[2,3]", "[2,1]"};
        channelComboBox = new JComboBox<String>(channels);
        parametersPanel.add(channelComboBox);
        check = new JCheckBox("Display all outputs");
        check.setSelected(false);
        parametersPanel.add(check);

        // --- Buttons Panel ---
        JPanel footerPanel = new JPanel(new GridLayout(1, 2));
        footerPanel.setBorder(BorderFactory.createEtchedBorder());
        JPanel progressPanel = new JPanel(new BorderLayout());
        progressPanel.setBorder(BorderFactory.createEmptyBorder(10, 5, 10, 5));
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttonPanel.setBorder(BorderFactory.createEtchedBorder());
        cancelButton = new JButton("Cancel");
        installButton = new JButton("Install");
        runButton = new JButton("Run");
        buttonPanel.add(cancelButton);
        buttonPanel.add(installButton);
        buttonPanel.add(runButton);
        
        bar = new JProgressBar();
        progressPanel.add(bar, BorderLayout.CENTER);
        footerPanel.add(progressPanel);
        footerPanel.add(buttonPanel);

        // Add components to main panel
        mainPanel.add(modelPanel);
        mainPanel.add(customModelPanel);
        mainPanel.add(Box.createVerticalStrut(10)); // spacing
        mainPanel.add(parametersPanel);
        mainPanel.add(Box.createVerticalStrut(10)); // spacing
        mainPanel.add(footerPanel);
        mainPanel.setBorder(BorderFactory.createEmptyBorder());

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
    
    public void close() {
    	if (model != null && model.isLoaded())
    		model.close();
    }

    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("Cellpose Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new CellposeGUI(null));
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
    		try {
				runCellpose();
			} catch (IOException | RunModelException e1) {
				e1.printStackTrace();
			}
    	} else if (e.getSource() == this.installButton) {
    		installCellpose();
    	}
    }
    
    private < T extends RealType< T > & NativeType< T > > void runCellpose() throws IOException, RunModelException {
    	installCellpose();
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	String modelPath = (String) this.modelComboBox.getSelectedItem();
    	if (modelPath.equals(CUSOTM_STR))
    		modelPath = this.customModelPathField.getText();
    	if (model == null || !model.isLoaded())
    		model = Cellpose.init(modelPath);
    	if (rai.dimensionsAsLongArray().length == 4) {
    		runCellposeOnFramesStack(rai);
    	} else {
    		runCellposeOnTensor(rai);
    	}
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runCellposeOnFramesStack(RandomAccessibleInterval<R> rai) throws RunModelException {
    	long[] dims = rai.dimensionsAsLongArray();
		RandomAccessibleInterval<T> outMaskRai = Cast.unchecked(ArrayImgs.floats(new long[] {dims[0], dims[1], dims[3]}));
		for (int i = 0; i < rai.dimensionsAsLongArray()[3]; i ++) {
	    	List<Tensor<R>> inList = new ArrayList<Tensor<R>>();
	    	Tensor<R> inIm = Tensor.build("input", "xyc", Views.hyperSlice(rai, 3, i));
	    	inList.add(inIm);
	    	
	    	List<Tensor<T>> outputList = new ArrayList<Tensor<T>>();
	    	Tensor<T> outMask = Tensor.build("mask", "xy", Views.hyperSlice(outMaskRai, 2, i));
	    	outputList.add(outMask);
	    	
	    	model.run(inList, outputList);
		}
    	consumer.display(outMaskRai, "xyt", "mask");
    	if (!check.isSelected())
    		return;
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runCellposeOnTensor(RandomAccessibleInterval<R> rai) throws RunModelException {
		Tensor<R> tensor = Tensor.build("input", "xyc", rai);
    	List<Tensor<R>> inList = new ArrayList<Tensor<R>>();
		inList.add(tensor);
    	List<Tensor<T>> out = model.run(inList);
    	consumer.display(out.get(0).getData(), out.get(0).getAxesOrderString(), out.get(0).getName());
    	if (!check.isSelected())
    		return;
    }
    
    private void installCellpose() {
    	if (Cellpose.isInstalled())
    		return;
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
