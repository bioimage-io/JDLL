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
package io.bioimage.modelrunner.gui.custom;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.border.EmptyBorder;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.EnvironmentInstaller;
import io.bioimage.modelrunner.gui.workers.InstallEnvWorker;
import io.bioimage.modelrunner.model.special.stardist.Stardist2D;
import io.bioimage.modelrunner.model.special.stardist.StardistAbstract;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.view.Views;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

public class StardistGUI extends JPanel implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
    private final ConsumerInterface consumer;
    private String whichLoaded;
    private StardistAbstract model;
    private String inputTitle;
    
    private Runnable cancelCallback;
    Thread workerThread;
    
	private JComboBox<String> modelComboBox;
	private JLabel customLabel;
    private JTextField customModelPathField;
    private JButton browseButton;
    private JSpinner minPercField;
    private JSpinner maxPercField;
    private JProgressBar bar;
    private JButton cancelButton, installButton, runButton;
    
    private final String CUSTOM_STR = "your custom model";
    private static List<String> VAR_NAMES = Arrays.asList(new String[] {
    		"Select a model:", "Custom Model Path:", "Normalization low percentile:", "Normalization low percentile:"
    });

    public StardistGUI(ConsumerInterface consumer) {
        // Set a modern-looking border layout with padding
    	this.consumer = consumer;
    	List<JComponent> componentList = new ArrayList<JComponent>();
        setLayout(new BorderLayout());
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));

        // --- Model Selection Panel ---
        JPanel modelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelPanel.add(new JLabel(VAR_NAMES.get(0)));
        String[] models = {"StarDist Fluorescence Nuclei Segmentation", "StarDist H&E Nuclei Segmentation", CUSTOM_STR};
        modelComboBox = new JComboBox<String>(models);
        modelPanel.add(modelComboBox);

        // Panel for custom model file path
        JPanel customModelPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        customLabel = new JLabel(VAR_NAMES.get(1));
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
        parametersPanel.add(new JLabel(VAR_NAMES.get(2)));
        SpinnerNumberModel modelL = new SpinnerNumberModel(1., 0., 100., 0.01);
        minPercField= new JSpinner(modelL);
        parametersPanel.add(minPercField);
        // Channel selection
        parametersPanel.add(new JLabel(VAR_NAMES.get(3)));
        SpinnerNumberModel modelH = new SpinnerNumberModel(99.8, 0., 100., 0.01);
        maxPercField= new JSpinner(modelH);
        parametersPanel.add(maxPercField);

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
		bar.setStringPainted(true);
		bar.setString("");
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

        this.consumer.setVariableNames(VAR_NAMES);
        componentList.add(this.modelComboBox);
        componentList.add(this.customModelPathField);
        componentList.add(this.minPercField);
        componentList.add(this.maxPercField);
        this.consumer.setComponents(componentList);
        this.installButton.addActionListener(this);
        this.runButton.addActionListener(this);
        this.cancelButton.addActionListener(this);

        // Enable when custom selected
        modelComboBox.addPopupMenuListener(new PopupMenuListener() {
            @Override
            public void popupMenuWillBecomeVisible(PopupMenuEvent e) {}
            @Override
            public void popupMenuCanceled(PopupMenuEvent e) {}

            @Override
            public void popupMenuWillBecomeInvisible(PopupMenuEvent e) {
            	boolean enabled = modelComboBox.getSelectedItem().equals(CUSTOM_STR);
                customLabel.setEnabled(enabled);
                customModelPathField.setEnabled(enabled);
                browseButton.setEnabled(enabled);
            }

        });

        // You can add additional listeners for the Cancel, Install, and Run buttons here.
    }
    
    public void setCancelCallback(Runnable cancelCallback) {
    	this.cancelCallback = cancelCallback;
    }
    
    public void close() {
    	if (model != null && model.isLoaded())
    		model.close();
    }

    // For demonstration purposes: a main method to show the UI in a JFrame.
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("StarDist Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new StardistGUI(null));
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
    		workerThread = new Thread(() -> {
        		try {
    				runStardist();
    				startModelInstallation(false);
    			} catch (IOException | RunModelException | LoadModelException e1) {
    				e1.printStackTrace();
    				startModelInstallation(false);
    				SwingUtilities.invokeLater(() -> this.bar.setString("Error running the model"));
    			}
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.installButton) {
    		workerThread = new Thread(() -> {
        		installStardist();
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.cancelButton) {
    		cancel();
    	}
    }
    
    private void cancel() {
    	if (workerThread != null && workerThread.isAlive())
    		workerThread.interrupt();
    	if (model != null)
    		model.close();
    	if (cancelCallback != null)
    		cancelCallback.run();
    }
    
    private < T extends RealType< T > & NativeType< T > > void runStardist() throws IOException, RunModelException, LoadModelException {
    	startModelInstallation(true);
    	installStardist(weightsInstalled(), StardistAbstract.isInstalled());
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	this.inputTitle = consumer.getFocusedImageName();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	SwingUtilities.invokeLater(() ->{
    		this.bar.setIndeterminate(true);
    		this.bar.setString("Loading model");
    	});
    	String selectedModel = (String) this.modelComboBox.getSelectedItem();
    	String modelype = "" + selectedModel;
    	if (modelype.equals(CUSTOM_STR))
    		selectedModel = customModelPathField.getText();
    	
    	if (modelype.equals(CUSTOM_STR) 
    			&& (whichLoaded == null || model == null || model.isClosed() || !whichLoaded.equals(selectedModel)))
    		model = StardistAbstract.init(selectedModel);
    	else if (!modelype.equals(CUSTOM_STR) 
    			&& (whichLoaded == null || model == null || model.isClosed() || !whichLoaded.equals(selectedModel))) {
			try {
				model = Stardist2D.fromPretained(selectedModel, consumer.getModelsDir(), false);
			} catch (InterruptedException e) {
				e.printStackTrace();
				return;
			}
    	} else if (model == null)
    		throw new IllegalArgumentException();
    	if (!model.isLoaded())
    		model.loadModel();
    	
    	whichLoaded = selectedModel;
    	SwingUtilities.invokeLater(() ->{
    		this.bar.setString("Running the model");
    	});
    	if (rai.dimensionsAsLongArray().length == 4) {
    		runStardistOnFramesStack(rai);
    	} else {
    		runStardistOnTensor(rai);
    	}
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runStardistOnFramesStack(RandomAccessibleInterval<R> rai) throws RunModelException {
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
    	consumer.display(outMaskRai, "xyb", "mask");
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runStardistOnTensor(RandomAccessibleInterval<R> rai) throws RunModelException {
		Tensor<R> tensor = Tensor.build("input", "xyc", rai);
    	List<Tensor<R>> inList = new ArrayList<Tensor<R>>();
		inList.add(tensor);
    	List<Tensor<T>> out = model.run(inList);
    	for (Tensor<T> tt : out) {
    		if (tt.getAxesOrder().length == 1)
    			continue;
        	consumer.display(tt.getData(), tt.getAxesOrderString(), getOutputName(tt.getName()));
    	}
    }
    
    private String getOutputName(String tensorName) {
    	int index = inputTitle.lastIndexOf(".");
    	index = index == -1 ? inputTitle.length() : index;
    	String noExtension = inputTitle.substring(0, index);
    	String extension = ".tif";
    	return noExtension + "_" + tensorName + extension;
    }
    
    private void installStardist() {
    	startModelInstallation(true);
    	boolean envInstalled = StardistAbstract.isInstalled();
    	boolean wwInstalled = weightsInstalled();
    	if (envInstalled && wwInstalled) {
        	startModelInstallation(false);
    		return;
    	}
    	installStardist(wwInstalled, envInstalled);
    }
    
    private void installStardist(boolean wwInstalled, boolean envInstalled) {
    	if (wwInstalled && envInstalled)
    		return;
    	SwingUtilities.invokeLater(() -> this.bar.setString("Installing..."));
    	CountDownLatch latch = !wwInstalled && !envInstalled ? new CountDownLatch(2) : new CountDownLatch(1);
    	if (!wwInstalled)
    		installModelWeights(latch);
    	if (!envInstalled)
    		installEnv(latch);
    	try {
			latch.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
    }
    
    private boolean weightsInstalled() {
    	String model = (String) this.modelComboBox.getSelectedItem();
    	if (model.equals(CUSTOM_STR)) {
    		return true;
    	}
    	try {
			Stardist2D pretrained = Stardist2D.fromPretained(model, consumer.getModelsDir(), false);
			if (pretrained == null)
				return false;
		} catch (Exception e) {
			return false;
		}
    	return true;
    }
    
    private void installModelWeights(CountDownLatch latch) {
    	Consumer<Double> cons = (d) -> {
    		double perc = Math.round(d * 1000) / 10.0d;
    		SwingUtilities.invokeLater(() -> {
        		this.bar.setValue((int) Math.floor(perc));
        		this.bar.setString(perc + "% of weights");
    		});
    	};
		SwingUtilities.invokeLater(() -> bar.setIndeterminate(false));
		Thread dwnlThread = new Thread(() -> {
			try {
				Stardist2D.donwloadPretrained((String) modelComboBox.getSelectedItem(), this.consumer.getModelsDir(), cons);
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}
			latch.countDown();
			checkModelInstallationFinished(latch);
		});
		dwnlThread.start();
    }
    
    private void installEnv(CountDownLatch latch) {
    	String msg = "Installation of Python environments might take up to 20 minutes.";
    	String question = "Install Python for StarDist";
    	if (StardistAbstract.isInstalled() || 
    			JOptionPane.showConfirmDialog(null, msg, question, JOptionPane.YES_NO_OPTION) != JOptionPane.YES_OPTION) {
			latch.countDown();
			checkModelInstallationFinished(latch);
    		return;
    	}
		JDialog installerFrame = new JDialog();
		installerFrame.setTitle("Installing StarDist");
		installerFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
    	Consumer<Boolean> callback = (bool) -> {
    		checkModelInstallationFinished(latch);
    		if (installerFrame.isVisible())
    			installerFrame.dispose();
    	};
    	InstallEnvWorker worker = new InstallEnvWorker("StarDist", latch, callback);
		EnvironmentInstaller installerPanel = EnvironmentInstaller.create(worker);
		Consumer<String> cons = (s) ->{
			installerPanel.updateText(s, Color.black);
			if (latch.getCount() != 1)
				return;
			SwingUtilities.invokeLater(() ->{
				if (!bar.isIndeterminate() || (bar.isIndeterminate() && !bar.getString().equals("Installing Python"))) {
					bar.setIndeterminate(true);
					bar.setString("Installing Python");
				}
			});
		};
		worker.setConsumer(cons);
    	worker.execute();
		installerPanel.addToFrame(installerFrame);
    	installerFrame.setSize(600, 300);
    }
    
    private void checkModelInstallationFinished(CountDownLatch latch) {
    	if (latch.getCount() == 0)
    		startModelInstallation(false);
    }
    
    private void startModelInstallation(boolean isStarting) {
    	SwingUtilities.invokeLater(() -> {
        	this.runButton.setEnabled(!isStarting);
        	this.installButton.setEnabled(!isStarting);
        	this.modelComboBox.setEnabled(!isStarting);
        	this.minPercField.setEnabled(!isStarting);
        	this.maxPercField.setEnabled(!isStarting);
        	if (isStarting) {
        		this.bar.setString("Checking stardist installed...");
        		this.bar.setIndeterminate(true);
        	} else {
        		this.bar.setIndeterminate(false);
		    	this.bar.setValue(0);
		    	this.bar.setString("");
        	}
    	});
    }
    
    private void browseFiles() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int option = fileChooser.showOpenDialog(StardistGUI.this);
        if (option == JFileChooser.APPROVE_OPTION) {
            customModelPathField.setText(fileChooser.getSelectedFile().getAbsolutePath());
        }
    }
}
