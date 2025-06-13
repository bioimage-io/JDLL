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

import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;
import javax.swing.event.PopupMenuEvent;
import javax.swing.event.PopupMenuListener;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.EnvironmentInstaller;
import io.bioimage.modelrunner.gui.custom.gui.StarDistGUI;
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

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

public class StarDistPluginUI extends StarDistGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
    private final ConsumerInterface consumer;
    private String whichLoaded;
    private StardistAbstract model;
    private String inputTitle;
    
    public HashMap<String, Double> threshMap = new HashMap<String, Double>();
    
    private Runnable cancelCallback;
    Thread workerThread;
    
    private static boolean INSTALLED_WEIGHTS = false;
    
    private static boolean INSTALLED_ENV = false;

    public StarDistPluginUI(ConsumerInterface consumer) {
        // Set a modern-looking border layout with padding
    	this.consumer = consumer;
    	List<JComponent> componentList = new ArrayList<JComponent>();

        this.consumer.setVariableNames(VAR_NAMES);
        componentList.add(this.modelComboBox);
        componentList.add(this.customModelPathField);
        componentList.add(optionalParams.getMinPercField());
        componentList.add(optionalParams.getMaxPercField());
        this.consumer.setComponents(componentList);
        this.footer.getButtons().getInstallButton().addActionListener(this);
        this.footer.getButtons().getRunButton().addActionListener(this);
        this.footer.getButtons().getCancelButton().addActionListener(this);

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
                frame.getContentPane().add(new StarDistPluginUI(null));
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
    	} else if (e.getSource() == this.footer.getButtons().getRunButton()) {
    		workerThread = new Thread(() -> {
        		try {
    				runStardist();
    				startModelInstallation(false);
    			} catch (Exception e1) {
    				e1.printStackTrace();
    				startModelInstallation(false);
    				SwingUtilities.invokeLater(() -> this.footer.getBar().setString("Error running the model"));
    			}
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.footer.getButtons().getInstallButton()) {
    		workerThread = new Thread(() -> {
        		installStardist();
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.footer.getButtons().getCancelButton()) {
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
    
    private void saveParams() {
    	LinkedHashMap<String, String> map = new LinkedHashMap<String, String>();
    	String modelPath = (String) this.modelComboBox.getSelectedItem();
    	if (modelPath.equals(CUSTOM_STR))
    		map.put("model", this.customModelPathField.getText());
    	else
    		map.put("model", modelPath);
    	map.put("prob_thresh", "" + this.thresholdSlider.getSlider().getValue() / 1000d);
    	map.put("min_percentile", "" + this.optionalParams.getMinPercField().getValue());
    	map.put("max_percentile", "" + this.optionalParams.getMaxPercField().getValue());
    	this.consumer.notifyParams(map);
    }
    
    private < T extends RealType< T > & NativeType< T > > void runStardist() throws IOException, RunModelException, LoadModelException {
    	saveParams();
    	renewThreshold();
    	startModelInstallation(true);
    	if (!INSTALLED_WEIGHTS || !INSTALLED_ENV)
        	installStardist(weightsInstalled(), (INSTALLED_ENV = StardistAbstract.isInstalled()));
    	if (!INSTALLED_WEIGHTS || !INSTALLED_ENV)
    		return;
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	this.inputTitle = consumer.getFocusedImageName();
    	SwingUtilities.invokeLater(() ->{
    		this.footer.getBar().setIndeterminate(true);
    		this.footer.getBar().setString("Loading model");
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
    	model.setThreshold((double) (this.thresholdSlider.getSlider().getValue() / 1000d));
    	whichLoaded = selectedModel;
    	SwingUtilities.invokeLater(() ->{
    		this.footer.getBar().setString("Running the model");
    	});
    	runStardistOnFramesStack(rai);
    }
    
    private void renewThreshold() {
    	double val = this.thresholdSlider.getSlider().getValue() / 1000d;
    	if (modelComboBox.getSelectedItem().equals("StarDist Fluorescence Nuclei Segmentation")) {
    		this.thresh1D = val;
    	} else if (modelComboBox.getSelectedItem().equals("StarDist H&E Nuclei Segmentation")) {
    		thresh3D = val;
    	} else {
    		threshMap.put(this.customModelPathField.getText().trim(), val);
    	}
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runStardistOnFramesStack(RandomAccessibleInterval<R> rai) throws RunModelException {
    	rai = addDimsToInput(rai, model);
    	long[] inDims = rai.dimensionsAsLongArray();
    	long[] outDims;
    	if (model.is2D())
    		outDims = new long[] {inDims[0], inDims[1], 1, inDims[3]};
    	else
    		outDims = new long[] {inDims[0], inDims[1], 1, inDims[3], inDims[4]};
		RandomAccessibleInterval<T> outMaskRai = Cast.unchecked(ArrayImgs.floats(outDims));
		for (int i = 0; i < inDims[inDims.length - 1]; i ++) {
	    	List<Tensor<R>> inList = new ArrayList<Tensor<R>>();
	    	Tensor<R> inIm = Tensor.build("input", model.is2D() ? "xyc" : "xycz", Views.hyperSlice(rai, inDims.length - 1, i));
	    	inList.add(inIm);
	    	
	    	List<Tensor<T>> outputList = new ArrayList<Tensor<T>>();
	    	Tensor<T> outMask = Tensor.build("mask", model.is2D() ? "xyc" : "xycz", Views.hyperSlice(outMaskRai, outDims.length - 1, i));
	    	outputList.add(outMask);
	    	
	    	model.run(inList, outputList);
		}
    	consumer.display(outMaskRai, model.is2D() ? "xycb" : "xyczb", getOutputName("mask"));
    }
    
    private static <R extends RealType<R> & NativeType<R>>
    RandomAccessibleInterval<R> addDimsToInput(RandomAccessibleInterval<R> rai, StardistAbstract model) {
    	int nChannels = model.getNChannels();
    	boolean is2d = model.is2D();
    	long[] dims = rai.dimensionsAsLongArray();
    	if (dims.length == 2 && nChannels == 1 && is2d)
    		return Views.addDimension(Views.addDimension(rai, 0, 0), 0, 0);
    	else if (dims.length == 3 && dims[2] == nChannels && is2d)
    		return Views.addDimension(rai, 0, 0);
    	else if (dims.length == 4 && dims[2] == nChannels && is2d)
    		return rai;
    	else if (dims.length == 5 && dims[2] == nChannels && is2d)
    		return Views.hyperSlice(rai, 3, 0);
    	else if (dims.length == 3 && dims[2] != nChannels && nChannels == 1 && !is2d) {
    		rai = Views.permute(Views.addDimension(rai, 0, 0), 2, 3);
    		return Views.addDimension(rai, 0, 0);
    	} else if (dims.length == 4 && dims[2] != nChannels && nChannels == 1 && !is2d)
    		return Views.permute(Views.permute(Views.addDimension(rai, 0, 0), 3, 4), 2, 3);
    	else if (dims.length == 4 && dims[2] == nChannels && !is2d)
    		return Views.addDimension(rai, 0, 0);
    	else if (dims.length == 5 && dims[2] == nChannels && !is2d)
    		return rai;
    	else if (dims.length == 3 && dims[2] != nChannels && is2d)
    		throw new IllegalArgumentException(String.format("Number of channels required for this model is: %s."
    				+ " The number of channels (third dimension) in the image provided: %s.", nChannels, dims[2]));
    	else if (dims.length == 3 && dims[2] != nChannels && is2d)
    		throw new IllegalArgumentException(String.format("Number of channels required for this model is: %s."
    				+ " The number of channels (third dimension) in the image provided: %s.", nChannels, dims[2]));
    	else if (dims.length == 2 && nChannels > 1)
    		throw new IllegalArgumentException(String.format("Model requires %s channels", nChannels));
    	else if (dims.length == 2 && !is2d)
    		throw new IllegalArgumentException("Model is 3d, 2d image provided");
    	else
    		throw new IllegalArgumentException(
    				String.format("Unsupported dimensions for %s model with %s channels. Dimension order should be (X, Y, C, Z, B or T)"
    						, is2d ? "2D" : "3D", nChannels));
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
    	SwingUtilities.invokeLater(() -> this.footer.getBar().setString("Installing..."));
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
    		INSTALLED_WEIGHTS = true;
    		return true;
    	}
    	try {
			Stardist2D pretrained = Stardist2D.fromPretained(model, consumer.getModelsDir(), false);
			if (pretrained == null)
				return false;
		} catch (Exception e) {
			return false;
		}
		INSTALLED_WEIGHTS = true;
    	return true;
    }
    
    private void installModelWeights(CountDownLatch latch) {
    	Consumer<Double> cons = (d) -> {
    		double perc = Math.round(d * 1000) / 10.0d;
    		SwingUtilities.invokeLater(() -> {
        		this.footer.getBar().setValue((int) Math.floor(perc));
        		this.footer.getBar().setString(perc + "% of weights");
    		});
    	};
		SwingUtilities.invokeLater(() -> footer.getBar().setIndeterminate(false));
		Thread dwnlThread = new Thread(() -> {
			try {
				Stardist2D.downloadPretrained((String) modelComboBox.getSelectedItem(), this.consumer.getModelsDir(), cons);
				INSTALLED_WEIGHTS = true;
			} catch (IllegalArgumentException e) {
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
    		INSTALLED_ENV = bool;
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
				if (!footer.getBar().isIndeterminate() || (footer.getBar().isIndeterminate() && !footer.getBar().getString().equals("Installing Python"))) {
					footer.getBar().setIndeterminate(true);
					footer.getBar().setString("Installing Python");
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
        	this.footer.getButtons().getRunButton().setEnabled(!isStarting);
        	this.footer.getButtons().getInstallButton().setEnabled(!isStarting);
        	this.modelComboBox.setEnabled(!isStarting);
        	this.optionalParams.getMinPercField().setEnabled(!isStarting);
        	this.optionalParams.getMaxPercField().setEnabled(!isStarting);
        	if (isStarting) {
        		this.footer.getBar().setString("Checking stardist installed...");
        		this.footer.getBar().setIndeterminate(true);
        	} else {
        		this.footer.getBar().setIndeterminate(false);
		    	this.footer.getBar().setValue(0);
		    	this.footer.getBar().setString("");
        	}
    	});
    }
    
    private void browseFiles() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int option = fileChooser.showOpenDialog(StarDistPluginUI.this);
        if (option == JFileChooser.APPROVE_OPTION) {
            customModelPathField.setText(fileChooser.getSelectedFile().getAbsolutePath());
        }
    }
}
