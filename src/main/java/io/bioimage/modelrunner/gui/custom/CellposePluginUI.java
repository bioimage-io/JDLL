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

import javax.swing.DefaultComboBoxModel;
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
import io.bioimage.modelrunner.gui.custom.gui.CellposeGUI;
import io.bioimage.modelrunner.gui.workers.InstallEnvWorker;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

public class CellposePluginUI extends CellposeGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
    private static boolean INSTALLED_WEIGHTS = false;
    
    private static boolean INSTALLED_ENV = false;
    
    private final ConsumerInterface consumer;
    private String whichLoaded;
    private Cellpose model;
    private String inputTitle;
    
    private Runnable cancelCallback;
    Thread workerThread;

    public CellposePluginUI(ConsumerInterface consumer) {
        // Set a modern-looking border layout with padding
    	this.consumer = consumer;
    	List<JComponent> componentList = new ArrayList<JComponent>();

        if (consumer.getFocusedImageChannels() != null && consumer.getFocusedImageChannels() == 1) {
        	this.nucleiCbox.setModel(new DefaultComboBoxModel<>(GRAYSCALE_LIST));
        	this.cytoCbox.setModel(new DefaultComboBoxModel<>(GRAYSCALE_LIST));
        } else if (consumer.getFocusedImageChannels() != null && consumer.getFocusedImageChannels() == 3) {
            	this.nucleiCbox.setModel(new DefaultComboBoxModel<>(RGB_LIST));
            	this.cytoCbox.setModel(new DefaultComboBoxModel<>(RGB_LIST));
        } else {
        	this.nucleiCbox.setModel(new DefaultComboBoxModel<>(ALL_LIST));
        	this.cytoCbox.setModel(new DefaultComboBoxModel<>(ALL_LIST));
        }


        this.consumer.setVariableNames(VAR_NAMES);
        componentList.add(this.modelComboBox);
        componentList.add(this.customModelPathField);
        componentList.add(this.cytoCbox);
        componentList.add(this.nucleiCbox);
        componentList.add(this.diameterField);
        componentList.add(this.check);
        this.consumer.setComponents(componentList);
        this.footer.getButtons().getCancelButton().addActionListener(this);
        this.footer.getButtons().getInstallButton().addActionListener(this);
        this.footer.getButtons().getRunButton().addActionListener(this);
        this.browseButton.addActionListener(this);

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
                JFrame frame = new JFrame("Cellpose Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new CellposePluginUI(null));
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setResizable(true);
                frame.setSize(400, 200);
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
    				runCellpose();
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
        		installCellpose();
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
    	if (diameterField.getText() != null && !diameterField.getText().equals(""))
    		map.put("diameter", diameterField.getText());
    	map.put("cyto_color", (String) cytoCbox.getSelectedItem());
    	map.put("nuclei_color", (String) nucleiCbox.getSelectedItem());
    	map.put("display_all", "" + this.check.isSelected());
    	this.consumer.notifyParams(map);
    }
    
    private < T extends RealType< T > & NativeType< T > > void runCellpose() throws IOException, RunModelException, LoadModelException {
    	saveParams();
    	startModelInstallation(true);
    	if (!INSTALLED_WEIGHTS || !INSTALLED_ENV)
    		installCellpose(weightsInstalled(), (INSTALLED_ENV = Cellpose.isInstalled()));
    	if (!INSTALLED_WEIGHTS || !INSTALLED_ENV)
    		return;
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	this.inputTitle = consumer.getFocusedImageName();
    	SwingUtilities.invokeLater(() ->{
    		footer.getBar().setIndeterminate(true);
    		footer.getBar().setString("Loading model");
    	});
    	String modelPath = (String) this.modelComboBox.getSelectedItem();
    	if (modelPath.equals(CUSTOM_STR))
    		modelPath = this.customModelPathField.getText();
    	else
    		modelPath = Cellpose.findPretrainedModelInstalled(modelPath, consumer.getModelsDir());
    	if (whichLoaded != null && !whichLoaded.equals(modelPath))
    		model.close();
    	if (model == null || !model.isLoaded()) {
    		model = Cellpose.init(modelPath);
    		model.loadModel();
    	}
    	whichLoaded = modelPath;
    	SwingUtilities.invokeLater(() ->{
    		footer.getBar().setString("Running the model");
    	});
    	if (diameterField.getText() != null &&!diameterField.getText().equals(""))
    		model.setDiameter(Float.parseFloat(diameterField.getText()));
    	model.setChannels(new int[] {CHANNEL_MAP.get(cytoCbox.getSelectedItem()), CHANNEL_MAP.get(nucleiCbox.getSelectedItem())});
    	runCellposeOnFramesStack(rai);
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runCellposeOnFramesStack(RandomAccessibleInterval<R> rai) throws RunModelException {
    	rai = addDimsToInput(rai, cytoCbox.getSelectedItem().equals("gray") ? 1 : 3);
    	long[] inDims = rai.dimensionsAsLongArray();
    	long[] outDims = new long[] {inDims[0], inDims[1], inDims[3]};
		RandomAccessibleInterval<T> outMaskRai = Cast.unchecked(ArrayImgs.unsignedShorts(outDims));
		RandomAccessibleInterval<T> output1 = Cast.unchecked(ArrayImgs.unsignedBytes(new long[] {inDims[0], inDims[1], 3, inDims[3]}));
		RandomAccessibleInterval<T> output2 = Cast.unchecked(ArrayImgs.floats(new long[] {2, inDims[0], inDims[1], inDims[3]}));
		RandomAccessibleInterval<T> output3 = Cast.unchecked(ArrayImgs.floats(new long[] {inDims[0], inDims[1], inDims[3]}));
		RandomAccessibleInterval<T> output4 = Cast.unchecked(ArrayImgs.floats(new long[] {inDims[0], inDims[1], 3, inDims[3]}));
		
		for (int i = 0; i < rai.dimensionsAsLongArray()[3]; i ++) {
			String msg = "Running the model " + (i + 1) + "/" + rai.dimensionsAsLongArray()[3];
			SwingUtilities.invokeLater(() -> footer.getBar().setString(msg));
	    	List<Tensor<R>> inList = new ArrayList<Tensor<R>>();
	    	Tensor<R> inIm = Tensor.build("input", "xyc", Views.hyperSlice(rai, 3, i));
	    	inList.add(inIm);
	    	
	    	List<Tensor<T>> outputList = new ArrayList<Tensor<T>>();
	    	Tensor<T> outMask = Tensor.build("labels", "xy", Views.hyperSlice(outMaskRai, 2, i));
	    	outputList.add(outMask);
	    	Tensor<T> flows0 = Tensor.build("flows_0", "xyc", Views.hyperSlice(output1, 3, i));
	    	outputList.add(flows0);
	    	Tensor<T> flows1 = Tensor.build("flows_1", "cxy", Views.hyperSlice(output2, 3, i));
	    	outputList.add(flows1);
	    	Tensor<T> flows2 = Tensor.build("flows_2", "xy", Views.hyperSlice(output3, 2, i));
	    	outputList.add(flows2);
	    	Tensor<T> st = Tensor.buildEmptyTensor("styles", "i");
	    	outputList.add(st);
	    	Tensor<T> dn = Tensor.build("image_dn", "xyc", Views.hyperSlice(output4, 3, i));
	    	outputList.add(dn);
	    	
	    	model.run(inList, outputList);
		}
    	consumer.display(outMaskRai, "xyb", getOutputName("labels"));
    	if (!check.isSelected())
    		return;
    	consumer.display(output1, "xycb", getOutputName("flows_0"));
    	consumer.display(output2, "cxyb", getOutputName("flows_1"));
    	consumer.display(output3, "xyb", getOutputName("flows_2"));
    	consumer.display(output4, "xycb", getOutputName("image_dn"));
    }
    
    private static <R extends RealType<R> & NativeType<R>>
    RandomAccessibleInterval<R> addDimsToInput(RandomAccessibleInterval<R> rai, int nChannels) {
    	long[] dims = rai.dimensionsAsLongArray();
    	if (dims.length == 2 && nChannels == 1)
    		return Views.addDimension(Views.addDimension(rai, 0, 0), 0, 0);
    	else if (dims.length == 2)
    		throw new IllegalArgumentException("Cyto and nuclei specified for RGB image and image provided is grayscale.");
    	else if (dims.length == 3 && dims[2] == nChannels)
    		return Views.addDimension(rai, 0, 0);
    	else if (dims.length == 3 && nChannels == 1)
    		return Views.permute(Views.addDimension(rai, 0, 0), 2, 3);
    	else if (dims.length >= 3 && dims[2] == 1 && nChannels == 3)
    		throw new IllegalArgumentException("Expected RGB (3 channels) image and got instead grayscale image (1 channel).");
    	else if (dims.length == 4 && dims[2] == nChannels)
    		return rai;
    	else if (dims.length == 5 && dims[2] == nChannels && dims[4] != 1)
    		return Views.hyperSlice(rai, 3, 0);
    	else if (dims.length == 5 && dims[2] == nChannels && dims[4] == 1)
    		return Views.hyperSlice(Views.permute(rai, 3, 4), 3, 0);
    	else if (dims.length == 4 && dims[2] != nChannels && nChannels == 1) {
    		rai = Views.hyperSlice(rai, 2, 0);
    		rai = Views.addDimension(rai, 0, 0);
    		return Views.permute(rai, 2, 3);
    	} else if (dims.length == 5 && dims[2] != nChannels)
    		throw new IllegalArgumentException("Expected grayscale (1 channel) image and got instead RGB image (3 channels).");
    	else
    		throw new IllegalArgumentException("Unsupported dimensions for Cellpose model");
    }
    
    private String getOutputName(String tensorName) {
    	String noExtension;
    	if (inputTitle.lastIndexOf(".") != -1)
    		noExtension = inputTitle.substring(0, inputTitle.lastIndexOf("."));
    	else
    		noExtension = inputTitle;
    	String extension = ".tif";
    	return noExtension + "_" + tensorName + extension;
    }
    
    private void installCellpose() {
    	startModelInstallation(true);
    	boolean envInstalled = Cellpose.isInstalled();
    	boolean wwInstalled = weightsInstalled();
    	if (envInstalled && wwInstalled) {
        	startModelInstallation(false);
    		return;
    	}
    	installCellpose(wwInstalled, envInstalled);
    }
    
    private void installCellpose(boolean wwInstalled, boolean envInstalled) {
    	if (wwInstalled && envInstalled)
    		return;
    	SwingUtilities.invokeLater(() -> footer.getBar().setString("Installing..."));
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
			String path = Cellpose.findPretrainedModelInstalled(model, consumer.getModelsDir());
			if (path == null)
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
        		footer.getBar().setValue((int) Math.floor(perc));
        		footer.getBar().setString(perc + "% of weights");
    		});
    	};
    	SwingUtilities.invokeLater(() -> footer.getBar().setIndeterminate(false));
		Thread dwnlThread = new Thread(() -> {
			try {
				Cellpose.donwloadPretrained((String) modelComboBox.getSelectedItem(), this.consumer.getModelsDir(), cons);
				INSTALLED_WEIGHTS = true;
			} catch (IOException | InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
			latch.countDown();
			checkModelInstallationFinished(latch);
		});
		dwnlThread.start();
    }
    
    private void installEnv(CountDownLatch latch) {
    	String msg = "Installation of Python environments might take up to 20 minutes.";
    	String question = "Install Python for Cellpose";
    	if (JOptionPane.showConfirmDialog(null, msg, question, JOptionPane.YES_NO_OPTION) != JOptionPane.YES_OPTION) {
			latch.countDown();
			checkModelInstallationFinished(latch);
    		return;
    	}
		JDialog installerFrame = new JDialog();
		installerFrame.setTitle("Installing Cellpose");
		installerFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
    	Consumer<Boolean> callback = (bool) -> {
    		INSTALLED_ENV = bool;
    		checkModelInstallationFinished(latch);
    		if (installerFrame.isVisible())
    			installerFrame.dispose();
    	};
    	InstallEnvWorker worker = new InstallEnvWorker("Cellpose", latch, callback);
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
        	footer.getButtons().getRunButton().setEnabled(!isStarting);
        	footer.getButtons().getInstallButton().setEnabled(!isStarting);
        	modelComboBox.setEnabled(!isStarting);
        	diameterField.setEnabled(!isStarting);
        	cytoCbox.setEnabled(!isStarting);
        	nucleiCbox.setEnabled(!isStarting);
        	check.setEnabled(!isStarting);
        	if (isStarting) {
        		footer.getBar().setString("Checking cellpose installed...");
        		footer.getBar().setIndeterminate(true);
        	} else {
        		footer.getBar().setIndeterminate(false);
        		footer.getBar().setValue(0);
        		footer.getBar().setString("");
        	}
    	});
    }
    
    private void browseFiles() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int option = fileChooser.showOpenDialog(CellposePluginUI.this);
        if (option == JFileChooser.APPROVE_OPTION) {
            customModelPathField.setText(fileChooser.getSelectedFile().getAbsolutePath());
        }
    }
}
