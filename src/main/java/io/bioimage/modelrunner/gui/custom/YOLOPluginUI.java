/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import org.apposed.appose.BuildException;
import org.apposed.appose.builder.PixiBuilderFactory;
import org.apposed.appose.tool.Pixi;

import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloGUI;
import io.bioimage.modelrunner.model.detection.Detection;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

public class YOLOPluginUI extends YoloGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
    private static final String YOLO_MODELS_SUBDIR = "yolo";
    private static final String YOLO_WEIGHTS_EXTENSION = ".pt";
    private static final String PRERAINED_URL_FORMAT = "https://github.com/ultralytics/assets/releases/download/v8.4.0/%s";
    private static final String[][] PRETRAINED_MODELS = new String[][] {
            {"YOLO26n", "yolo26n.pt"},
            {"YOLO26m", "yolo26m.pt"},
            {"YOLO26x", "yolo26x.pt"}
    };
    
    private static final HashMap<String, Long> YOLO_PRETRAINED_SIZE;
    static {
    	YOLO_PRETRAINED_SIZE = new HashMap<String, Long>();
    	YOLO_PRETRAINED_SIZE.put("yolo26n.pt", 5_544_453L);
    	YOLO_PRETRAINED_SIZE.put("yolo26m.pt", 44_255_705L);
    	YOLO_PRETRAINED_SIZE.put("yolo26x.pt", 118_667_365L);
    }
    
    private static final String ENV_NAME = DLModelPytorch.COMMON_PYTORCH_ENV_NAME;
    
    private final ConsumerInterface consumer;
    private String whichLoaded;
    private Yolo model;
    private String inputTitle;
    private boolean cancelled = false;
    
    private Runnable cancelCallback;
    Thread workerThread;

    /**
     * Creates a new CellposePluginUI.
     *
     * @param consumer the consumer parameter.
     */
    public YOLOPluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
    	super(adapter);
        // Set a modern-looking border layout with padding
    	this.consumer = consumer;
    	List<JComponent> componentList = new ArrayList<JComponent>();

    	this.inferencePanel.getModelSelectionPanel().setModels(buildYoloModelEntries());
        this.consumer.setVariableNames(null);
        componentList.add(this.inferencePanel.getModelSelectionPanel().getModelComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getOpenImagesComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getFocusButton());
        componentList.add(this.inferencePanel.getImageDisplayPanel());
        this.consumer.setComponents(componentList);
        this.inferencePanel.getDrawButton().addActionListener(this);
        this.inferencePanel.getRefreshButton().addActionListener(this);
        this.inferencePanel.getActionPanel().getCancelButton().addActionListener(this);
        this.inferencePanel.getActionPanel().getRunButton().addActionListener(this);
        consumer.updateGUI();
    }

    private LinkedHashMap<String, String> buildYoloModelEntries() {
        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        String modelsDir = consumer == null ? null : consumer.getModelsDir();
        File yoloDir = modelsDir == null ? new File(YOLO_MODELS_SUBDIR) : new File(modelsDir, YOLO_MODELS_SUBDIR);

        for (String[] pretrained : PRETRAINED_MODELS) {
            models.put("[Pretrained] " + pretrained[0], new File(yoloDir, pretrained[1]).getAbsolutePath());
        }

        File[] customModels = yoloDir.listFiles(file -> file.isFile()
                && file.getName().toLowerCase().endsWith(YOLO_WEIGHTS_EXTENSION)
                && !isPretrainedWeightsFile(file.getName()));
        if (customModels == null) {
            return models;
        }
        Arrays.sort(customModels, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        for (File modelFile : customModels) {
            models.put("[Custom] " + removeWeightsExtension(modelFile.getName()), modelFile.getAbsolutePath());
        }
        return models;
    }

    private static boolean isPretrainedWeightsFile(String fileName) {
        for (String[] pretrained : PRETRAINED_MODELS) {
            if (pretrained[1].equalsIgnoreCase(fileName)) {
                return true;
            }
        }
        return false;
    }

    private static String removeWeightsExtension(String fileName) {
        if (fileName.toLowerCase().endsWith(YOLO_WEIGHTS_EXTENSION)) {
            return fileName.substring(0, fileName.length() - YOLO_WEIGHTS_EXTENSION.length());
        }
        return fileName;
    }
    
    /**
     * Sets cancel callback.
     *
     * @param cancelCallback the cancelCallback parameter.
     */
    public void setCancelCallback(Runnable cancelCallback) {
    	this.cancelCallback = cancelCallback;
    }
    
    /**
     * Executes close.
     */
    public void close() {
    	if (model != null && model.isLoaded())
    		model.close();
    }

    // For demonstration purposes: a main method to show the UI in a JFrame.
    /**
     * Executes main.
     *
     * @param args the args parameter.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            /**
             * Executes run.
             */
            public void run() {
                JFrame frame = new JFrame("Yolo Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new YOLOPluginUI(null, null));
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frame.setResizable(true);
                frame.setSize(400, 200);
            }
        });
    }
    
    /**
     * Executes action performed.
     *
     * @param e the e parameter.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
    	if (e.getSource() == this.inferencePanel.getActionPanel().getRunButton()) {
    		workerThread = new Thread(() -> {
        		try {
        			runYOLO();
    			} catch (Exception e1) {
    				if (cancelled)
    					return;
    				e1.printStackTrace();
    			}
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.inferencePanel.getActionPanel().getCancelButton()) {
    		cancel();
    	}
    }
    
    private void cancel() {
    	cancelled = true;
    	if (workerThread != null && workerThread.isAlive())
    		workerThread.interrupt();
    	if (model != null)
    		model.close();
    	if (cancelCallback != null)
    		cancelCallback.run();
    }
    
    private void saveParams() {
    	this.consumer.notifyParams(null);
    }
    
    private < T extends RealType< T > & NativeType< T > > void runYOLO() throws RunModelException, LoadModelException, BuildException, IOException {
    	saveParams();
    	String modelPath = this.inferencePanel.getModelSelectionPanel().getSelectedModelValue();
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	this.inputTitle = consumer.getFocusedImageName();
    	if (whichLoaded != null && !whichLoaded.equals(modelPath))
    		model.close();
    	if (model == null || !model.isLoaded()) {
    		installIfNeeded(modelPath);
    		model = Yolo.init(modelPath);
    		model.loadModel();
    	}
    	whichLoaded = modelPath;
    	Float diameter = null;
    	runYOLOOnFramesStack(rai, diameter);
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runYOLOOnFramesStack(RandomAccessibleInterval<R> rai, Float diameter) throws RunModelException {
    	rai = addDimsToInput(rai, rai.dimensionsAsLongArray().length > 2  && rai.dimensionsAsLongArray()[2] == 3 ? 3 : 1);
    	
    	List<Tensor<R>> outTensor = model.inference(Tensor.build("input", "xycb", rai));
    	
    	consumer.displayDetections(Detection.fromBN6Tensor(outTensor.get(0)));
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
    
    private boolean isModelInstalled(String modelPath) {
    	return new File(modelPath).isFile() 
    			&& YOLO_PRETRAINED_SIZE.get(new File(modelPath).getName()) == new File(modelPath).length();
    }
    
    private boolean isEnvInstalled() {
		PixiBuilderFactory builder = new PixiBuilderFactory();
		return builder.canWrap(new File(Pixi.BASE_PATH, ENV_NAME));
    }
    
    private void installIfNeeded(String modelPath) {
    	boolean wwInstalled = isModelInstalled(modelPath);
    	boolean envInstalled = isEnvInstalled();
    	if (envInstalled && wwInstalled)
    		return;
    	CountDownLatch latch = !wwInstalled && !envInstalled ? new CountDownLatch(2) : new CountDownLatch(1);
    	if (!wwInstalled)
    		installModelWeights(modelPath, latch);
    	if (!envInstalled)
    		installEnv(latch);
    	try {
			latch.await();
		} catch (InterruptedException e) {
	    	if (cancelled)
	    		return;
			e.printStackTrace();
		}
    }
    
    private void installModelWeights(String modelPath, CountDownLatch latch) {
    	String modelName = new File(modelPath).getName();
    	String downloadURL = String.format(PRERAINED_URL_FORMAT, new File(modelPath).getName());
    	Consumer<Double> cons = (d) -> {
    		double perc = Math.round(d * 1000) / 10.0d;
    		SwingUtilities.invokeLater(() -> {
        		YOLOPluginUI.this.inferencePanel.getLogPanel().appendHtml("Downloading " + modelName + " weigths: " + perc + "%");
    		});
    	};
    	Thread thread = Thread.currentThread();
		Thread dwnlThread = new Thread(() -> {
			try {
				FileDownloader fd = new FileDownloader(downloadURL, new File(modelPath), false);
				fd.setPartialProgressConsumer(cons);
				fd.download(thread);
				if (!isModelInstalled(modelPath))
					throw new IOException("Model not found or incorrect byte size: " + modelPath);
			} catch (IOException | ExecutionException e) {
		    	if (cancelled)
		    		return;
				e.printStackTrace();
			}
			latch.countDown();
		});
		dwnlThread.start();
    }
    
    private void installEnv(CountDownLatch latch) {
    	PixiEnvironmentSpec env = DLModelPytorch.resolvePytorchEnv();
    	Consumer<String> cons = (str) -> {
    		YOLOPluginUI.this.inferencePanel.getLogPanel().appendHtml(str);
    	};
		try {
			PixiEnvironmentManager.installRequirements(env, cons);
		} catch (InterruptedException e) {
		} catch (BuildException e) {
			e.printStackTrace();
		}
		latch.countDown();
    }
}
