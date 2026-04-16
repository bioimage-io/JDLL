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

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloGUI;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.view.Views;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class YOLOPluginUI extends YoloGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
    private static boolean INSTALLED_ENV = false;
    
    private static HashMap<String, Boolean> INSTALLED_WEIGHTS;
    
    static {
    	INSTALLED_WEIGHTS = new HashMap<String, Boolean>();
    	INSTALLED_WEIGHTS.put("cyto3", false);
    	INSTALLED_WEIGHTS.put("cyto2", false);
    	INSTALLED_WEIGHTS.put("cyto", false);
    	INSTALLED_WEIGHTS.put("nuclei", false);
    }
    
    private final ConsumerInterface consumer;
    private String whichLoaded;
    private Cellpose model;
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
                JFrame frame = new JFrame("Cellpose Plugin");
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
    	String modelPath = (String) this.inferencePanel.getModelSelectionPanel().getModelComboBox().getSelectedItem();
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	this.inputTitle = consumer.getFocusedImageName();
    	if (whichLoaded != null && !whichLoaded.equals(modelPath))
    		model.close();
    	if (model == null || !model.isLoaded()) {
    		model = Cellpose.init(modelPath);
    		model.loadModel();
    	}
    	whichLoaded = modelPath;
    	Float diameter = null;
    	runYOLOOnFramesStack(rai, diameter);
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runYOLOOnFramesStack(RandomAccessibleInterval<R> rai, Float diameter) throws RunModelException {
    	rai = addDimsToInput(rai, rai.dimensionsAsLongArray().length > 2  && rai.dimensionsAsLongArray()[2] == 3 ? 3 : 1);
    	long[] inDims = rai.dimensionsAsLongArray();
    	long[] outDims = new long[] {inDims[0], inDims[1], inDims[3]};
		RandomAccessibleInterval<T> outMaskRai = Cast.unchecked(ArrayImgs.unsignedShorts(outDims));
		RandomAccessibleInterval<T> output1 = Cast.unchecked(ArrayImgs.unsignedBytes(new long[] {inDims[0], inDims[1], 3, inDims[3]}));
		RandomAccessibleInterval<T> output2 = Cast.unchecked(ArrayImgs.floats(new long[] {2, inDims[0], inDims[1], inDims[3]}));
		RandomAccessibleInterval<T> output3 = Cast.unchecked(ArrayImgs.floats(new long[] {inDims[0], inDims[1], inDims[3]}));
		RandomAccessibleInterval<T> output4 = Cast.unchecked(ArrayImgs.floats(new long[] {inDims[0], inDims[1], 3, inDims[3]}));
		
		for (int i = 0; i < rai.dimensionsAsLongArray()[3]; i ++) {
			String msg = "Running the model " + (i + 1) + "/" + rai.dimensionsAsLongArray()[3];
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
	    	
	    	if (diameter != null)
	    		model.setDiameter(diameter);
	    	model.run(inList, outputList);
		}
    	consumer.display(outMaskRai, "xyb", getOutputName("labels"));
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
}
