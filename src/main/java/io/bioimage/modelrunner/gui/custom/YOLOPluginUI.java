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
import javax.swing.Timer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloGUI;
import io.bioimage.modelrunner.gui.custom.yolo.YoloInferenceService;
import io.bioimage.modelrunner.gui.custom.yolo.YoloInstaller;
import io.bioimage.modelrunner.gui.custom.yolo.YoloModelRegistry;
import io.bioimage.modelrunner.gui.custom.yolo.YoloTrainingConfig;
import io.bioimage.modelrunner.gui.custom.yolo.YoloTrainingService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

public class YOLOPluginUI extends YoloGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;

    private final ConsumerInterface consumer;
    private final YoloInstaller installer = new YoloInstaller();
    private final YoloInferenceService inferenceService = new YoloInferenceService(installer);
    private final YoloTrainingService trainingService = new YoloTrainingService(installer);
    private boolean cancelled = false;
    private Timer trainingTimer;
    private long trainingStartMillis;
    private long lastProgressMillis;
    private int lastProgressStep;
    private int currentTrainingStep;
    private int totalTrainingSteps;
    private double currentSecondsPerStep = Double.NaN;
    
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

    	String modelsDir = consumer == null ? null : consumer.getModelsDir();
    	LinkedHashMap<String, String> yoloModelEntries = YoloModelRegistry.buildModelEntries(modelsDir);
    	this.inferencePanel.getModelSelectionPanel().setModels(yoloModelEntries);
    	this.trainPanel.setBaseModels(yoloModelEntries);
        if (this.consumer == null) {
            return;
        }
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
        this.trainPanel.getTrainActionPanel().getCancelButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getRunButton().addActionListener(this);
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
    	inferenceService.close();
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
    		cancelled = false;
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
    	} else if (e.getSource() == this.trainPanel.getTrainActionPanel().getRunButton()) {
    		trainYOLO();
    	} else if (e.getSource() == this.trainPanel.getTrainActionPanel().getCancelButton()) {
    		cancel();
    	}
    }
    
    private void cancel() {
    	cancelled = true;
    	if (workerThread != null && workerThread.isAlive())
    		workerThread.interrupt();
    	inferenceService.close();
    	finishTrainingUiState();
    	if (cancelCallback != null)
    		cancelCallback.run();
    }
    
    private void saveParams() {
    	this.consumer.notifyParams(null);
    }
    
    private < T extends RealType< T > & NativeType< T > > void runYOLO()
    		throws RunModelException, LoadModelException, BuildException, IOException,
    		ExecutionException, InterruptedException {
    	saveParams();
    	String modelPath = this.inferencePanel.getModelSelectionPanel().getSelectedModelValue();
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	Consumer<String> logConsumer = str -> SwingUtilities.invokeLater(() ->
    			YOLOPluginUI.this.inferencePanel.getLogPanel().appendHtml(str));
    	consumer.displayDetections(inferenceService.run(modelPath, rai, logConsumer));
    }

    public void trainYOLO() {
    	if (!trainPanel.validateTrainingFields()) {
    		return;
    	}
    	cancelled = false;
    	startTrainingUiState();
    	workerThread = new Thread(() -> {
    		try {
    			YoloTrainingConfig config = readTrainingConfig();
    			Consumer<String> logConsumer = str -> SwingUtilities.invokeLater(() ->{
					YOLOPluginUI.this.inferencePanel.getLogPanel().appendHtml(str);
					System.err.println(str);
    			});
    			trainingService.train(config,
    					progress -> SwingUtilities.invokeLater(() -> {
    						updateTrainingProgressState(progress.getStep(), progress.getTotalSteps());
    						trainPanel.getLossGraphPanel().addTrainValue(
    								firstKey(progress.getLosses(), "loss"),
    								progress.getStep(),
    								progress.getEpoch(),
    								progress.getPrimaryLoss());
    					}),
    					preview -> SwingUtilities.invokeLater(() ->
    							logConsumer.accept("Validation preview checkpoint: " + preview.getCheckpointPath())),
    					logConsumer);
    			refreshYoloModels();
    		} catch (Exception e) {
    			if (!cancelled) {
    				e.printStackTrace();
    			}
    		} finally {
    			SwingUtilities.invokeLater(() -> finishTrainingUiState());
    		}
    	});
    	workerThread.start();
    }

    private void startTrainingUiState() {
    	trainingStartMillis = System.currentTimeMillis();
    	lastProgressMillis = trainingStartMillis;
    	lastProgressStep = 0;
    	currentTrainingStep = 0;
    	totalTrainingSteps = 0;
    	currentSecondsPerStep = Double.NaN;
    	trainPanel.setTrainingRunning(true);
    	tabs.setEnabledAt(0, false);
    	trainPanel.getLossGraphPanel().clearValues();
    	trainPanel.getMetricGraphPanel().clearValues();
    	trainPanel.getLossGraphPanel().setTrainingStatus(true, 0, 0, 0L, Double.NaN);
    	trainPanel.getMetricGraphPanel().setTrainingStatus(true, 0, 0, 0L, Double.NaN);
    	if (trainingTimer != null) {
    		trainingTimer.stop();
    	}
    	trainingTimer = new Timer(1000, e -> updateTrainingGraphStatus());
    	trainingTimer.start();
    }

    private void finishTrainingUiState() {
    	if (trainingTimer != null) {
    		trainingTimer.stop();
    		trainingTimer = null;
    	}
    	trainPanel.setTrainingRunning(false);
    	tabs.setEnabledAt(0, true);
    }

    private void updateTrainingProgressState(int step, int totalSteps) {
    	long now = System.currentTimeMillis();
    	if (totalSteps > 0) {
    		totalTrainingSteps = totalSteps;
    	}
    	if (step > lastProgressStep) {
    		currentSecondsPerStep = (now - lastProgressMillis) / 1000.0d / (step - lastProgressStep);
    		lastProgressMillis = now;
    		lastProgressStep = step;
    	}
    	currentTrainingStep = Math.max(currentTrainingStep, step);
    	updateTrainingGraphStatus();
    }

    private void updateTrainingGraphStatus() {
    	long elapsed = Math.max(0L, System.currentTimeMillis() - trainingStartMillis);
    	trainPanel.getLossGraphPanel().setTrainingStatus(true, currentTrainingStep, totalTrainingSteps, elapsed, currentSecondsPerStep);
    	trainPanel.getMetricGraphPanel().setTrainingStatus(true, currentTrainingStep, totalTrainingSteps, elapsed, currentSecondsPerStep);
    }

    private YoloTrainingConfig readTrainingConfig() {
    	String modelsDir = consumer == null ? null : consumer.getModelsDir();
    	return YoloTrainingConfig.fromUi(
    			trainPanel.getModelNameField().getText(),
    			trainPanel.getDatasetField().getText(),
    			Integer.parseInt(trainPanel.getEpochsField().getText().trim()),
    			trainPanel.getFineTuneRadio().isSelected(),
    			trainPanel.getSelectedBaseModelValue(),
    			modelsDir);
    }

    private void refreshYoloModels() {
    	String modelsDir = consumer == null ? null : consumer.getModelsDir();
    	LinkedHashMap<String, String> yoloModelEntries = YoloModelRegistry.buildModelEntries(modelsDir);
    	SwingUtilities.invokeLater(() -> {
    		inferencePanel.getModelSelectionPanel().setModels(yoloModelEntries);
    		trainPanel.setBaseModels(yoloModelEntries);
    	});
    }

    private static String firstKey(java.util.Map<String, Double> values, String fallback) {
    	return values == null || values.isEmpty() ? fallback : values.keySet().iterator().next();
    }
}
