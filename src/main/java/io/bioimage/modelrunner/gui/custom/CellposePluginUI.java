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
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

public class CellposePluginUI extends CellposeGUI implements ActionListener {

    private static final long serialVersionUID = 5381352117710530216L;
    
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
        	this.nucleiCbox.setModel(new DefaultComboBoxModel<>(RGB_LIST));
        	this.cytoCbox.setModel(new DefaultComboBoxModel<>(RGB_LIST));
        }


        this.consumer.setVariableNames(VAR_NAMES);
        componentList.add(this.modelComboBox);
        componentList.add(this.customModelPathField);
        componentList.add(this.cytoCbox);
        componentList.add(this.nucleiCbox);
        componentList.add(this.diameterField);
        componentList.add(this.check);
        this.consumer.setComponents(componentList);
        this.footer.buttons.cancelButton.addActionListener(this);
        this.footer.buttons.installButton.addActionListener(this);
        this.footer.buttons.runButton.addActionListener(this);
        this.browseButton.addActionListener(this);

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
    	} else if (e.getSource() == this.footer.buttons.runButton) {
    		workerThread = new Thread(() -> {
        		try {
    				runCellpose();
    				startModelInstallation(false);
    			} catch (IOException | RunModelException | LoadModelException e1) {
    				e1.printStackTrace();
    				startModelInstallation(false);
    				SwingUtilities.invokeLater(() -> this.footer.bar.setString("Error running the model"));
    			}
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.footer.buttons.installButton) {
    		workerThread = new Thread(() -> {
        		installCellpose();
    		});
    		workerThread.start();
    	} else if (e.getSource() == this.footer.buttons.cancelButton) {
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
    
    private < T extends RealType< T > & NativeType< T > > void runCellpose() throws IOException, RunModelException, LoadModelException {
    	startModelInstallation(true);
    	installCellpose(weightsInstalled(), Cellpose.isInstalled());
    	RandomAccessibleInterval<T> rai = consumer.getFocusedImageAsRai();
    	this.inputTitle = consumer.getFocusedImageName();
    	if (rai == null) {
    		JOptionPane.showMessageDialog(null, "Please open an image", "No image open", JOptionPane.ERROR_MESSAGE);
    		return;
    	}
    	SwingUtilities.invokeLater(() ->{
    		footer.bar.setIndeterminate(true);
    		footer.bar.setString("Loading model");
    	});
    	String modelPath = (String) this.modelComboBox.getSelectedItem();
    	if (modelPath.equals(CUSOTM_STR))
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
    		footer.bar.setString("Running the model");
    	});
    	model.setChannels(new int[] {CHANNEL_MAP.get(cytoCbox.getSelectedItem()), CHANNEL_MAP.get(nucleiCbox.getSelectedItem())});
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
    	consumer.display(outMaskRai, "xyb", "mask");
    	if (!check.isSelected())
    		return;
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runCellposeOnTensor(RandomAccessibleInterval<R> rai) throws RunModelException {
		Tensor<R> tensor = Tensor.build("input", "xyc", rai);
    	List<Tensor<R>> inList = new ArrayList<Tensor<R>>();
		inList.add(tensor);
    	List<Tensor<T>> out = model.run(inList);
    	for (Tensor<T> tt : out) {
    		if (!check.isSelected() && !tt.getName().equals("labels"))
    			continue;
    		else if (tt.getAxesOrder().length == 1)
    			continue;
        	consumer.display(tt.getData(), tt.getAxesOrderString(), getOutputName(tt.getName()));
    	}
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
    	SwingUtilities.invokeLater(() -> footer.bar.setString("Installing..."));
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
    	if (model.equals(CUSOTM_STR))
    		return true;
    	try {
			String path = Cellpose.findPretrainedModelInstalled(model, consumer.getModelsDir());
			if (path == null)
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
        		footer.bar.setValue((int) Math.floor(perc));
        		footer.bar.setString(perc + "% of weights");
    		});
    	};
    	SwingUtilities.invokeLater(() -> footer.bar.setIndeterminate(false));
		Thread dwnlThread = new Thread(() -> {
			try {
				Cellpose.donwloadPretrained((String) modelComboBox.getSelectedItem(), this.consumer.getModelsDir(), cons);
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
    	if (Cellpose.isInstalled() || 
    			JOptionPane.showConfirmDialog(null, msg, question, JOptionPane.YES_NO_OPTION) != JOptionPane.YES_OPTION) {
			latch.countDown();
			checkModelInstallationFinished(latch);
    		return;
    	}
		JDialog installerFrame = new JDialog();
		installerFrame.setTitle("Installing Cellpose");
		installerFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
    	Runnable callback = () -> {
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
				if (!footer.bar.isIndeterminate() || (footer.bar.isIndeterminate() && !footer.bar.getString().equals("Installing Python"))) {
					footer.bar.setIndeterminate(true);
					footer.bar.setString("Installing Python");
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
        	footer.buttons.runButton.setEnabled(!isStarting);
        	footer.buttons.installButton.setEnabled(!isStarting);
        	modelComboBox.setEnabled(!isStarting);
        	diameterField.setEnabled(!isStarting);
        	cytoCbox.setEnabled(!isStarting);
        	nucleiCbox.setEnabled(!isStarting);
        	check.setEnabled(!isStarting);
        	if (isStarting) {
        		footer.bar.setString("Checking cellpose installed...");
        		footer.bar.setIndeterminate(true);
        	} else {
        		footer.bar.setIndeterminate(false);
        		footer.bar.setValue(0);
        		footer.bar.setString("");
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
