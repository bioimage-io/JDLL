package io.bioimage.modelrunner.gui.workers;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

import javax.swing.SwingWorker;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.model.special.stardist.StardistAbstract;

public class InstallEnvWorker extends SwingWorker<Void, Void> {

    private final String modelFamily;
    private Consumer<String> consumer;
    private final CountDownLatch latch;
    private final Runnable callback;
    
    private Thread workerThread;

    public InstallEnvWorker(ModelDescriptor descriptor, CountDownLatch latch, Runnable callback) {
        this.modelFamily = descriptor.getModelFamily();
        this.latch = latch;
        this.callback = callback;
    }

    public InstallEnvWorker(String modelFamily, CountDownLatch latch, Runnable callback) {
        this.modelFamily = modelFamily;
        this.latch = latch;
        this.callback = callback;
    }
    
    public void setConsumer(Consumer<String> consumer) {
    	this.consumer = consumer;
    }
    
    public String getModelFamily() {
    	return this.modelFamily;
    }

    public CountDownLatch getCountDownLatch() {
    	return this.latch;
    }

    @Override
    protected Void doInBackground() {
    	workerThread = Thread.currentThread();
    	try {
            if (modelFamily.toLowerCase().equals(ModelDescriptor.STARDIST)) {
            	StardistAbstract.installRequirements(consumer);
            } else if (modelFamily.toLowerCase().equals(ModelDescriptor.CELLPOSE)) {
            	Cellpose.installRequirements(consumer);
            } else if (modelFamily.toLowerCase().equals(ModelDescriptor.BIOIMAGEIO))
            	DLModelPytorch.installRequirements(consumer);
		} catch (IOException | RuntimeException | MambaInstallException | ArchiveException
				| URISyntaxException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
		}
        return null;
    }

    @Override
    protected void done() {
    	latch.countDown();
        if (callback != null) {
            callback.run();
        }
    }
    
    public void stopBackground() {
    	if (workerThread != null && workerThread.isAlive())
    		workerThread.interrupt();
    }
}
