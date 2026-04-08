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
package io.bioimage.modelrunner.gui.workers;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import javax.swing.SwingWorker;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.model.special.stardist.StardistAbstract;

public class InstallEnvWorker extends SwingWorker<Boolean, Void> {

    private final String modelFamily;
    private Consumer<String> consumer;
    private final CountDownLatch latch;
    private final Consumer<Boolean> callback;
    
    private Thread workerThread;

    /**
     * Creates a new InstallEnvWorker.
     *
     * @param descriptor the descriptor parameter.
     * @param latch the latch parameter.
     * @param callback the callback parameter.
     */
    public InstallEnvWorker(ModelDescriptor descriptor, CountDownLatch latch, Consumer<Boolean> callback) {
        this.modelFamily = descriptor.getModelFamily();
        this.latch = latch;
        this.callback = callback;
    }

    /**
     * Creates a new InstallEnvWorker.
     *
     * @param modelFamily the modelFamily parameter.
     * @param latch the latch parameter.
     * @param callback the callback parameter.
     */
    public InstallEnvWorker(String modelFamily, CountDownLatch latch, Consumer<Boolean> callback) {
        this.modelFamily = modelFamily;
        this.latch = latch;
        this.callback = callback;
    }
    
    /**
     * Sets consumer.
     *
     * @param consumer the consumer parameter.
     */
    public void setConsumer(Consumer<String> consumer) {
    	this.consumer = consumer;
    }
    
    /**
     * Gets model family.
     *
     * @return the resulting string.
     */
    public String getModelFamily() {
    	return this.modelFamily;
    }

    /**
     * Gets count down latch.
     *
     * @return the resulting value.
     */
    public CountDownLatch getCountDownLatch() {
    	return this.latch;
    }

    /**
     * Executes do in background.
     *
     * @return the resulting value.
     */
    @Override
    protected Boolean doInBackground() {
    	workerThread = Thread.currentThread();
    	try {
            if (modelFamily.toLowerCase().equals(ModelDescriptor.STARDIST)) {
            	StardistAbstract.installRequirements(consumer);
            } else if (modelFamily.toLowerCase().equals(ModelDescriptor.CELLPOSE)) {
            	Cellpose.installRequirements(consumer);
            } else if (modelFamily.toLowerCase().equals(ModelDescriptor.BIOIMAGEIO))
            	PixiEnvironmentManager.installRequirements(DLModelPytorch.resolvePytorchEnv(), consumer);
		} catch (IOException | RuntimeException e) {
			e.printStackTrace();
			return false;
		} catch (InterruptedException e) {
			return false;
		}
        return true;
    }

    /**
     * Executes done.
     */
    @Override
    protected void done() {
    	Boolean success = false;
    	try {
			success = this.get();
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
    	latch.countDown();
        if (callback != null) {
            callback.accept(success);
        }
    }
    
    /**
     * Executes stop background.
     */
    public void stopBackground() {
    	if (workerThread != null && workerThread.isAlive())
    		workerThread.interrupt();
    }
}
