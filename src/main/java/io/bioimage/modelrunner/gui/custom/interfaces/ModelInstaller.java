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
package io.bioimage.modelrunner.gui.custom.interfaces;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

public interface ModelInstaller {


    /**
     * Returns whether environment installed.
     *
     * @return true if environment installed; false otherwise.
     */
    public boolean isEnvironmentInstalled();

    /**
     * Returns whether model installed.
     *
     * @param modelPath the model path.
     * @return true if model installed; false otherwise.
     */
    public boolean isModelInstalled(String modelPath);

    /**
     * Performs install if needed.
     *
     * @param modelPath the model path.
     * @param logConsumer the log consumer callback.
     * @throws IOException if an I/O error occurs.
     * @throws ExecutionException if an asynchronous operation fails.
     * @throws InterruptedException if the current thread is interrupted.
     * @throws BuildException if the Python environment or service cannot be built.
     */
    public void installIfNeeded(String modelPath, Consumer<String> logConsumer)
    		throws IOException, ExecutionException, InterruptedException, BuildException;

    /**
     * Performs install environment.
     *
     * @param logConsumer the log consumer callback.
     * @throws InterruptedException if the current thread is interrupted.
     * @throws BuildException if the Python environment or service cannot be built.
     */
    public void installEnvironment(Consumer<String> logConsumer)
    		throws InterruptedException, BuildException;

    /**
     * Performs install model weights.
     *
     * @param modelPath the model path.
     * @param logConsumer the log consumer callback.
     * @throws IOException if an I/O error occurs.
     * @throws ExecutionException if an asynchronous operation fails.
     * @throws InterruptedException if the current thread is interrupted.
     */
    public void installModelWeights(String modelPath, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException;
}
