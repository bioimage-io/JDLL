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
package io.bioimage.modelrunner.gui.custom.unet;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.special.unet.Unet;

public class UnetInstaller implements ModelInstaller {

    /**
     * Returns whether environment installed.
     *
     * @return true if installed.
     */
    public boolean isEnvironmentInstalled() {
        return Unet.isInstalled();
    }

    /**
     * Returns whether model installed.
     *
     * @param modelPath the model path.
     * @return true if installed.
     */
    public boolean isModelInstalled(String modelPath) {
        return UnetModelRegistry.isModelPath(modelPath);
    }

    /**
     * Installs requirements if needed.
     *
     * @param modelPath the model path.
     * @param logConsumer the log consumer.
     * @throws IOException if model path is invalid.
     * @throws ExecutionException if async install fails.
     * @throws InterruptedException if interrupted.
     * @throws BuildException if the environment build fails.
     */
    public void installIfNeeded(String modelPath, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException {
        if (!isEnvironmentInstalled()) {
            installEnvironment(logConsumer);
        }
        if (!isModelInstalled(modelPath)) {
            installModelWeights(modelPath, logConsumer);
        }
    }

    /**
     * Installs the environment.
     *
     * @param logConsumer the log consumer.
     * @throws InterruptedException if interrupted.
     * @throws BuildException if the environment build fails.
     */
    public void installEnvironment(Consumer<String> logConsumer) throws InterruptedException, BuildException {
        PixiEnvironmentManager.installRequirements(Unet.resolvePytorchEnv(), logConsumer);
    }

    /**
     * UNet currently uses local models only.
     *
     * @param modelPath the model path.
     * @param logConsumer the log consumer.
     * @throws IOException if model path is invalid.
     */
    public void installModelWeights(String modelPath, Consumer<String> logConsumer)
            throws IOException {
        File model = modelPath == null ? null : new File(modelPath);
        throw new IOException("UNet model is not installed or is not a valid local model: "
                + (model == null ? "" : model.getAbsolutePath()));
    }
}
