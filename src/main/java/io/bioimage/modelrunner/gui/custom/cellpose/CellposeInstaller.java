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
package io.bioimage.modelrunner.gui.custom.cellpose;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;

/**
 * Installs Cellpose and resolves official model aliases to local weights.
 */
public class CellposeInstaller {

    private static final long LOG_INTERVAL_MILLIS = 50L;

    private final String modelsDirectory;

    public CellposeInstaller(String modelsDirectory) {
        this.modelsDirectory = modelsDirectory == null
                ? new File("models").getAbsolutePath()
                : modelsDirectory;
    }

    public String installAndResolve(String model, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException {
        if (!Cellpose.isInstalled()) {
            accept(logConsumer, "Installing Cellpose environment.");
            PixiEnvironmentManager.installRequirements(Cellpose.resolvePytorchEnv(), logConsumer);
        }
        File custom = model == null ? null : new File(model);
        if (custom != null && custom.isFile()) {
            return custom.getAbsolutePath();
        }
        String installed = Cellpose.findPretrainedModelInstalled(model, modelsDirectory);
        if (installed != null) {
            return installed;
        }
        accept(logConsumer, "Downloading Cellpose " + model + " weights.");
        ThrottledProgress progress = new ThrottledProgress(logConsumer);
        String resolved = Cellpose.donwloadPretrained(model, modelsDirectory,
                value -> progress.accept("Downloading Cellpose " + model + " weights: "
                        + Math.round(value * 1000.0) / 10.0 + "%"));
        progress.acceptNow("Downloading Cellpose " + model + " weights: 100.0%");
        return resolved;
    }

    private static void accept(Consumer<String> consumer, String message) {
        if (consumer != null) {
            consumer.accept(message);
        }
    }

    private static final class ThrottledProgress {
        private final Consumer<String> consumer;
        private long lastUpdate;

        private ThrottledProgress(Consumer<String> consumer) {
            this.consumer = consumer;
        }

        private void accept(String message) {
            long now = System.currentTimeMillis();
            if (consumer != null && now - lastUpdate >= LOG_INTERVAL_MILLIS) {
                lastUpdate = now;
                consumer.accept(message);
            }
        }

        private void acceptNow(String message) {
            if (consumer != null) {
                lastUpdate = System.currentTimeMillis();
                consumer.accept(message);
            }
        }
    }
}
