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
package io.bioimage.modelrunner.gui.custom.stardist;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.special.stardist.StarDist;
import io.bioimage.modelrunner.utils.ZipUtils;

public class StardistInstaller  implements ModelInstaller {

    public boolean isEnvironmentInstalled() {
        return StarDist.isInstalled();
    }

    public boolean isModelInstalled(String modelPath) {
        return StardistModelRegistry.isInstalled(modelPath);
    }

    public void installIfNeeded(String modelPath, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException {
        if (!isEnvironmentInstalled()) {
            installEnvironment(logConsumer);
        }
        if (!isModelInstalled(modelPath)) {
            installModelWeights(modelPath, logConsumer);
        }
    }

    public void installEnvironment(Consumer<String> logConsumer) throws InterruptedException, BuildException {
        PixiEnvironmentSpec spec = StarDist.resolvePytorchEnv();
        PixiEnvironmentManager.installRequirements(spec, (str) -> {System.out.println(str);});
    }

    public void installModelWeights(String modelPath, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException {
        if (!StardistModelRegistry.canDownload(modelPath)) {
            throw new IOException("StarDist weights are not installed and cannot be downloaded automatically: " + modelPath);
        }

        File modelFile = new File(modelPath);
        File parent = modelFile.getParentFile().getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("Could not create StarDist model directory: " + parent.getAbsolutePath());
        }

        String modelName = modelFile.getParentFile().getName();
        File zipFile = new File(parent.getAbsolutePath(), modelName + ".zip");
        FileDownloader downloader = new FileDownloader(StardistModelRegistry.downloadUrl(modelName), zipFile, false);
        downloader.setPartialProgressConsumer(progress -> {
            if (logConsumer != null) {
                double percent = Math.round(progress * 1000) / 10.0d;
                logConsumer.accept("Downloading " + modelName + " weights: " + percent + "%");
            }
        });
        downloader.download(Thread.currentThread());
        if (logConsumer != null) {
            logConsumer.accept("Unzipping " + modelName + " weights.");
        }
        File modelFolder = new File(parent.getAbsolutePath(), modelName);
        ZipUtils.unzipFolder(zipFile.getAbsolutePath(), modelFolder.getAbsolutePath(), progress -> {
            if (logConsumer != null) {
                double percent = Math.round(progress * 1000) / 10.0d;
                logConsumer.accept("Unzipping " + modelName + " weights: " + percent + "%");
            }
        });

        if (!isModelInstalled(modelPath)) {
            throw new IOException("Model not found or incorrect byte size: " + modelPath);
        }
    }
}
