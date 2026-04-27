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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.builder.PixiBuilderFactory;
import org.apposed.appose.tool.Pixi;

import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;

public class YoloInstaller {

    private static final String ENV_NAME = DLModelPytorch.COMMON_PYTORCH_ENV_NAME;

    public boolean isEnvironmentInstalled() {
        PixiBuilderFactory builder = new PixiBuilderFactory();
        return builder.canWrap(new File(Pixi.BASE_PATH, ENV_NAME));
    }

    public boolean isModelInstalled(String modelPath) {
        return YoloModelRegistry.isInstalled(modelPath);
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
        PixiEnvironmentSpec env = DLModelPytorch.resolvePytorchEnv();
        PixiEnvironmentManager.installRequirements(env, logConsumer);
    }

    public void installModelWeights(String modelPath, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException {
        if (!YoloModelRegistry.canDownload(modelPath)) {
            throw new IOException("YOLO weights are not installed and cannot be downloaded automatically: " + modelPath);
        }

        File modelFile = new File(modelPath);
        File parent = modelFile.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("Could not create YOLO model directory: " + parent.getAbsolutePath());
        }

        String modelName = modelFile.getName();
        FileDownloader downloader = new FileDownloader(YoloModelRegistry.downloadUrl(modelPath), modelFile, false);
        downloader.setPartialProgressConsumer(progress -> {
            if (logConsumer != null) {
                double percent = Math.round(progress * 1000) / 10.0d;
                logConsumer.accept("Downloading " + modelName + " weights: " + percent + "%");
            }
        });
        downloader.download(Thread.currentThread());

        if (!isModelInstalled(modelPath)) {
            throw new IOException("Model not found or incorrect byte size: " + modelPath);
        }
    }
}
