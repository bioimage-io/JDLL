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

    private static final long PROGRESS_UPDATE_INTERVAL_MILLIS = 50L;

    /**
     * Returns whether environment installed.
     *
     * @return true if environment installed; false otherwise.
     */
    public boolean isEnvironmentInstalled() {
        return StarDist.isInstalled();
    }

    /**
     * Returns whether model installed.
     *
     * @param modelPath the model path.
     * @return true if model installed; false otherwise.
     */
    public boolean isModelInstalled(String modelPath) {
        return StardistModelRegistry.isInstalled(modelPath);
    }

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
            throws IOException, ExecutionException, InterruptedException, BuildException {
        if (!isEnvironmentInstalled()) {
            installEnvironment(logConsumer);
        }
        if (!isModelInstalled(modelPath)) {
            installModelWeights(modelPath, logConsumer);
        }
    }

    /**
     * Performs install environment.
     *
     * @param logConsumer the log consumer callback.
     * @throws InterruptedException if the current thread is interrupted.
     * @throws BuildException if the Python environment or service cannot be built.
     */
    public void installEnvironment(Consumer<String> logConsumer) throws InterruptedException, BuildException {
        PixiEnvironmentSpec spec = StarDist.resolvePytorchEnv();
        PixiEnvironmentManager.installRequirements(spec, (str) -> {System.out.println(str);});
    }

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
            throws IOException, ExecutionException, InterruptedException {
        if (!StardistModelRegistry.canDownload(modelPath)) {
            throw new IOException("StarDist weights are not installed and cannot be downloaded automatically: " + modelPath);
        }
        ThrottledLogConsumer progressLog = new ThrottledLogConsumer(logConsumer, PROGRESS_UPDATE_INTERVAL_MILLIS);

        File modelFile = new File(modelPath);
        File parent = modelFile.getParentFile().getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("Could not create StarDist model directory: " + parent.getAbsolutePath());
        }

        String modelName = modelFile.getParentFile().getName();
        File zipFile = new File(parent.getAbsolutePath(), modelName + ".zip");
        FileDownloader downloader = new FileDownloader(StardistModelRegistry.downloadUrl(modelName), zipFile, false);
        downloader.setPartialProgressConsumer(progress -> {
            double percent = roundedPercent(progress);
            progressLog.accept("Downloading " + modelName + " weights: " + percent + "%");
        });
        downloader.download(Thread.currentThread());
        progressLog.acceptNow("Downloading " + modelName + " weights: 100.0%");
        progressLog.acceptNow("Unzipping " + modelName + " weights.");
        File modelFolder = new File(parent.getAbsolutePath(), modelName);
        ZipUtils.unzipFolder(zipFile.getAbsolutePath(), modelFolder.getAbsolutePath(), progress -> {
            double percent = roundedPercent(progress);
            progressLog.accept("Unzipping " + modelName + " weights: " + percent + "%");
        });
        progressLog.acceptNow("Unzipping " + modelName + " weights: 100.0%");

        if (!isModelInstalled(modelPath)) {
            throw new IOException("Model not found or incorrect byte size: " + modelPath);
        }
    }

    private static double roundedPercent(double progress) {
        double bounded = Math.max(0.0d, Math.min(1.0d, progress));
        return Math.round(bounded * 1000) / 10.0d;
    }

    private static final class ThrottledLogConsumer {

        private final Consumer<String> delegate;
        private final long minIntervalMillis;
        private long lastUpdateMillis;

        private ThrottledLogConsumer(Consumer<String> delegate, long minIntervalMillis) {
            this.delegate = delegate;
            this.minIntervalMillis = minIntervalMillis;
        }

        private void accept(String message) {
            if (delegate == null) {
                return;
            }
            long now = System.currentTimeMillis();
            if (now - lastUpdateMillis < minIntervalMillis) {
                return;
            }
            lastUpdateMillis = now;
            delegate.accept(message);
        }

        private void acceptNow(String message) {
            if (delegate == null) {
                return;
            }
            lastUpdateMillis = System.currentTimeMillis();
            delegate.accept(message);
        }
    }
}
