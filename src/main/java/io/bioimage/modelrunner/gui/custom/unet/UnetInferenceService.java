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

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.special.unet.Unet;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class UnetInferenceService {

    private static final int PATCH_PROGRESS_BAR_WIDTH = 20;

    private final ModelInstaller installer;
    private String loadedModelPath;
    private String loadedDevice;
    private Unet model;

    /**
     * Creates a new UnetInferenceService instance.
     *
     * @param installer the installer.
     */
    public UnetInferenceService(ModelInstaller installer) {
        this.installer = installer;
    }

    /**
     * Runs inference.
     *
     * @param <T> the T type parameter.
     * @param <R> the R type parameter.
     * @param modelPath the model path.
     * @param rai the input RAI.
     * @param logConsumer the log consumer.
     * @param usePatchProgressBar the usePatchProgressBar parameter.
     * @param device the device.
     * @return the output tensors.
     * @throws RunModelException if inference fails.
     * @throws LoadModelException if loading fails.
     * @throws BuildException if environment build fails.
     * @throws IOException if I/O fails.
     * @throws ExecutionException if async install fails.
     * @throws InterruptedException if interrupted.
     */
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> run(String modelPath, RandomAccessibleInterval<T> rai, Consumer<String> logConsumer,
            boolean usePatchProgressBar, String device)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        ensureLoaded(modelPath, normalizeDevice(device), logConsumer,
                progress -> appendProgressLog(progress, logConsumer, usePatchProgressBar));
        return runLoadedModel(rai);
    }

    /**
     * Runs inference with progress.
     *
     * @param <T> the T type parameter.
     * @param <R> the R type parameter.
     * @param modelPath the model path.
     * @param rai the input RAI.
     * @param progressConsumer the progress consumer.
     * @param device the device.
     * @return the output tensors.
     * @throws RunModelException if inference fails.
     * @throws LoadModelException if loading fails.
     * @throws BuildException if environment build fails.
     * @throws IOException if I/O fails.
     * @throws ExecutionException if async install fails.
     * @throws InterruptedException if interrupted.
     */
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> runWithProgress(String modelPath, RandomAccessibleInterval<T> rai,
            Consumer<InferenceProgress> progressConsumer, String device)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        ensureLoaded(modelPath, normalizeDevice(device), null, progressConsumer);
        return runLoadedModel(rai);
    }

    /**
     * Closes resources.
     */
    public void close() {
        if (model != null && model.isLoaded()) {
            model.close();
        }
        model = null;
        loadedModelPath = null;
        loadedDevice = null;
    }

    /**
     * Cancels current inference.
     */
    public void cancelCurrentInference() {
        if (model != null) {
            model.cancelCurrentInference();
        }
    }

    private void ensureLoaded(String modelPath, String device, Consumer<String> logConsumer,
            Consumer<InferenceProgress> progressConsumer)
            throws BuildException, IOException, LoadModelException, ExecutionException, InterruptedException {
        if (loadedModelPath != null && (!loadedModelPath.equals(modelPath) || !device.equals(loadedDevice))) {
            close();
        }
        if (model == null || !model.isLoaded()) {
            installer.installIfNeeded(modelPath, logConsumer);
            model = Unet.fromFile(modelPath, progressConsumer, device);
            loadedModelPath = modelPath;
            loadedDevice = device;
        } else {
            model.setInferenceProgressConsumer(progressConsumer);
        }
    }

    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> runLoadedModel(RandomAccessibleInterval<T> rai) throws RunModelException {
        RandomAccessibleInterval<T> input = addDimsToInput(rai, model.getInputChannels());
        return model.inference(Tensor.build("input", "xycb", input));
    }

    private static <T extends RealType<T> & NativeType<T>>
    RandomAccessibleInterval<T> addDimsToInput(RandomAccessibleInterval<T> rai, int nChannels) {
        long[] dims = rai.dimensionsAsLongArray();
        if (dims.length == 2) {
            return Views.addDimension(Views.addDimension(rai, 0, 0), 0, 0);
        }
        if (dims.length == 3) {
            RandomAccessibleInterval<T> channels = trimChannelsIfNeeded(rai, nChannels);
            return Views.addDimension(channels, 0, 0);
        }
        if (dims.length == 4) {
            return rai;
        }
        if (dims.length == 5) {
            return Views.hyperSlice(rai, 3, 0);
        }
        throw new IllegalArgumentException("Unsupported dimensions for UNet model.");
    }

    private static <T extends RealType<T> & NativeType<T>>
    RandomAccessibleInterval<T> trimChannelsIfNeeded(RandomAccessibleInterval<T> rai, int nChannels) {
        long[] dims = rai.dimensionsAsLongArray();
        if (dims.length < 3 || nChannels <= 0 || dims[2] <= nChannels) {
            return rai;
        }
        long[] min = new long[dims.length];
        long[] max = dims.clone();
        for (int i = 0; i < max.length; i ++) {
            max[i] = rai.min(i) + dims[i] - 1L;
            min[i] = rai.min(i);
        }
        max[2] = min[2] + nChannels - 1L;
        FinalInterval interval = new FinalInterval(min, max);
        IntervalView<T> view = Views.interval(rai, interval);
        return view;
    }

    private static String normalizeDevice(String device) {
        if (device == null) {
            return "cpu";
        }
        String normalized = device.trim().toLowerCase();
        return "cuda".equals(normalized) || "mps".equals(normalized) ? normalized : "cpu";
    }

    private static void appendProgressLog(InferenceProgress progress, Consumer<String> logConsumer,
            boolean usePatchProgressBar) {
        if (progress == null || logConsumer == null) {
            return;
        }
        switch (progress.getPhase()) {
            case MODEL_LOADING:
                logConsumer.accept("Loading model: " + progress.getDetail());
                break;
            case MODEL_LOADED:
                logConsumer.accept("Model loaded.");
                break;
            case INFERENCE_START:
                logConsumer.accept("Starting inference on " + progress.getTotalPatches() + " patch(es).");
                break;
            case PATCH_START:
                logConsumer.accept(usePatchProgressBar
                        ? patchProgressBar(progress.getPatchIndex(), progress.getTotalPatches())
                        : patchProgressText(progress.getPatchIndex(), progress.getTotalPatches()));
                break;
            case TASK_RETRY:
                logConsumer.accept(progress.getDetail());
                break;
            case MERGE_START:
                logConsumer.accept("Merging patch predictions.");
                break;
            case INFERENCE_END:
                logConsumer.accept("Inference finished.");
                break;
            default:
                break;
        }
    }

    private static String patchProgressBar(int patchIndex, int totalPatches) {
        int safeTotal = Math.max(1, totalPatches);
        int safePatch = Math.max(0, Math.min(patchIndex, safeTotal));
        int hashes = (int) Math.floor((safePatch / (double) safeTotal) * PATCH_PROGRESS_BAR_WIDTH);
        StringBuilder builder = new StringBuilder(PATCH_PROGRESS_BAR_WIDTH + 16);
        for (int i = 0; i < PATCH_PROGRESS_BAR_WIDTH; i++) {
            builder.append(i < hashes ? '#' : '.');
        }
        builder.append(' ').append(safePatch).append('/').append(safeTotal);
        return builder.toString();
    }

    private static String patchProgressText(int patchIndex, int totalPatches) {
        int safeTotal = Math.max(1, totalPatches);
        int safePatch = Math.max(0, Math.min(patchIndex, safeTotal));
        return "Patch " + safePatch + "/" + safeTotal;
    }
}
