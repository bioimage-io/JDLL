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

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.detection.Detection;
import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class YoloInferenceService {

    private final YoloInstaller installer;
    private String loadedModelPath;
    private Yolo model;

    public YoloInferenceService(YoloInstaller installer) {
        this.installer = installer;
    }

    public <T extends RealType<T> & NativeType<T>>
    List<Detection> run(String modelPath, RandomAccessibleInterval<T> rai, Consumer<String> logConsumer)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        ensureLoaded(modelPath, logConsumer);
        return runLoadedModel(rai);
    }

    public void close() {
        if (model != null && model.isLoaded()) {
            model.close();
        }
        model = null;
        loadedModelPath = null;
    }

    private void ensureLoaded(String modelPath, Consumer<String> logConsumer)
            throws BuildException, IOException, LoadModelException, ExecutionException, InterruptedException {
        if (loadedModelPath != null && !loadedModelPath.equals(modelPath)) {
            close();
        }
        if (model == null || !model.isLoaded()) {
            installer.installIfNeeded(modelPath, logConsumer);
            model = Yolo.init(modelPath);
            model.loadModel();
            loadedModelPath = modelPath;
        }
    }

    private <T extends RealType<T> & NativeType<T>>
    List<Detection> runLoadedModel(RandomAccessibleInterval<T> rai) throws RunModelException {
        RandomAccessibleInterval<T> input = addDimsToInput(rai,
                rai.dimensionsAsLongArray().length > 2 && rai.dimensionsAsLongArray()[2] == 3 ? 3 : 1);
        List<Tensor<T>> outTensor = model.inference(Tensor.build("input", "xycb", input));
        return Detection.fromBN6Tensor(outTensor.get(0));
    }

    private static <R extends RealType<R> & NativeType<R>>
    RandomAccessibleInterval<R> addDimsToInput(RandomAccessibleInterval<R> rai, int nChannels) {
        long[] dims = rai.dimensionsAsLongArray();
        if (dims.length == 2 && nChannels == 1) {
            return Views.addDimension(Views.addDimension(rai, 0, 0), 0, 0);
        } else if (dims.length == 2) {
            throw new IllegalArgumentException("YOLO expected RGB image and got a grayscale image.");
        } else if (dims.length == 3 && dims[2] == nChannels) {
            return Views.addDimension(rai, 0, 0);
        } else if (dims.length == 3 && nChannels == 1) {
            return Views.permute(Views.addDimension(rai, 0, 0), 2, 3);
        } else if (dims.length >= 3 && dims[2] == 1 && nChannels == 3) {
            throw new IllegalArgumentException("Expected RGB (3 channels) image and got grayscale image (1 channel).");
        } else if (dims.length == 4 && dims[2] == nChannels) {
            return rai;
        } else if (dims.length == 5 && dims[2] == nChannels && dims[4] != 1) {
            return Views.hyperSlice(rai, 3, 0);
        } else if (dims.length == 5 && dims[2] == nChannels && dims[4] == 1) {
            return Views.hyperSlice(Views.permute(rai, 3, 4), 3, 0);
        } else if (dims.length == 4 && dims[2] != nChannels && nChannels == 1) {
            rai = Views.hyperSlice(rai, 2, 0);
            rai = Views.addDimension(rai, 0, 0);
            return Views.permute(rai, 2, 3);
        } else if (dims.length == 5 && dims[2] != nChannels) {
            throw new IllegalArgumentException("Expected grayscale (1 channel) image and got RGB image (3 channels).");
        } else {
            throw new IllegalArgumentException("Unsupported dimensions for YOLO model.");
        }
    }
}
