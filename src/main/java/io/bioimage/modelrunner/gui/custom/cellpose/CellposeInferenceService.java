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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

/**
 * Maintains the Cellpose Python service and joins frame-wise outputs.
 */
public class CellposeInferenceService {

    private final CellposeInstaller installer;
    private volatile Cellpose model;
    private String loadedModelPath;
    private String loadedDevice;

    public CellposeInferenceService(CellposeInstaller installer) {
        this.installer = installer;
    }

    public <T extends RealType<T> & NativeType<T>,
            R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> run(String modelSelection, RandomAccessibleInterval<T> image,
            int[] channels, Float diameter, String device, Consumer<String> logConsumer)
            throws Exception {
        String modelPath = installer.installAndResolve(modelSelection, logConsumer);
        ensureLoaded(modelPath, device, logConsumer);
        model.setChannels(channels);
        if (diameter == null) {
            model.clearDiameter();
        } else {
            model.setDiameter(diameter.floatValue());
        }
        boolean grayscaleChannels = channels[0] == 0 && channels[1] == 0;
        int inputChannels = !grayscaleChannels || hasExplicitRgbAxis(image) ? 3 : 1;
        int denoisedChannels = channels[0] == 0 ? 1 : 2;
        return runFrames(image, inputChannels, denoisedChannels, diameter, logConsumer);
    }

    private synchronized void ensureLoaded(String modelPath, String device, Consumer<String> logConsumer)
            throws Exception {
        if (model != null && (!modelPath.equals(loadedModelPath) || !device.equals(loadedDevice))) {
            model.close();
            model = null;
        }
        if (model == null || !model.isLoaded()) {
            accept(logConsumer, "Loading Cellpose model.");
            model = Cellpose.init(modelPath, device);
            model.loadModel();
            loadedModelPath = modelPath;
            loadedDevice = device;
            accept(logConsumer, "Cellpose model loaded.");
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> runFrames(RandomAccessibleInterval<T> image, int inputChannels,
            int denoisedChannels, Float diameter, Consumer<String> logConsumer)
            throws RunModelException {
        RandomAccessibleInterval<T> input = addDimsToInput(image, inputChannels);
        long[] dimensions = input.dimensionsAsLongArray();
        int frames = Math.toIntExact(dimensions[3]);
        RandomAccessibleInterval<UnsignedShortType> labels =
                ArrayImgs.unsignedShorts(dimensions[0], dimensions[1], frames);
        RandomAccessibleInterval<UnsignedByteType> flowRgb =
                ArrayImgs.unsignedBytes(dimensions[0], dimensions[1], 3, frames);
        RandomAccessibleInterval<FloatType> flowVectors =
                ArrayImgs.floats(2, dimensions[0], dimensions[1], frames);
        RandomAccessibleInterval<FloatType> probability =
                ArrayImgs.floats(dimensions[0], dimensions[1], frames);
        RandomAccessibleInterval<FloatType> denoised =
                ArrayImgs.floats(dimensions[0], dimensions[1], denoisedChannels, frames);

        for (int frame = 0; frame < frames; frame++) {
            if (Thread.currentThread().isInterrupted()) {
                throw new RunModelException("Cellpose inference cancelled.");
            }
            accept(logConsumer, "Running image " + (frame + 1) + "/" + frames + ".");
            if (diameter != null) {
                model.setDiameter(diameter.floatValue());
            }
            List<Tensor<R>> outputs = model.inference(
                    Tensor.build("input", "xyc", Views.hyperSlice(input, 3, frame)));
            copyOutputs(outputs, labels, flowRgb, flowVectors, probability, denoised, frame);
        }

        List<Tensor<R>> outputs = new ArrayList<Tensor<R>>();
        outputs.add((Tensor<R>) (Tensor) Tensor.build("labels", "xyb", labels));
        outputs.add((Tensor<R>) (Tensor) Tensor.build("flows_0", "xycb", flowRgb));
        outputs.add((Tensor<R>) (Tensor) Tensor.build("flows_1", "cxyb", flowVectors));
        outputs.add((Tensor<R>) (Tensor) Tensor.build("flows_2", "xyb", probability));
        outputs.add((Tensor<R>) (Tensor) Tensor.build("image_dn", "xycb", denoised));
        return outputs;
    }

    private static <T extends RealType<T> & NativeType<T>> void copyOutputs(
            List<Tensor<T>> outputs, RandomAccessibleInterval<UnsignedShortType> labels,
            RandomAccessibleInterval<UnsignedByteType> flowRgb,
            RandomAccessibleInterval<FloatType> flowVectors,
            RandomAccessibleInterval<FloatType> probability,
            RandomAccessibleInterval<FloatType> denoised, int frame) {
        if (outputs.size() != 6) {
            throw new IllegalArgumentException(
                    "Cellpose returned " + outputs.size() + " outputs instead of 6.");
        }
        copy(outputs.get(0).getData(), Views.hyperSlice(labels, 2, frame));
        copy(outputs.get(1).getData(), Views.hyperSlice(flowRgb, 3, frame));
        copy(outputs.get(2).getData(), Views.hyperSlice(flowVectors, 3, frame));
        copy(outputs.get(3).getData(), Views.hyperSlice(probability, 2, frame));
        copy(outputs.get(5).getData(), Views.hyperSlice(denoised, 3, frame));
    }

    private static <S extends RealType<S> & NativeType<S>,
            T extends RealType<T> & NativeType<T>>
    void copy(RandomAccessibleInterval<S> source, RandomAccessibleInterval<T> target) {
        if (!Arrays.equals(source.dimensionsAsLongArray(), target.dimensionsAsLongArray())) {
            throw new IllegalArgumentException("Cellpose output dimensions do not match: "
                    + Arrays.toString(source.dimensionsAsLongArray()) + " vs "
                    + Arrays.toString(target.dimensionsAsLongArray()));
        }
        Cursor<S> sourceCursor = Views.flatIterable(source).cursor();
        Cursor<T> targetCursor = Views.flatIterable(target).cursor();
        while (sourceCursor.hasNext()) {
            targetCursor.next().setReal(sourceCursor.next().getRealDouble());
        }
    }

    private static <T extends RealType<T> & NativeType<T>>
    RandomAccessibleInterval<T> addDimsToInput(RandomAccessibleInterval<T> image, int channels) {
        long[] dimensions = image.dimensionsAsLongArray();
        if (dimensions.length == 2 && channels == 1) {
            return Views.addDimension(Views.addDimension(image, 0, 0), 0, 0);
        }
        if (dimensions.length == 2) {
            throw new IllegalArgumentException("RGB Cellpose channels require a three-channel image.");
        }
        if (dimensions.length == 3 && dimensions[2] == channels) {
            return Views.addDimension(image, 0, 0);
        }
        if (dimensions.length == 3 && channels == 1) {
            return Views.permute(Views.addDimension(image, 0, 0), 2, 3);
        }
        if (dimensions.length == 4 && dimensions[2] == channels) {
            return image;
        }
        if (dimensions.length == 5 && dimensions[2] == channels && dimensions[4] != 1) {
            return Views.hyperSlice(image, 3, 0);
        }
        if (dimensions.length == 5 && dimensions[2] == channels && dimensions[4] == 1) {
            return Views.hyperSlice(Views.permute(image, 3, 4), 3, 0);
        }
        if (dimensions.length == 4 && channels == 1) {
            RandomAccessibleInterval<T> firstChannel = Views.hyperSlice(image, 2, image.min(2));
            return Views.permute(Views.addDimension(firstChannel, 0, 0), 2, 3);
        }
        throw new IllegalArgumentException("Unsupported image dimensions for the selected Cellpose channels.");
    }

    private static <T extends RealType<T> & NativeType<T>>
    boolean hasExplicitRgbAxis(RandomAccessibleInterval<T> image) {
        return image.numDimensions() >= 4 && image.dimension(2) == 3;
    }

    public void cancelCurrentInference() {
        Cellpose activeModel = model;
        if (activeModel != null) {
            activeModel.cancelCurrentInference();
        }
    }

    public synchronized void close() {
        if (model != null) {
            model.close();
            model = null;
        }
        loadedModelPath = null;
        loadedDevice = null;
    }

    private static void accept(Consumer<String> consumer, String message) {
        if (consumer != null) {
            consumer.accept(message);
        }
    }
}
