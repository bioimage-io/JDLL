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
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

/** Maintains the Python service; a Run owns immutable settings for one image/stack/folder. */
public class CellposeInferenceService {
    private final CellposeInstaller installer;
    private volatile Cellpose model;
    private String loadedModelPath;
    private String loadedDevice;

    public static final class Run {
        private final String modelSelection;
        private final int[] channels;
        private final String device;
        private final boolean intermediates;
        private final Float diameter;
        private boolean diameterLogged;

        public Run(String modelSelection, int[] channels, Float diameter, String device, boolean intermediates) {
            if (modelSelection == null || modelSelection.trim().isEmpty())
                throw new IllegalArgumentException("Please select a Cellpose model.");
            CellposeChannels.validate(channels, Integer.MAX_VALUE);
            if (diameter != null && (!Float.isFinite(diameter) || diameter <= 0))
                throw new IllegalArgumentException("Diameter must be finite and positive.");
            this.modelSelection = modelSelection;
            this.channels = channels.clone();
            this.diameter = diameter;
            this.device = device;
            this.intermediates = intermediates;
        }

        public Float getDiameter() { return diameter; }
    }

    public CellposeInferenceService(CellposeInstaller installer) { this.installer = installer; }

    /** Legacy convenience call: each invocation is a separate run. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> run(String modelSelection, RandomAccessibleInterval<T> image,
            int[] channels, Float diameter, String device, Consumer<String> log) throws Exception {
        return (List) run(new Run(modelSelection, channels, diameter, device, true), image, log);
    }

    /** Inputs have explicit X,Y,C,Z,T axes, with trailing singleton axes optional. */
    public <T extends RealType<T> & NativeType<T>> List<Tensor<FloatType>> run(
            Run run, RandomAccessibleInterval<T> image, Consumer<String> log) throws Exception {
        RandomAccessibleInterval<T> input = canonicalInput(image);
        CellposeChannels.validate(run.channels, input.dimension(2));
        String path = installer.installAndResolve(run.modelSelection, log);
        ensureLoaded(path, run.device, log);
        Cellpose active = model;
        active.setChannels(new int[] {1, run.channels[1] == 0 ? 0 : 2});
        active.setIncludeIntermediateOutputs(run.intermediates);
        active.setLogConsumer(message -> {
            if (run.diameter == null || !run.diameterLogged) accept(log, message);
        });
        try {
            return runFrames(active, run, input, log);
        } finally {
            active.setLogConsumer(null);
        }
    }

    protected Cellpose createModel(String path, String device) throws Exception {
        return Cellpose.init(path, device);
    }

    private synchronized void ensureLoaded(String path, String device, Consumer<String> log) throws Exception {
        if (model != null && (!path.equals(loadedModelPath) || !device.equals(loadedDevice))) {
            model.close();
            model = null;
        }
        if (model == null || !model.isLoaded()) {
            accept(log, "Loading Cellpose model.");
            model = createModel(path, device);
            model.loadModel();
            loadedModelPath = path;
            loadedDevice = device;
            accept(log, "Cellpose model loaded.");
        }
    }

    private <T extends RealType<T> & NativeType<T>> List<Tensor<FloatType>> runFrames(
            Cellpose model, Run run, RandomAccessibleInterval<T> input, Consumer<String> log) throws RunModelException {
        long depth = input.dimension(3), frames = input.dimension(4);
        List<Tensor<FloatType>> result = new ArrayList<>();
        int[] indices = run.intermediates ? new int[] {0, 1, 2, 3, 5} : new int[] {0};
        String[] names = {"labels", "flows_0", "flows_1", "flows_2", "image_dn"};
        String[] axes = {"xy", "xyc", "cxy", "xy", "xyc"};
        for (long t = 0; t < frames; t++) {
            for (long z = 0; z < depth; z++) {
                if (Thread.currentThread().isInterrupted()) throw new RunModelException("Cellpose inference cancelled.");
                accept(log, "Running plane " + (t * depth + z + 1) + "/" + (depth * frames) + ".");
                if (run.diameter == null) model.clearDiameter();
                else model.setDiameter(run.diameter);
                RandomAccessibleInterval<T> plane = Views.hyperSlice(Views.hyperSlice(input, 4, t), 3, z);
                List<Tensor<FloatType>> outputs = model.inference(
                        Tensor.build("input", "xyc", CellposeChannels.pack(plane, run.channels)));
                if (outputs.size() != (run.intermediates ? 6 : 1))
                    throw new IllegalArgumentException("Unexpected Cellpose output count: " + outputs.size());
                run.diameterLogged = true;
                for (int i = 0; i < indices.length; i++) {
                    RandomAccessibleInterval<FloatType> source = outputs.get(indices[i]).getData();
                    if (result.size() <= i) {
                        long[] shape = Arrays.copyOf(source.dimensionsAsLongArray(), source.numDimensions() + 2);
                        // Cellpose omits the auxiliary denoised channel when that plane is constant.
                        if (i == 4 && run.channels[1] != 0) shape[2] = 2;
                        shape[shape.length - 2] = depth;
                        shape[shape.length - 1] = frames;
                        result.add(Tensor.build(names[i], axes[i] + "zt", ArrayImgs.floats(shape)));
                    }
                    RandomAccessibleInterval<FloatType> target = result.get(i).getData();
                    target = Views.hyperSlice(Views.hyperSlice(target, target.numDimensions() - 1, t),
                            target.numDimensions() - 2, z);
                    if (i == 4 && source.dimension(2) == 1 && target.dimension(2) == 2)
                        target = Views.interval(target, new long[] {0, 0, 0},
                                new long[] {target.dimension(0) - 1, target.dimension(1) - 1, 0});
                    copy(source, target);
                }
            }
        }
        return result;
    }

    static <S extends RealType<S> & NativeType<S>, T extends RealType<T> & NativeType<T>>
    void copy(RandomAccessibleInterval<S> source, RandomAccessibleInterval<T> target) {
        if (!Arrays.equals(source.dimensionsAsLongArray(), target.dimensionsAsLongArray()))
            throw new IllegalArgumentException("Cellpose output dimensions do not match: "
                    + Arrays.toString(source.dimensionsAsLongArray()) + " vs " + Arrays.toString(target.dimensionsAsLongArray()));
        Cursor<S> from = Views.flatIterable(source).cursor();
        Cursor<T> to = Views.flatIterable(target).cursor();
        while (from.hasNext()) to.next().setReal(from.next().getRealDouble());
    }

    static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> canonicalInput(
            RandomAccessibleInterval<T> image) {
        if (image.numDimensions() < 2 || image.numDimensions() > 5)
            throw new IllegalArgumentException("Expected X,Y,C,Z,T axes, with optional trailing singleton axes.");
        RandomAccessibleInterval<T> result = Views.zeroMin(image);
        while (result.numDimensions() < 5) result = Views.addDimension(result, 0, 0);
        return result;
    }

    public void cancelCurrentInference() {
        Cellpose active = model;
        if (active != null) active.cancelCurrentInference();
    }

    public synchronized void close() {
        if (model != null) { model.close(); model = null; }
        loadedModelPath = null;
        loadedDevice = null;
    }

    private static void accept(Consumer<String> consumer, String message) {
        if (consumer != null) consumer.accept(message);
    }
}
