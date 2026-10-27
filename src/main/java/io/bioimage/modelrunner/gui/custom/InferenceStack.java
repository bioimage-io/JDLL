/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/** Iterates host XYCZT images without copying inputs, keeping Z and time separate. */
public final class InferenceStack<T extends RealType<T> & NativeType<T>> {
    @FunctionalInterface
    public interface Runner<T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> {
        List<Tensor<R>> run(RandomAccessibleInterval<T> input) throws RunModelException;
    }

    private final RandomAccessibleInterval<T> image;
    private final boolean volume;
    private final int originalDimensions;
    private volatile int currentUnit;

    public InferenceStack(RandomAccessibleInterval<T> input, boolean volume) {
        originalDimensions = input.numDimensions();
        if (originalDimensions < 2 || originalDimensions > 5)
            throw new IllegalArgumentException("Expected XY, XYC, XYCZ or XYCZT image data.");
        this.volume = volume;
        RandomAccessibleInterval<T> prepared = Views.zeroMin(input);
        // The compact three-axis form for a volume is XYZ rather than XYC.
        if (volume && originalDimensions == 3)
            prepared = Views.permute(Views.addDimension(prepared, 0, 0), 2, 3);
        while (prepared.numDimensions() < 5) prepared = Views.addDimension(prepared, 0, 0);
        image = prepared;
    }

    public int depth() { return Math.toIntExact(image.dimension(3)); }
    public int frames() { return Math.toIntExact(image.dimension(4)); }
    public int currentTime() { return time(currentUnit); }
    public int size() { return Math.multiplyExact(volume ? 1 : depth(), frames()); }
    public int z(int unit) { return volume ? 0 : unit % depth(); }
    public int time(int unit) { return volume ? unit : unit / depth(); }

    public RandomAccessibleInterval<T> input(int unit) {
        currentUnit = unit;
        RandomAccessibleInterval<T> frame = Views.hyperSlice(image, 4, time(unit));
        return volume ? frame : Views.hyperSlice(frame, 3, z(unit));
    }

    /** Converts per-plane/volume patch events to progress for the entire stack. */
    public Consumer<InferenceProgress> progress(Consumer<InferenceProgress> consumer) {
        if (consumer == null || size() == 1) return consumer;
        return event -> {
            int patches = event.getTotalPatches();
            switch (event.getPhase()) {
                case INFERENCE_START:
                    if (currentUnit == 0) consumer.accept(InferenceProgress.inferenceStart(Math.multiplyExact(patches, size())));
                    break;
                case PATCH_START:
                    consumer.accept(InferenceProgress.patchStart(currentUnit * patches + event.getPatchIndex(),
                            Math.multiplyExact(patches, size())));
                    break;
                case PATCH_END:
                    consumer.accept(InferenceProgress.patchEnd(currentUnit * patches + event.getPatchIndex(),
                            Math.multiplyExact(patches, size())));
                    break;
                case MERGE_START:
                case INFERENCE_END:
                    if (currentUnit == size() - 1) consumer.accept(event);
                    break;
                default:
                    consumer.accept(event);
            }
        };
    }

    public <R extends RealType<R> & NativeType<R>> List<Tensor<R>> run(
            Runner<T, R> runner, BooleanSupplier cancelled) throws RunModelException {
        List<Tensor<R>> reference = null;
        List<List<RandomAccessibleInterval<R>>> images = new ArrayList<>();
        for (int unit = 0; unit < size(); unit++) {
            checkCancelled(cancelled.getAsBoolean());
            List<Tensor<R>> outputs = runner.run(input(unit));
            checkCancelled(cancelled.getAsBoolean());
            if (reference == null) {
                reference = outputs;
                for (int i = 0; i < outputs.size(); i++) images.add(new ArrayList<>());
            }
            if (outputs.size() != reference.size()) throw new RunModelException("Output count changed between stack entries.");
            for (int i = 0; i < outputs.size(); i++) {
                Tensor<R> output = outputs.get(i), first = reference.get(i);
                if (!output.getAxesOrderString().equals(first.getAxesOrderString())
                        || !Arrays.equals(output.getData().dimensionsAsLongArray(), first.getData().dimensionsAsLongArray()))
                    throw new RunModelException("Output geometry changed between stack entries.");
                images.get(i).add(Views.zeroMin(output.getData()));
            }
        }
        List<Tensor<R>> result = new ArrayList<>();
        for (int i = 0; i < images.size(); i++) {
            Tensor<R> first = reference.get(i);
            String axes = first.getAxesOrderString();
            List<RandomAccessibleInterval<R>> frames = new ArrayList<>();
            for (int t = 0; t < frames(); t++) {
                frames.add(!volume && originalDimensions >= 4
                        ? Views.stack(images.get(i).subList(t * depth(), (t + 1) * depth()))
                        : images.get(i).get(t));
            }
            if (!volume && originalDimensions >= 4) axes += "z";
            RandomAccessibleInterval<R> data = originalDimensions == 5 ? Views.stack(frames) : frames.get(0);
            if (originalDimensions == 5) axes += "t";
            result.add(Tensor.build(first.getName(), axes, data));
        }
        return result;
    }

    public static void checkCancelled(boolean cancelled) throws RunModelException {
        if (cancelled || Thread.currentThread().isInterrupted()) throw new RunModelException("Inference cancelled.");
    }
}
