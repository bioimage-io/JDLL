/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.crossgoose;

import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationInferenceService;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.special.crossgoose.CrossGoose;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/** Keeps a Cross-GOOSE model loaded across inference calls. */
public final class CrossGooseInferenceService implements DenseSegmentationInferenceService {

    private final CrossGooseInstaller installer;
    private CrossGoose model;
    private String loadedPath;
    private String loadedDevice;
    private Rectangle2D.Double objectReference;

    public CrossGooseInferenceService(CrossGooseInstaller installer) {
        this.installer = installer;
    }

    @Override
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> run(String modelPath, RandomAccessibleInterval<T> input,
            Consumer<String> logConsumer, boolean progressBar, String device)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        ensureLoaded(modelPath, normalizeDevice(device), logConsumer,
                progress -> appendProgress(progress, logConsumer));
        applyObjectSize();
        return infer(input);
    }

    @Override
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> runWithProgress(String modelPath, RandomAccessibleInterval<T> input,
            Consumer<InferenceProgress> progressConsumer, String device)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException {
        ensureLoaded(modelPath, normalizeDevice(device), null, progressConsumer);
        applyObjectSize();
        return infer(input);
    }

    @Override public void setObjectSize(List<Rectangle2D.Double> boxes) {
        objectReference = boxes == null || boxes.isEmpty() ? null : boxes.get(0);
        applyObjectSize();
    }

    @Override public void cancelCurrentInference() {
        if (model != null) model.cancelCurrentInference();
    }

    @Override public void close() {
        if (model != null && model.isLoaded()) model.close();
        model = null;
        loadedPath = null;
        loadedDevice = null;
    }

    private void ensureLoaded(String path, String device, Consumer<String> logConsumer,
            Consumer<InferenceProgress> progressConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, LoadModelException {
        if (loadedPath != null && (!loadedPath.equals(path) || !device.equals(loadedDevice))) close();
        if (model == null || !model.isLoaded()) {
            installer.installIfNeeded(path, logConsumer);
            model = CrossGoose.fromFile(path, progressConsumer, device);
            loadedPath = path;
            loadedDevice = device;
        } else {
            model.setInferenceProgressConsumer(progressConsumer);
        }
    }

    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> infer(RandomAccessibleInterval<T> input) throws RunModelException {
        RandomAccessibleInterval<T> prepared = addDimensions(input);
        return model.inference(Tensor.build("input", "xycb", prepared));
    }

    private static <T extends RealType<T> & NativeType<T>>
    RandomAccessibleInterval<T> addDimensions(RandomAccessibleInterval<T> input) {
        if (input.numDimensions() == 2) {
            return Views.addDimension(Views.addDimension(input, 0, 0), 0, 0);
        }
        if (input.numDimensions() == 3) {
            RandomAccessibleInterval<T> channels = input.dimension(2) > 2
                    ? Views.interval(input, new FinalInterval(
                            new long[] {input.min(0), input.min(1), input.min(2)},
                            new long[] {input.max(0), input.max(1), input.min(2) + 1}))
                    : input;
            return Views.addDimension(channels, 0, 0);
        }
        if (input.numDimensions() == 4) return input;
        if (input.numDimensions() == 5) return Views.hyperSlice(input, 3, input.min(3));
        throw new IllegalArgumentException("Cross-GOOSE expects a 2D XY or XYC image.");
    }

    private static String normalizeDevice(String value) {
        String device = value == null ? "cpu" : value.trim().toLowerCase();
        return "cuda".equals(device) || "mps".equals(device) ? device : "cpu";
    }

    private void applyObjectSize() {
        if (model == null) return;
        if (objectReference == null || objectReference.width <= 0.0 || objectReference.height <= 0.0) {
            model.setObjectSize(null);
            return;
        }
        double area = objectReference.width * objectReference.height;
        model.setObjectSize(2.0 * Math.sqrt(area / Math.PI));
    }

    private static void appendProgress(InferenceProgress progress, Consumer<String> logConsumer) {
        if (progress == null || logConsumer == null) return;
        switch (progress.getPhase()) {
            case MODEL_LOADING: logConsumer.accept("Loading model: " + progress.getDetail()); break;
            case MODEL_LOADED: logConsumer.accept("Model loaded."); break;
            case INFERENCE_START:
                logConsumer.accept("Starting inference on " + progress.getTotalPatches() + " patch(es).");
                break;
            case PATCH_START:
                logConsumer.accept("Patch " + progress.getPatchIndex() + "/" + progress.getTotalPatches());
                break;
            case MERGE_START: logConsumer.accept("Merging patch predictions."); break;
            case INFERENCE_END: logConsumer.accept("Inference finished."); break;
            case TASK_RETRY: logConsumer.accept(progress.getDetail()); break;
            default: break;
        }
    }
}
