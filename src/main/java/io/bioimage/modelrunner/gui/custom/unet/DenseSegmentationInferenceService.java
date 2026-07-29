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
package io.bioimage.modelrunner.gui.custom.unet;

import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Inference operations required by the shared dense-segmentation UI. */
public interface DenseSegmentationInferenceService {

    <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> run(String modelPath, RandomAccessibleInterval<T> input,
            Consumer<String> logConsumer, boolean usePatchProgressBar, String device)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException;

    <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> runWithProgress(String modelPath, RandomAccessibleInterval<T> input,
            Consumer<InferenceProgress> progressConsumer, String device)
            throws RunModelException, LoadModelException, BuildException, IOException,
            ExecutionException, InterruptedException;

    void setObjectSize(List<Rectangle2D.Double> boxes);
    void cancelCurrentInference();
    void close();
}
