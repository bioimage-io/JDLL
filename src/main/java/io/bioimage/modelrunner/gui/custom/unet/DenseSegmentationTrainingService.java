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

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;

/** Training operations required by the shared dense-segmentation UI. */
public interface DenseSegmentationTrainingService {

    void train(DenseSegmentationTrainingConfig config,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException;

    void close();
}
