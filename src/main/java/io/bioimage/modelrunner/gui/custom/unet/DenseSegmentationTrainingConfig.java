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

/** Common values consumed by the shared dense-segmentation training UI. */
public interface DenseSegmentationTrainingConfig {

    String getModelName();
    String getDatasetPath();
    int getEpochs();
    boolean isFineTune();
    String getBaseModelPath();
    String getScratchArchitecture();
    String getModelsDir();
    String getOutputModelDir();
    String getDevice();
}
