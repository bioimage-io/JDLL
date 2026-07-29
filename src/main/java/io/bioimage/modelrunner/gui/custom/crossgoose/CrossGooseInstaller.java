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

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.special.crossgoose.CrossGoose;

/** Installs the shared PyTorch environment used by Cross-GOOSE. */
public final class CrossGooseInstaller implements ModelInstaller {

    @Override public boolean isEnvironmentInstalled() { return CrossGoose.isInstalled(); }
    @Override public boolean isModelInstalled(String path) { return CrossGooseModelRegistry.isModelPath(path); }

    @Override
    public void installIfNeeded(String modelPath, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException {
        if (!isEnvironmentInstalled()) installEnvironment(logConsumer);
        if (!isModelInstalled(modelPath)) installModelWeights(modelPath, logConsumer);
    }

    @Override
    public void installEnvironment(Consumer<String> logConsumer) throws InterruptedException, BuildException {
        PixiEnvironmentManager.installRequirements(CrossGoose.resolvePytorchEnv(), logConsumer);
    }

    @Override
    public void installModelWeights(String modelPath, Consumer<String> logConsumer) throws IOException {
        File path = modelPath == null ? null : new File(modelPath);
        throw new IOException("Cross-GOOSE model is not a valid local model directory: "
                + (path == null ? "" : path.getAbsolutePath()));
    }
}
