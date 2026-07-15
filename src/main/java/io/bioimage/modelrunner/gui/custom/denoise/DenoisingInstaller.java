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
package io.bioimage.modelrunner.gui.custom.denoise;

import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;

/** Installs the shared PyTorch environment used by {@code jdll-denoise}. */
public final class DenoisingInstaller {

    public boolean isInstalled() {
        return PixiEnvironmentManager.isInstalled(DLModelPytorchProtected.resolvePytorchEnv());
    }

    public void installIfNeeded(Consumer<String> consumer) throws InterruptedException, BuildException {
        if (!isInstalled()) {
            PixiEnvironmentManager.installRequirements(DLModelPytorchProtected.resolvePytorchEnv(), consumer);
        }
    }
}
