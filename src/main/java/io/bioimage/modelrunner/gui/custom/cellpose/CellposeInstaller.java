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

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.model.special.cellpose.CellposeModels;

/**
 * Installs Cellpose and resolves official model aliases to local weights.
 */
public class CellposeInstaller {

    private final String modelsDirectory;

    public CellposeInstaller(String modelsDirectory) {
        this.modelsDirectory = modelsDirectory == null
                ? new File("models").getAbsolutePath()
                : modelsDirectory;
    }

    public String installAndResolve(String model, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException {
        PixiEnvironmentManager.installRequirements(Cellpose.resolvePytorchEnv(), logConsumer);
        return CellposeModels.resolve(model, modelsDirectory, logConsumer);
    }
}
