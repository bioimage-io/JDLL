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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.model.special.yolo.YoloTrainingProgress;
import io.bioimage.modelrunner.model.special.yolo.YoloValidationPreview;

public class YoloTrainingService {

    private final YoloInstaller installer;

    public YoloTrainingService(YoloInstaller installer) {
        this.installer = installer;
    }

    public void train(YoloTrainingConfig config,
            Consumer<YoloTrainingProgress> progressConsumer,
            Consumer<YoloValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        validate(config);
        if (!installer.isEnvironmentInstalled()) {
            installer.installEnvironment(logConsumer);
        }
        if (config.isFineTune() && !installer.isModelInstalled(config.getBaseModelPath())) {
            installer.installModelWeights(config.getBaseModelPath(), logConsumer);
        }
        Yolo.train(config.getEpochs(), config.getBaseModelPath(), config.getDatasetYamlPath(),
                config.getOutputWeightsPath(), config.getImageSize(), config.getPreviewEpochPeriod(),
                progressConsumer, previewConsumer, logConsumer);
    }

    private static void validate(YoloTrainingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Training configuration cannot be null.");
        }
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a name for the YOLO model.");
        }
        if (config.getModelName().contains("/") || config.getModelName().contains("\\")
                || config.getModelName().contains("..")) {
            throw new IllegalArgumentException("The YOLO model name cannot contain path separators or '..'.");
        }
        if (config.getDatasetYamlPath() == null || config.getDatasetYamlPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide the YOLO dataset data.yaml path.");
        }
        File datasetYaml = new File(config.getDatasetYamlPath());
        if (!datasetYaml.isFile()) {
            throw new IllegalArgumentException("The training dataset must point to an existing YOLO data.yaml file: "
                    + config.getDatasetYamlPath());
        }
        String yamlName = datasetYaml.getName().toLowerCase();
        if (!yamlName.endsWith(".yaml") && !yamlName.endsWith(".yml")) {
            throw new IllegalArgumentException("The training dataset must be a YOLO YAML file: "
                    + config.getDatasetYamlPath());
        }
        if (config.getEpochs() <= 0) {
            throw new IllegalArgumentException("The number of epochs must be greater than zero.");
        }
        if (config.getImageSize() <= 0) {
            throw new IllegalArgumentException("The YOLO image size must be greater than zero.");
        }
        if (config.isFineTune() && (config.getBaseModelPath() == null || config.getBaseModelPath().trim().isEmpty())) {
            throw new IllegalArgumentException("Please select a base YOLO model for fine tuning.");
        }
    }
}
