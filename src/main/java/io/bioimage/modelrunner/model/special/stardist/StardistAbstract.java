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
package io.bioimage.modelrunner.model.special.stardist;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Deprecated compatibility shim for old StarDist callers.
 * <p>
 * New code should use {@link StarDist} directly. This class only keeps static
 * utility entry points while existing callers are migrated.
 */
@Deprecated
public abstract class StardistAbstract extends DLModelPytorchProtected {

	public static StardistAbstract init(String modelPath) throws IOException, BuildException {
		throw new UnsupportedOperationException("Use StarDist.fromFile(modelPath) instead.");
	}

	public static PixiEnvironmentSpec resolvePytorchEnv() {
		return StarDist.resolvePytorchEnv();
	}

	public static boolean isInstalled() {
		return StarDist.isInstalled();
	}

	public static void installDefaultRequirements() throws InterruptedException, BuildException {
		StarDist.installDefaultRequirements();
	}

	public static void installDefaultRequirements(Consumer<String> consumer)
			throws InterruptedException, BuildException {
		StarDist.installDefaultRequirements(consumer);
	}

	public static Map<String, Object> defaultTrainingConfig(int epochs) {
		return StarDist.defaultTrainingConfig(epochs);
	}

	public static Map<String, Object> defaultModelConfig2D() {
		return StarDist.defaultModelConfig2D();
	}

	public static Map<String, Object> defaultModelConfig3D() {
		return StarDist.defaultModelConfig3D();
	}

	public static void train(int epochs, String dataDir, String gtDir, String outputDir,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		StarDist.train(epochs, dataDir, gtDir, outputDir,
				progressConsumer, previewConsumer, logConsumer);
	}

	public static void train(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		StarDist.train(dataDir, gtDir, outputDir, gpu, imageChannels, labelColorMode,
				validFraction, config, progressConsumer, previewConsumer, logConsumer);
	}

	public static String buildTrainingCode(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config) {
		return StarDist.buildTrainingCode(dataDir, gtDir, outputDir, gpu,
				imageChannels, labelColorMode, validFraction, config);
	}

	public void setInferenceProgressConsumer(Consumer<InferenceProgress> progressConsumer) {
		throw new UnsupportedOperationException("Use StarDist directly.");
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> inference(Tensor<T>... inputs) throws RunModelException {
		throw new RunModelException("Use StarDist directly.");
	}
}
