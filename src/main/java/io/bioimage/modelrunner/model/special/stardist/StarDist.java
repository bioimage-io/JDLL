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

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;
import org.apposed.appose.util.Messages;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Unified StarDist entry point.
 * <p>
 * Static factories create the Java object, start the Python service and
 * instantiate the Python StarDist model immediately.
 */
public final class StarDist extends DLModelPytorchProtected {

	public enum Dimensionality {
		TWO_D,
		THREE_D
	}

	public enum SourceType {
		FILE,
		CONFIG,
		DEFAULT
	}

	private static final String MODULE_2D = "StarDist2D";
	private static final String MODULE_3D = "StarDist3D";
	private static final String CONFIG_2D = "Config2D";
	private static final String CONFIG_3D = "Config3D";

	private final Dimensionality dimensionality;
	private final SourceType sourceType;

	private StarDist(String modelPath, Map<String, Object> configMap,
			Dimensionality dimensionality, SourceType sourceType)
			throws BuildException, IOException {
		super(modelPath, modelPath, modelPath, modelPath, configMap);
		this.dimensionality = dimensionality;
		this.sourceType = sourceType;
	}

	private StarDist(String modelName, Map<String, Object> configMap,
			Dimensionality dimensionality, SourceType sourceType, boolean pathlessModel)
			throws BuildException, IOException {
		super(modelName, configMap, pathlessModel);
		this.dimensionality = dimensionality;
		this.sourceType = sourceType;
	}

	public static StarDist fromFile(String modelPath)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = loadModelConfig(modelPath);
		StarDist model = new StarDist(modelPath, config, inferDimensionality(config), SourceType.FILE);
		model.loadModel();
		return model;
	}

	public static StarDist fromConfigJson(String configJsonPath)
			throws IOException, BuildException, LoadModelException {
		if (configJsonPath == null || !new File(configJsonPath).isFile()) {
			throw new IllegalArgumentException("StarDist config JSON does not exist: " + configJsonPath);
		}
		return fromConfig(JSONUtils.load(configJsonPath));
	}

	public static StarDist fromConfig(Map<String, Object> config)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> normalized = normalizedConfig(config);
		StarDist model = new StarDist("stardist-config", normalized,
				inferDimensionality(normalized), SourceType.CONFIG, true);
		model.loadModel();
		return model;
	}

	public static StarDist fromDefault()
			throws IOException, BuildException, LoadModelException {
		return fromDefault2D();
	}

	public static StarDist fromDefault2D()
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = defaultModelConfig2D();
		StarDist model = new StarDist("stardist-default-2d", config,
				Dimensionality.TWO_D, SourceType.DEFAULT, true);
		model.loadModel();
		return model;
	}

	public static StarDist fromDefault3D()
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = defaultModelConfig3D();
		StarDist model = new StarDist("stardist-default-3d", config,
				Dimensionality.THREE_D, SourceType.DEFAULT, true);
		model.loadModel();
		return model;
	}

	public Dimensionality getDimensionality() {
		return dimensionality;
	}

	public SourceType getSourceType() {
		return sourceType;
	}

	public Map<String, Object> getConfig() {
		return new LinkedHashMap<String, Object>(config);
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	Map<String, RandomAccessibleInterval<R>> inference(RandomAccessibleInterval<T> image)
			throws RunModelException {
		try {
			return run(image);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RunModelException(Messages.stackTrace(e));
		} catch (TaskException | IOException e) {
			throw new RunModelException(Messages.stackTrace(e));
		}
	}

	public void trainWithOptions(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> trainingConfig,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		StardistAbstract.train(dataDir, gtDir, outputDir, gpu, imageChannels,
				labelColorMode, validFraction, trainingConfig,
				progressConsumer, previewConsumer, logConsumer);
	}

	public void train(String dataDir, String gtDir, String outputDir, int epochs,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		StardistAbstract.train(dataDir, gtDir, outputDir, true, "grayscale",
				"grayscale", 0.15d, defaultTrainingConfig(epochs),
				progressConsumer, previewConsumer, logConsumer);
	}

	public void train(String dataDir, String outputDir, int epochs,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, null, outputDir, epochs, progressConsumer, previewConsumer, logConsumer);
	}

	@Override
	protected String createImportsCode() {
		if (sourceType == SourceType.FILE) {
			String module = is2D() ? MODULE_2D : MODULE_3D;
			return String.format(LOAD_MODEL_CODE_ABSTRACT, module, module,
					module, module, module, this.name, this.basedir);
		}
		return createModelFromConfigCode();
	}

	@Override
	protected <T extends RealType<T> & NativeType<T>> void checkInput(RandomAccessibleInterval<T> image) {
		long[] dims = image.dimensionsAsLongArray();
		if (is2D()) {
			if (dims.length == 2 && nChannels == 1) {
				return;
			}
			if (dims.length == 3 && dims[2] == nChannels) {
				return;
			}
			throw new IllegalArgumentException("StarDist 2D expects XY or XYC input with "
					+ nChannels + " channel(s).");
		}
		if (dims.length == 3 && nChannels == 1) {
			return;
		}
		if (dims.length == 4 && dims[3] == nChannels) {
			return;
		}
		throw new IllegalArgumentException("StarDist 3D expects XYZ or XYZC input with "
				+ nChannels + " channel(s).");
	}

	@Override
	protected <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() {
		RandomAccessibleInterval<T> mask = shma.getSharedRAI();
		RandomAccessibleInterval<T> maskCopy;
		if (is2D() && nChannels > 1 && mask.dimensionsAsLongArray().length > 2) {
			long[] maxPos = mask.maxAsLongArray();
			maxPos[2] = 0;
			IntervalView<T> maskInterval = Views.interval(mask, mask.minAsLongArray(), maxPos);
			maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(maskInterval),
					Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
		} else {
			maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(mask),
					Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
		}
		shma.close();
		return maskCopy;
	}

	@Override
	public boolean is2D() {
		return dimensionality == Dimensionality.TWO_D;
	}

	@Override
	public boolean is3D() {
		return dimensionality == Dimensionality.THREE_D;
	}

	private String createModelFromConfigCode() {
		String nl = System.lineSeparator();
		String module = is2D() ? MODULE_2D : MODULE_3D;
		String configClass = is2D() ? CONFIG_2D : CONFIG_3D;
		return ""
				+ "import inspect, json, os" + nl
				+ "if 'np' not in globals().keys():" + nl
				+ "  import numpy as np" + nl
				+ "  globals()['np'] = np" + nl
				+ "if 'shared_memory' not in globals().keys():" + nl
				+ "  from multiprocessing import shared_memory" + nl
				+ "  globals()['shared_memory'] = shared_memory" + nl
				+ "from stardist.models import " + module + ", " + configClass + nl
				+ "_config_dict = json.loads(r'''" + TrainingCodeUtils.toJson(config) + "''')" + nl
				+ "_config_dict.pop('axes', None)" + nl
				+ "try:" + nl
				+ "  _config = " + configClass + "(**_config_dict)" + nl
				+ "except TypeError:" + nl
				+ "  _allowed = set(inspect.signature(" + configClass + ".__init__).parameters.keys())" + nl
				+ "  _allowed.discard('self')" + nl
				+ "  _config = " + configClass + "(**{k: v for k, v in _config_dict.items() if k in _allowed})" + nl
				+ "model = " + module + "(_config, name='" + TrainingCodeUtils.py(name) + "', basedir=None)" + nl
				+ "globals()['model'] = model" + nl;
	}

	private static Dimensionality inferDimensionality(Map<String, Object> config) {
		String axes = inferAxes(config, false);
		return axes.indexOf('z') >= 0 ? Dimensionality.THREE_D : Dimensionality.TWO_D;
	}

	private static Map<String, Object> loadModelConfig(String modelPath) throws IOException {
		File path = new File(modelPath);
		File modelDir = path.isFile() ? path.getParentFile() : path;
		File configFile = new File(modelDir, "config.json");
		if (!configFile.isFile() && path.isFile()) {
			String name = path.getName();
			int dot = name.lastIndexOf('.');
			String base = dot < 0 ? name : name.substring(0, dot);
			configFile = new File(modelDir, base + ".json");
		}
		if (configFile.isFile()) {
			return JSONUtils.load(configFile.getAbsolutePath());
		}
		Map<String, Object> config = defaultModelConfig2D();
		String lower = path.getName().toLowerCase();
		if (lower.contains("color") || lower.contains("rgb")) {
			config.put("n_channel_in", 3);
			config.put("axes", "YXC");
		}
		return config;
	}
}
