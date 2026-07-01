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

import java.awt.Rectangle;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.LongStream;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentResolver;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.python.methods.ConvertDims;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.model.tiling.TileInfo;
import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.model.tiling.merger.DenseMerger;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.system.GpuCompatibility;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.transformations.ScaleRangeTransformation;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * Unified StarDist model entry point.
 */
public final class StarDist extends DLModelPytorchProtected {

	public enum Dimensionality {
		TWO_D,
		THREE_D
	}

	private static final String PIXI_TOML = "tomls/stardist-pixi.toml";
	private static final String COMMON_STARDIST_ENV_NAME = "stardist-jdll";
	private static final String STARDIST_CUDA_VERSION = "11.2";

	private static final long DEFAULT_DENSE_TILE_XY = 512L;
	private static final long DEFAULT_DENSE_OUTPUT_HALO_XY = 96L;
	private static final String DENSE_OUTPUT_AXES = "byxc";
	private static final double MAX_OBJECT_TILE_AREA_RATIO = 0.1d;
	private static final double MIN_OBJECT_TILE_AREA_RATIO = 0.00018692d;
	private static final double IDEAL_OBJECT_TILE_AREA_RATIO = 0.04d;

	private static final String LOAD_MODEL_CODE_2D = ""
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  task.export(os=os)" + System.lineSeparator()
			+ "if 'contextlib' not in globals().keys():" + System.lineSeparator()
			+ "  import contextlib" + System.lineSeparator()
			+ "  task.export(contextlib=contextlib)" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  task.export(shared_memory=shared_memory)" + System.lineSeparator()
			+ "if 'tf' not in globals().keys():" + System.lineSeparator()
			+ "  import tensorflow as tf" + System.lineSeparator()
			+ "  task.export(tf=tf)" + System.lineSeparator()
			+ "_jdll_requested_device = '%s'" + System.lineSeparator()
			+ "_jdll_tf_device = '/CPU:0'" + System.lineSeparator()
			+ "try:" + System.lineSeparator()
			+ "  if _jdll_requested_device == 'cpu':" + System.lineSeparator()
			+ "    try:" + System.lineSeparator()
			+ "      tf.config.set_visible_devices([], 'GPU')" + System.lineSeparator()
			+ "    except Exception:" + System.lineSeparator()
			+ "      pass" + System.lineSeparator()
			+ "  elif _jdll_requested_device in ('cuda', 'mps', 'gpu'):" + System.lineSeparator()
			+ "    _jdll_gpus = tf.config.list_physical_devices('GPU')" + System.lineSeparator()
			+ "    if _jdll_gpus:" + System.lineSeparator()
			+ "      for _jdll_gpu in _jdll_gpus:" + System.lineSeparator()
			+ "        try:" + System.lineSeparator()
			+ "          tf.config.experimental.set_memory_growth(_jdll_gpu, True)" + System.lineSeparator()
			+ "        except Exception:" + System.lineSeparator()
			+ "          pass" + System.lineSeparator()
			+ "      _jdll_tf_device = '/GPU:0'" + System.lineSeparator()
			+ "except Exception:" + System.lineSeparator()
			+ "  _jdll_tf_device = '/CPU:0'" + System.lineSeparator()
			+ "task.export(_jdll_tf_device=_jdll_tf_device)" + System.lineSeparator()
			+ "if 'StarDist2D' not in globals().keys():" + System.lineSeparator()
			+ "  from stardist.models import StarDist2D, Config2D" + System.lineSeparator()
			+ "  from stardist.nms import non_maximum_suppression" + System.lineSeparator()
			+ "  from stardist.geometry import polygons_to_label" + System.lineSeparator()
			+ "  task.export(StarDist2D=StarDist2D)" + System.lineSeparator()
			+ "  task.export(Config2D=Config2D)" + System.lineSeparator()
			+ "  task.export(non_maximum_suppression=non_maximum_suppression)" + System.lineSeparator()
			+ "  task.export(polygons_to_label=polygons_to_label)" + System.lineSeparator()
			+ "def _jdll_stardist_config(values):" + System.lineSeparator()
			+ "  values = dict(values or {})" + System.lineSeparator()
			+ "  return Config2D(**values)" + System.lineSeparator()
			+ "def _jdll_load_stardist(source, config):" + System.lineSeparator()
			+ "  if source is None:" + System.lineSeparator()
			+ "    return StarDist2D(_jdll_stardist_config(config), name=None, basedir=None)" + System.lineSeparator()
			+ "  source = os.path.abspath(source)" + System.lineSeparator()
			+ "  model_dir = source if os.path.isdir(source) else os.path.dirname(source)" + System.lineSeparator()
			+ "  model = StarDist2D(None, name=os.path.basename(model_dir), basedir=os.path.dirname(model_dir))" + System.lineSeparator()
			+ "  if os.path.isfile(source) and source.lower().endswith('.h5'):" + System.lineSeparator()
			+ "    model.keras_model.load_weights(source)" + System.lineSeparator()
			+ "  return model" + System.lineSeparator()
			+ "with open(os.devnull, 'w') as _stardist_quiet, contextlib.redirect_stdout(_stardist_quiet), contextlib.redirect_stderr(_stardist_quiet):" + System.lineSeparator()
			+ "  with tf.device(_jdll_tf_device):" + System.lineSeparator()
			+ "    " + MODEL_VAR_NAME + " = _jdll_load_stardist(%s, %s)" + System.lineSeparator()
			+ "task.export(" + MODEL_VAR_NAME + "=" + MODEL_VAR_NAME + ")" + System.lineSeparator();

	private final String mpkPath;
	private final int nChannels;
	private final Dimensionality dimensionality;
	private final Map<String, Object> config;
	protected Rectangle objectSize;

	private Double threshold = null;
	private StarDist(String modelIdentity, Map<String, Object> config,
			Dimensionality dimensionality, Consumer<InferenceProgress> inferenceProgressConsumer, String device) throws IOException {
		super(modelIdentity, modelIdentity, modelIdentity, modelIdentity, config, true, device);
		File identityFile = new File(modelIdentity);
		File parent = identityFile.getParentFile();
		modelFolder = parent == null ? identityFile.getAbsoluteFile().getParent() : parent.getAbsolutePath();
		this.mpkPath = modelIdentity;
		this.config = normalizedConfig(config);
		this.nChannels = inferNChannels(this.config);
		this.dimensionality = dimensionality;
		this.environmentSpec = resolvePytorchEnv();
		super.setInferenceProgressConsumer(inferenceProgressConsumer);
	}

	/**
	 * Creates a StarDist from the file.
	 *
	 * @param modelPath the model path.
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromFile(String modelPath, Consumer<InferenceProgress> inferenceProgressConsumer)
			throws IOException, BuildException, LoadModelException {
		return fromFile(modelPath, inferenceProgressConsumer, null);
	}

	/**
	 * Creates a StarDist from the file.
	 *
	 * @param modelPath the model path.
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @param device the device.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromFile(String modelPath, Consumer<InferenceProgress> inferenceProgressConsumer, String device)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = loadModelConfig(modelPath);
		String modelIdentity = resolveModelIdentityFile(modelPath).getAbsolutePath();
		StarDist model = new StarDist(modelIdentity, config, inferDimensionality(config), inferenceProgressConsumer, device);
		model.loadModel();
		return model;
	}

	/**
	 * Creates a StarDist from the config JSON.
	 *
	 * @param configJsonPath the config JSON path.
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromConfigJson(String configJsonPath, Consumer<InferenceProgress> inferenceProgressConsumer)
			throws IOException, BuildException, LoadModelException {
		return fromConfigJson(configJsonPath, inferenceProgressConsumer, null);
	}

	/**
	 * Creates a StarDist from the config JSON.
	 *
	 * @param configJsonPath the config JSON path.
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @param device the device.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromConfigJson(String configJsonPath, Consumer<InferenceProgress> inferenceProgressConsumer, String device)
			throws IOException, BuildException, LoadModelException {
		if (configJsonPath == null || !new File(configJsonPath).isFile()) {
			throw new IllegalArgumentException("StarDist config JSON does not exist: " + configJsonPath);
		}
		return fromConfig(JSONUtils.load(configJsonPath), inferenceProgressConsumer, device);
	}

	/**
	 * Creates a StarDist from the config.
	 *
	 * @param config the config.
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromConfig(Map<String, Object> config, Consumer<InferenceProgress> inferenceProgressConsumer)
			throws IOException, BuildException, LoadModelException {
		return fromConfig(config, inferenceProgressConsumer, null);
	}

	/**
	 * Creates a StarDist from the config.
	 *
	 * @param config the config.
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @param device the device.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromConfig(Map<String, Object> config, Consumer<InferenceProgress> inferenceProgressConsumer, String device)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> normalized = normalizedConfig(config);
		StarDist model = new StarDist("stardist-config", normalized,
				inferDimensionality(normalized), inferenceProgressConsumer, device);
		model.loadModel();
		return model;
	}

	/**
	 * Creates a StarDist from the default.
	 *
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromDefault(Consumer<InferenceProgress> inferenceProgressConsumer)
			throws IOException, BuildException, LoadModelException {
		return fromDefault(inferenceProgressConsumer, null);
	}

	/**
	 * Creates a StarDist from the default.
	 *
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @param device the device.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromDefault(Consumer<InferenceProgress> inferenceProgressConsumer, String device)
			throws IOException, BuildException, LoadModelException {
		return fromDefault2D(inferenceProgressConsumer, device);
	}

	/**
	 * Creates a StarDist from the default2 d.
	 *
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromDefault2D(Consumer<InferenceProgress> inferenceProgressConsumer)
			throws IOException, BuildException, LoadModelException {
		return fromDefault2D(inferenceProgressConsumer, null);
	}

	/**
	 * Creates a StarDist from the default2 d.
	 *
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @param device the device.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromDefault2D(Consumer<InferenceProgress> inferenceProgressConsumer, String device)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = defaultModelConfig2D();
		StarDist model = new StarDist("stardist-default-2d", config,
				Dimensionality.TWO_D, inferenceProgressConsumer, device);
		model.loadModel();
		return model;
	}

	/**
	 * Creates a StarDist from the default3 d.
	 *
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromDefault3D(Consumer<InferenceProgress> inferenceProgressConsumer)
			throws IOException, BuildException, LoadModelException {
		return fromDefault3D(inferenceProgressConsumer, null);
	}

	/**
	 * Creates a StarDist from the default3 d.
	 *
	 * @param inferenceProgressConsumer the inference progress consumer callback.
	 * @param device the device.
	 * @return the created star dist.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public static StarDist fromDefault3D(Consumer<InferenceProgress> inferenceProgressConsumer, String device)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = defaultModelConfig3D();
		StarDist model = new StarDist("stardist-default-3d", config,
				Dimensionality.THREE_D, inferenceProgressConsumer, device);
		model.loadModel();
		return model;
	}

	/**
	 * Sets the object size.
	 *
	 * @param size the size.
	 */
	public void setObjectSize(Rectangle size) {
		this.objectSize = size;
	}

	/**
	 * Returns the dimensionality.
	 *
	 * @return the dimensionality.
	 */
	public Dimensionality getDimensionality() {
		return dimensionality;
	}

	/**
	 * Returns the config.
	 *
	 * @return the config.
	 */
	public Map<String, Object> getConfig() {
		return new LinkedHashMap<String, Object>(config);
	}

	/**
	 * Returns whether is2 d.
	 *
	 * @return true if is2 d; false otherwise.
	 */
	public boolean is2D() {
		return dimensionality == Dimensionality.TWO_D;
	}

	/**
	 * Returns whether is3 d.
	 *
	 * @return true if is3 d; false otherwise.
	 */
	public boolean is3D() {
		return dimensionality == Dimensionality.THREE_D;
	}

	/**
	 * Returns the n channels.
	 *
	 * @return the n channels.
	 */
	public int getNChannels() {
		return nChannels;
	}

	/**
	 * Sets the threshold.
	 *
	 * @param threshold the threshold.
	 */
	public void setThreshold(Double threshold) {
		this.threshold = threshold == null || !Double.isFinite(threshold)
				? null
				: Math.max(0.0d, Math.min(1.0d, threshold));
	}


	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param gpu whether to use GPU.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param trainingConfig the training config.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public void trainWithOptions(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> trainingConfig,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, gpu, imageChannels, labelColorMode,
				validFraction, trainingConfig, progressConsumer, previewConsumer, logConsumer);
	}

	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param epochs the epochs.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public void train(String dataDir, String gtDir, String outputDir, int epochs,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, true, "grayscale", "grayscale", 0.15d,
				defaultTrainingConfig(epochs), progressConsumer, previewConsumer, logConsumer);
	}

	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param outputDir the output directory.
	 * @param epochs the epochs.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public void train(String dataDir, String outputDir, int epochs,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, null, outputDir, epochs, progressConsumer, previewConsumer, logConsumer);
	}

	/**
	 * Returns the tile maker.
	 *
	 * @param <T> the T type parameter.
	 * @param <R> the R type parameter.
	 * @param inputs the inputs to process.
	 * @return the tile maker.
	 */
	@Override
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	Merger<Tensor<T>, Tensor<R>> getTileMaker(final List<Tensor<T>> inputs) {
		if (inputs == null || inputs.isEmpty()) {
			throw new IllegalArgumentException("StarDist tiling needs at least one input tensor.");
		}
		List<Tensor<T>> tiledInputs = new ArrayList<Tensor<T>>(inputs);
		Tensor<T> referenceInput = firstSpatialInput(tiledInputs);
		if (referenceInput == null) {
			throw new IllegalArgumentException("StarDist tiling needs one input tensor with x and y axes.");
		}
		String originalAxes = referenceInput.getAxesOrderString().toLowerCase();
		long[] originalDims = referenceInput.getData().dimensionsAsLongArray();
		long originalHeight = axisSize(originalDims, originalAxes, 'y');
		long originalWidth = axisSize(originalDims, originalAxes, 'x');
		double scale = objectScale(referenceInput);
		if (needsResize(scale)) {
			for (int i = 0; i < tiledInputs.size(); i ++) {
				Tensor<T> input = tiledInputs.get(i);
				if (hasSpatialAxes(input)) {
					tiledInputs.set(i, resizeTensor(input, scale));
				}
			}
			referenceInput = firstSpatialInput(tiledInputs);
		}
		applyInputNormalization(tiledInputs);
		List<TileInfo> inputInfo = new ArrayList<TileInfo>();
		for (Tensor<T> input : tiledInputs) {
			if (hasSpatialAxes(input)) {
				inputInfo.add(createInputTileInfo(input));
			}
		}
		TileMaker tileMaker = TileMaker.build(inputInfo, createDenseOutputTileInfo(referenceInput));
		DenseMerger<T, R> merger = new DenseMerger<T, R>(tileMaker);
		String referenceAxes = referenceInput.getAxesOrderString().toLowerCase();
		long[] referenceDims = referenceInput.getData().dimensionsAsLongArray();
		long imageHeight = axisSize(referenceDims, referenceAxes, 'y');
		long imageWidth = axisSize(referenceDims, referenceAxes, 'x');
		merger.addCallback(reconstructed -> runStardistNms(reconstructed, imageHeight, imageWidth));
		merger.addCallback(reconstructed -> restoreOriginalScale(reconstructed, originalHeight, originalWidth));
		merger.configure(tiledInputs);
		return merger;
	}

	private <R extends RealType<R> & NativeType<R>> List<Tensor<R>> runStardistNms(
			final List<Tensor<R>> reconstructed, final long imageHeight, final long imageWidth) {
		if (reconstructed == null || reconstructed.size() < 2) {
			return reconstructed;
		}
		try {
			String probName = "prob_" + java.util.UUID.randomUUID().toString().replace("-", "_");
			String distName = "dist_" + java.util.UUID.randomUUID().toString().replace("-", "_");
			List<String> names = Arrays.asList(probName, distName);
			String code = ConvertDims.getMethodDeclaration() + System.lineSeparator();
			code += "created_shms.clear()" + System.lineSeparator();
			code += "task.outputs.clear()" + System.lineSeparator();
			code += SHM_NAMES_KEY + " = []" + System.lineSeparator();
			code += DTYPES_KEY + " = []" + System.lineSeparator();
			code += DIMS_KEY + " = []" + System.lineSeparator();
			for (int i = 0; i < 2; i ++) {
				SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(reconstructed.get(i).getData(), false, false);
				code += codeToConvertShmaToPython(shma, names.get(i));
				inShmaList.add(shma);
			}
			code += probName + " = " + ConvertDims.getMethodName() + "(" + probName
					+ ", 'byx', out_order='yx', n_channels=1, output_type='numpy', contiguous=False)" + System.lineSeparator();
			code += distName + " = " + ConvertDims.getMethodName() + "(" + distName
					+ ", 'byxc', out_order='yxc', n_channels=" + configInt("n_rays", 32)
					+ ", output_type='numpy', contiguous=False)" + System.lineSeparator();
			code += "points, probi, disti = non_maximum_suppression(" + distName + ", " + probName
					+ ", grid=" + MODEL_VAR_NAME + ".config.grid, "
					+ "prob_thresh=" + (threshold == null ? MODEL_VAR_NAME + ".thresholds.prob" : threshold.toString()) + ", "
					+ "nms_thresh=" + MODEL_VAR_NAME + ".thresholds.nms, verbose=False)" + System.lineSeparator();
			code += "labels = polygons_to_label(disti, points, prob=probi, shape=("
					+ imageHeight + ", " + imageWidth + "))" + System.lineSeparator();
	        code += String.format("handle_output(labels.astype(np.float32, copy=False), %s, %s, %s, %s)",
	        		SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY)  + System.lineSeparator();
			code += taskOutputsCode();
			Map<String, RandomAccessibleInterval<R>> labels = executeCode(code);
			if (labels.isEmpty()) {
				return reconstructed;
			}
			RandomAccessibleInterval<R> labelImage = labels.values().iterator().next();
			//return Arrays.asList(Tensor.build("labels", "yx", Cast.unchecked(toStableIntImage(labelImage))));
			return Arrays.asList(Tensor.build("labels", "yx", labelImage));
		} catch (RunModelException e) {
			throw new IllegalStateException("StarDist NMS failed after dense tile reconstruction.", e);
		}
	}

	private static <T extends RealType<T> & NativeType<T>> Tensor<T> firstSpatialInput(final List<Tensor<T>> inputs) {
		for (Tensor<T> input : inputs) {
			if (hasSpatialAxes(input)) {
				return input;
			}
		}
		return null;
	}

	private <T extends RealType<T> & NativeType<T>> double objectScale(final Tensor<T> reference) {
		if (objectSize == null || objectSize.width <= 0 || objectSize.height <= 0) {
			return 1.0d;
		}
		String axes = reference.getAxesOrderString().toLowerCase();
		long[] dims = reference.getData().dimensionsAsLongArray();
		long x = axisSize(dims, axes, 'x');
		long y = axisSize(dims, axes, 'y');
		long tileX = Math.min(DEFAULT_DENSE_TILE_XY, x * 3L);
		long tileY = Math.min(DEFAULT_DENSE_TILE_XY, y * 3L);
		double objectArea = objectSize.getWidth() * objectSize.getHeight();
		double tileArea = tileX * (double) tileY;
		double ratio = objectArea / tileArea;
		if (ratio >= MIN_OBJECT_TILE_AREA_RATIO && ratio <= MAX_OBJECT_TILE_AREA_RATIO) {
			return 1.0d;
		}
		double scale = Math.sqrt((IDEAL_OBJECT_TILE_AREA_RATIO * tileArea) / objectArea);
		long roiX = Math.max(1L, tileX - 2L * Math.min(DEFAULT_DENSE_OUTPUT_HALO_XY, Math.max(0L, (tileX - 1L) / 2L)));
		long roiY = Math.max(1L, tileY - 2L * Math.min(DEFAULT_DENSE_OUTPUT_HALO_XY, Math.max(0L, (tileY - 1L) / 2L)));
		double imageArea = x * (double) y;
		return imageArea * scale * scale < roiX * (double) roiY * 0.9d
				? Math.sqrt((roiX * (double) roiY) / imageArea)
				: scale;
	}

	private static boolean needsResize(final double scale) {
		return Double.isFinite(scale) && Math.abs(scale - 1.0d) > 1e-6d;
	}

	private static <T extends RealType<T> & NativeType<T>> Tensor<T> resizeTensor(
			final Tensor<T> input, final double scale) {
		String axes = input.getAxesOrderString().toLowerCase();
		RandomAccessibleInterval<T> source = input.getData();
		long[] sourceDims = source.dimensionsAsLongArray();
		long[] targetDims = sourceDims.clone();
		int xAxis = axisIndex(axes, 'x');
		int yAxis = axisIndex(axes, 'y');
		targetDims[xAxis] = Math.max(1L, Math.round(sourceDims[xAxis] * scale));
		targetDims[yAxis] = Math.max(1L, Math.round(sourceDims[yAxis] * scale));
		Img<T> target = new ArrayImgFactory<T>(Util.getTypeFromInterval(source)).create(targetDims);
		LinearAxis xMap = new LinearAxis(sourceDims[xAxis], targetDims[xAxis]);
		LinearAxis yMap = new LinearAxis(sourceDims[yAxis], targetDims[yAxis]);
		LongStream.range(0L, lineCount(targetDims, xAxis)).parallel().forEach(line -> {
			RandomAccess<T> targetAccess = target.randomAccess();
			RandomAccess<T> sourceAccess = source.randomAccess();
			long[] targetPosition = linePosition(line, targetDims, xAxis);
			long[] sourcePosition = targetPosition.clone();
			int yy = Math.toIntExact(targetPosition[yAxis]);
			for (int x = 0; x < xMap.length(); x ++) {
				targetPosition[xAxis] = x;
				targetAccess.setPosition(targetPosition);
				targetAccess.get().setReal(bilinear(sourceAccess, sourcePosition, xAxis, yAxis, xMap, yMap, x, yy));
			}
		});
		return Tensor.build(input.getName(), input.getAxesOrderString(), target);
	}

	private static <T extends RealType<T> & NativeType<T>> double bilinear(
			final RandomAccess<T> access, final long[] position, final int xAxis, final int yAxis,
			final LinearAxis xMap, final LinearAxis yMap, final int x, final int y) {
		position[xAxis] = xMap.low[x];
		position[yAxis] = yMap.low[y];
		access.setPosition(position);
		double v00 = access.get().getRealDouble();
		position[xAxis] = xMap.high[x];
		access.setPosition(position);
		double v10 = access.get().getRealDouble();
		position[xAxis] = xMap.low[x];
		position[yAxis] = yMap.high[y];
		access.setPosition(position);
		double v01 = access.get().getRealDouble();
		position[xAxis] = xMap.high[x];
		access.setPosition(position);
		double v11 = access.get().getRealDouble();
		return (v00 * xMap.lowWeight[x] + v10 * xMap.highWeight[x]) * yMap.lowWeight[y]
				+ (v01 * xMap.lowWeight[x] + v11 * xMap.highWeight[x]) * yMap.highWeight[y];
	}

	private static final class LinearAxis {
		private final long[] low;
		private final long[] high;
		private final double[] lowWeight;
		private final double[] highWeight;

		private LinearAxis(final long sourceSize, final long targetSize) {
			int size = Math.toIntExact(targetSize);
			this.low = new long[size];
			this.high = new long[size];
			this.lowWeight = new double[size];
			this.highWeight = new double[size];
			double ratio = sourceSize / (double) targetSize;
			for (int i = 0; i < size; i ++) {
				double source = Math.max(0.0d, Math.min(sourceSize - 1.0d, (i + 0.5d) * ratio - 0.5d));
				long lo = (long) Math.floor(source);
				long hi = Math.min(sourceSize - 1L, lo + 1L);
				this.low[i] = lo;
				this.high[i] = hi;
				this.highWeight[i] = source - lo;
				this.lowWeight[i] = 1.0d - this.highWeight[i];
			}
		}

		private int length() {
			return low.length;
		}
	}

	private static <R extends RealType<R> & NativeType<R>> List<Tensor<R>> restoreOriginalScale(
			final List<Tensor<R>> tensors, final long originalHeight, final long originalWidth) {
		List<Tensor<R>> restored = new ArrayList<Tensor<R>>(tensors.size());
		for (Tensor<R> tensor : tensors) {
			restored.add(hasSpatialAxes(tensor) ? resizeTensorNearest(tensor, originalHeight, originalWidth) : tensor);
		}
		return restored;
	}

	private static <T extends RealType<T> & NativeType<T>> Tensor<T> resizeTensorNearest(
			final Tensor<T> input, final long height, final long width) {
		String axes = input.getAxesOrderString().toLowerCase();
		RandomAccessibleInterval<T> source = input.getData();
		long[] sourceDims = source.dimensionsAsLongArray();
		long[] targetDims = sourceDims.clone();
		int xAxis = axisIndex(axes, 'x');
		int yAxis = axisIndex(axes, 'y');
		targetDims[xAxis] = width;
		targetDims[yAxis] = height;
		if (Arrays.equals(sourceDims, targetDims)) {
			return input;
		}
		Img<T> target = new ArrayImgFactory<T>(Util.getTypeFromInterval(source)).create(targetDims);
		double xRatio = sourceDims[xAxis] / (double) targetDims[xAxis];
		double yRatio = sourceDims[yAxis] / (double) targetDims[yAxis];
		LongStream.range(0L, lineCount(targetDims, xAxis)).parallel().forEach(line -> {
			RandomAccess<T> sourceAccess = source.randomAccess();
			RandomAccess<T> targetAccess = target.randomAccess();
			long[] targetPosition = linePosition(line, targetDims, xAxis);
			long[] sourcePosition = targetPosition.clone();
			sourcePosition[yAxis] = Math.min(sourceDims[yAxis] - 1L, (long) Math.floor(targetPosition[yAxis] * yRatio));
			for (long x = 0L; x < targetDims[xAxis]; x ++) {
				targetPosition[xAxis] = x;
				sourcePosition[xAxis] = Math.min(sourceDims[xAxis] - 1L, (long) Math.floor(x * xRatio));
				sourceAccess.setPosition(sourcePosition);
				targetAccess.setPosition(targetPosition);
				targetAccess.get().set(sourceAccess.get());
			}
		});
		return Tensor.build(input.getName(), input.getAxesOrderString(), target);
	}

	private static long lineCount(final long[] dims, final int lineAxis) {
		long count = 1L;
		for (int d = 0; d < dims.length; d ++) {
			if (d != lineAxis) {
				count *= dims[d];
			}
		}
		return count;
	}

	private static long[] linePosition(long line, final long[] dims, final int lineAxis) {
		long[] position = new long[dims.length];
		for (int d = 0; d < dims.length; d ++) {
			if (d == lineAxis) {
				continue;
			}
			position[d] = line % dims[d];
			line /= dims[d];
		}
		return position;
	}

	private static <T extends RealType<T> & NativeType<T>> boolean hasSpatialAxes(final Tensor<T> tensor) {
		if (tensor == null || tensor.getAxesOrderString() == null) {
			return false;
		}
		String axes = tensor.getAxesOrderString().toLowerCase();
		return axes.indexOf('x') >= 0 && axes.indexOf('y') >= 0;
	}

	private <T extends RealType<T> & NativeType<T>> void applyInputNormalization(final List<Tensor<T>> inputs) {
		if (!normalizationEnabled()) {
			return;
		}
		ScaleRangeTransformation transform = new ScaleRangeTransformation();
		transform.setAxes(normalizationAxes());
		transform.setMinPercentile(normalizationMinPercentile());
		transform.setMaxPercentile(normalizationMaxPercentile());
		for (int i = 0; i < inputs.size(); i ++) {
			Tensor<T> input = inputs.get(i);
			if (hasSpatialAxes(input)) {
				applyNormalization(transform, inputs, i);
			}
		}
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	private static <T extends RealType<T> & NativeType<T>> void applyNormalization(
			final ScaleRangeTransformation transform, final List<Tensor<T>> inputs, final int index) {
		Tensor<T> input = inputs.get(index);
		if (Util.getTypeFromInterval(input.getData()) instanceof IntegerType) {
			inputs.set(index, (Tensor<T>) (Tensor) transform.apply(input));
		} else {
			transform.applyInPlace(input);
		}
	}

	private boolean normalizationEnabled() {
		Object value = firstPresent(config, "normalization", "normalize");
		if (value == null) {
			return true;
		}
		if (value instanceof Boolean) {
			return ((Boolean) value).booleanValue();
		}
		String text = value.toString().trim().toLowerCase();
		return !("false".equals(text) || "none".equals(text) || "no".equals(text) || "off".equals(text));
	}

	private Object normalizationAxes() {
		Object axes = firstPresent(config, "normalization_axes", "axis_norm");
		if (axes instanceof List<?>) {
			List<?> list = (List<?>) axes;
			if (!list.isEmpty() && list.get(0) instanceof Number) {
				String modelAxes = inferAxes(config, false);
				StringBuilder converted = new StringBuilder();
				for (Object axis : list) {
					int index = ((Number) axis).intValue();
					if (index >= 0 && index < modelAxes.length()) {
						converted.append(modelAxes.charAt(index));
					}
				}
				return converted.length() == 0 ? "yx" : converted.toString();
			}
		}
		return axes == null ? "yx" : axes;
	}

	private double normalizationMinPercentile() {
		Object percentiles = config.get("normalization_percentiles");
		if (percentiles instanceof List<?> && !((List<?>) percentiles).isEmpty()) {
			return asDouble(((List<?>) percentiles).get(0), 1.0d);
		}
		Object value = firstPresent(config, "normalization_pmin", "normalization_min_percentile");
		return asDouble(value == null ? config.get("pmin") : value, 1.0d);
	}

	private double normalizationMaxPercentile() {
		Object percentiles = config.get("normalization_percentiles");
		if (percentiles instanceof List<?> && ((List<?>) percentiles).size() > 1) {
			return asDouble(((List<?>) percentiles).get(1), 99.8d);
		}
		Object value = firstPresent(config, "normalization_pmax", "normalization_max_percentile");
		return asDouble(value == null ? config.get("pmax") : value, 99.8d);
	}

	private static double asDouble(final Object value, final double defaultValue) {
		if (value instanceof Number) {
			return ((Number) value).doubleValue();
		}
		if (value instanceof String && !((String) value).trim().isEmpty()) {
			return Double.parseDouble(((String) value).trim());
		}
		return defaultValue;
	}

	private static <T extends RealType<T> & NativeType<T>> TileInfo createInputTileInfo(final Tensor<T> input) {
		String axes = input.getAxesOrderString().toLowerCase();
		long[] imageDims = input.getData().dimensionsAsLongArray();
		long[] tileDims = imageDims.clone();
		int xAxis = axisIndex(axes, 'x');
		int yAxis = axisIndex(axes, 'y');
		tileDims[xAxis] = Math.min(DEFAULT_DENSE_TILE_XY, imageDims[xAxis] * 3);
		tileDims[yAxis] = Math.min(DEFAULT_DENSE_TILE_XY, imageDims[yAxis] * 3);
		return TileInfo.build(input.getName(), imageDims, axes, tileDims, axes);
	}

	private <T extends RealType<T> & NativeType<T>> List<TileInfo> createDenseOutputTileInfo(final Tensor<T> reference) {
		String axes = reference.getAxesOrderString().toLowerCase();
		long[] inputDims = reference.getData().dimensionsAsLongArray();
		long batch = axisSizeOrDefault(inputDims, axes, 'b', 1L);
		long y = axisSize(inputDims, axes, 'y');
		long x = axisSize(inputDims, axes, 'x');
		long tileY = Math.min(DEFAULT_DENSE_TILE_XY, y * 3);
		long tileX = Math.min(DEFAULT_DENSE_TILE_XY, x * 3);
		int gridY = grid(0);
		int gridX = grid(1);
		int nRays = configInt("n_rays", 32);
		long outputTileY = ceilDiv(tileY, gridY);
		long outputTileX = ceilDiv(tileX, gridX);
		long haloY = safeOutputHalo(outputTileY, gridY);
		long haloX = safeOutputHalo(outputTileX, gridX);

		List<TileInfo> outputInfo = new ArrayList<TileInfo>();
		TileInfo prob = TileInfo.build("output_0",
				new long[] {batch, ceilDiv(y, gridY), ceilDiv(x, gridX)},
				"byx",
				new long[] {1L, outputTileY, outputTileX},
				"byx");
		prob.setHalo(new long[] {0L, haloY, haloX}, "byx");

		TileInfo dist = TileInfo.build("output_1",
				new long[] {batch, ceilDiv(y, gridY), ceilDiv(x, gridX), nRays},
				DENSE_OUTPUT_AXES,
				new long[] {1L, outputTileY, outputTileX, nRays},
				DENSE_OUTPUT_AXES);
		dist.setHalo(new long[] {0L, haloY, haloX, 0L}, DENSE_OUTPUT_AXES);

		outputInfo.add(prob);
		outputInfo.add(dist);
		TileInfo.adaptHalos(outputInfo);
		return outputInfo;
	}

	private int grid(final int xyIndex) {
		Object value = config.get("grid");
		if (value instanceof List<?>) {
			List<?> grid = (List<?>) value;
			if (grid.size() > xyIndex && grid.get(xyIndex) instanceof Number) {
				return Math.max(1, ((Number) grid.get(xyIndex)).intValue());
			}
		}
		return 1;
	}

	private int configInt(final String key, final int defaultValue) {
		Object value = config.get(key);
		return value instanceof Number ? ((Number) value).intValue() : defaultValue;
	}

	private static int axisIndex(final String axes, final char axis) {
		int index = axes.indexOf(axis);
		if (index < 0) {
			throw new IllegalArgumentException("Axes '" + axes + "' do not contain axis '" + axis + "'.");
		}
		return index;
	}

	private static long axisSize(final long[] dims, final String axes, final char axis) {
		return dims[axisIndex(axes, axis)];
	}

	private static long axisSizeOrDefault(final long[] dims, final String axes, final char axis,
			final long defaultValue) {
		int index = axes.indexOf(axis);
		return index < 0 ? defaultValue : dims[index];
	}

	private static long ceilDiv(final long value, final int divisor) {
		return (long) Math.ceil(value / (double) Math.max(1, divisor));
	}

	private static long safeOutputHalo(final long outputTileSize, final int grid) {
		long requested = ceilDiv(DEFAULT_DENSE_OUTPUT_HALO_XY, grid);
		return Math.min(requested, Math.max(0L, (outputTileSize - 1L) / 2L));
	}

	/**
	 * Returns the output tensor axes.
	 *
	 * @param outputCount the output count.
	 * @return the output tensor axes.
	 */
	@Override
    protected String getOutputTensorAxes(int outputCount) {
		if (outputCount == 0)
			return "byx";
		else if (outputCount == 1)
			return "byxc";
		else
			throw new IllegalArgumentException("StarDist only has 2 outputs, more than 3 have been provided.");
	}

	/**
	 * Builds the model code.
	 *
	 * @return the created string.
	 */
	@Override
	protected String buildModelCode() {
		String source = mpkPath != null && new File(mpkPath).exists() ? "r'" + mpkPath + "'" : "None";
		String configStr = TrainingCodeUtils.toJson(config).replace("null", "None").replace("true", "True").replace("false", "False");
		return String.format(LOAD_MODEL_CODE_2D, device, source, configStr);
	}

	/**
	 * Creates the inputs code.
	 *
	 * @param <T> the T type parameter.
	 * @param inRais the in RAIs.
	 * @param names the names.
	 * @return the created string.
	 */
	@Override
	protected <T extends RealType<T> & NativeType<T>> 
		String createInputsCode(List<Tensor<T>> inRais, List<String> names) {
			String code = "";
			code += ConvertDims.getMethodDeclaration() + System.lineSeparator();
			code += "created_shms.clear()" + System.lineSeparator();
	        code += "task.outputs.clear()" + System.lineSeparator();
			code += SHM_NAMES_KEY + " = []" + System.lineSeparator();
			code += DTYPES_KEY + " = []" + System.lineSeparator();
			code += DIMS_KEY + " = []" + System.lineSeparator();
			List<SharedMemoryArray> shmas = createSharedMemoryArraysForInputs(inRais);
			for (int i = 0; i < inRais.size(); i ++) {
				SharedMemoryArray shma = shmas.get(i);
				code += codeToConvertShmaToPython(shma, names.get(i));
				inShmaList.add(shma);
				code += names.get(i) + " = " + ConvertDims.getMethodName() + "(" + names.get(i)
				+ ", '" + inRais.get(i).getAxesOrderString().toLowerCase()
				+ "', out_order='yxc', output_type='numpy', contiguous=False, n_channels="
				+ nChannels + ")" + System.lineSeparator();
			}
			code += "with tf.device(_jdll_tf_device):" + System.lineSeparator();
			code += "    _prob, _dist = " + MODEL_VAR_NAME + ".predict("
					+ names.get(0) + ", axes='YXC', normalizer=None, n_tiles=None, show_tile_progress=False)" + System.lineSeparator();
			code += OUTPUT_LIST_KEY + " = [np.expand_dims(_prob, 0), np.expand_dims(_dist, 0)]" + System.lineSeparator();
	        code += String.format("handle_output_list(%s, %s, %s, %s, %s)", OUTPUT_LIST_KEY,
	        		SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY)  + System.lineSeparator();
			code += taskOutputsCode();
			return code;
	}

	private static File resolveModelIdentityFile(String modelPath) {
		if (modelPath == null || modelPath.trim().isEmpty()) {
			throw new IllegalArgumentException("StarDist model path cannot be empty.");
		}
		File file = new File(modelPath);
		if (file.isFile()) {
			return file.getAbsoluteFile();
		}
		if (file.isDirectory()) {
			File modelFile = findModelFile(file);
			if (modelFile != null) {
				return modelFile.getAbsoluteFile();
			}
		}
		throw new IllegalArgumentException("Path provided does not point to a StarDist model: " + modelPath);
	}

	private static File findModelFile(File modelPath) {
		if (modelPath == null) {
			return null;
		}
		if (modelPath.isFile()) {
			return modelPath;
		}
		if (!modelPath.isDirectory()) {
			return null;
		}
		java.nio.file.Path modelFile = StardistModelRegistry.findModelFile(modelPath.getAbsolutePath());
		return modelFile == null ? null : modelFile.toFile();
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

	private static Map<String, Object> normalizedConfig(Map<String, Object> configMap) {
		Map<String, Object> config = new LinkedHashMap<String, Object>();
		if (configMap != null) {
			config.putAll(configMap);
		}
		if (!config.containsKey("axes")) {
			config.put("axes", "YXC");
		}
		if (!config.containsKey("n_channel_in")) {
			config.put("n_channel_in", 1);
		}
		return config;
	}

	private static int inferNChannels(Map<String, Object> config) {
		Object nChannels = config.get("n_channel_in");
		return nChannels instanceof Number ? ((Number) nChannels).intValue() : 1;
	}

	private static String inferAxes(Map<String, Object> config, boolean default3D) {
		Object axes = config.get("axes");
		if (axes != null && !axes.toString().trim().isEmpty()) {
			return axes.toString().toLowerCase();
		}
		return default3D ? "zyxc" : "yxc";
	}

	private static Dimensionality inferDimensionality(Map<String, Object> config) {
		return inferAxes(config, false).indexOf('z') >= 0 ? Dimensionality.THREE_D : Dimensionality.TWO_D;
	}

	/**
	 * Returns the result of resolve pytorch environment.
	 *
	 * @return the resulting pixi environment spec.
	 */
	public static PixiEnvironmentSpec resolvePytorchEnv() {
		String pixiTomlContent = String.format(java.util.Locale.ROOT,
				PixiEnvironmentResolver.readClasspathResourceAsString(PIXI_TOML),
				COMMON_STARDIST_ENV_NAME);
		String selectedEnvironment = resolveStardistEnvironmentName();
		return new PixiEnvironmentSpec(selectedEnvironment, pixiTomlContent,
				new File(org.apposed.appose.util.Environments.apposeEnvsDir(), COMMON_STARDIST_ENV_NAME),
				new ArrayList<String>());
	}

	private static String resolveStardistEnvironmentName() {
		String arch = PlatformDetection.getArch();
		if (PlatformDetection.isLinux()) {
			if (PlatformDetection.ARCH_X86_64.equals(arch)) {
				return GpuCompatibility.canInstallCudaInEnv(STARDIST_CUDA_VERSION)
						? "linux-x86-64-cuda"
						: "linux-x86-64-no-cuda";
			}
			if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
				return "linux-aarch64";
			}
		}
		if (PlatformDetection.isWindows() && PlatformDetection.ARCH_X86_64.equals(arch)) {
			return GpuCompatibility.canInstallCudaInEnv(STARDIST_CUDA_VERSION)
					? "win-x86-64-cuda"
					: "win-x86-64-no-cuda";
		}
		if (PlatformDetection.isMacOS()) {
			if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)
					|| PlatformDetection.isUsingRosseta()) {
				return "macos-arm64";
			}
			return "macos-x86-64";
		}
		throw new RuntimeException("Unsupported platform for StarDist: "
				+ PlatformDetection.getOs() + "-" + arch);
	}

	/**
	 * Returns whether installed.
	 *
	 * @return true if installed; false otherwise.
	 */
	public static boolean isInstalled() {
		try {
			return PixiEnvironmentManager.isInstalled(resolvePytorchEnv());
		} catch (Exception e) {
			return false;
		}
	}

	/**
	 * Runs model training.
	 *
	 * @param epochs the epochs.
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public static void train(int epochs, String dataDir, String gtDir, String outputDir,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, true, "grayscale", "grayscale", 0.15d,
				defaultTrainingConfig(epochs), progressConsumer, previewConsumer, logConsumer);
	}

	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param gpu whether to use GPU.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public static void train(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, gpu ? "cuda" : "cpu", imageChannels, labelColorMode, validFraction,
				config, progressConsumer, previewConsumer, logConsumer, null);
	}

	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param gpu whether to use GPU.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @param serviceConsumer the service consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public static void train(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer,
			Consumer<Service> serviceConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, gpu ? "cuda" : "cpu", imageChannels, labelColorMode, validFraction,
				config, progressConsumer, previewConsumer, logConsumer, serviceConsumer);
	}

	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param device the device.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @param serviceConsumer the service consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public static void train(String dataDir, String gtDir, String outputDir,
			String device, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer,
			Consumer<Service> serviceConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, device, imageChannels, labelColorMode, validFraction,
				config, progressConsumer, previewConsumer, logConsumer, null, serviceConsumer);
	}

	/**
	 * Runs model training.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param device the device.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @param progressConsumer the progress consumer callback.
	 * @param previewConsumer the preview consumer callback.
	 * @param logConsumer the log consumer callback.
	 * @param cancelSignalPath the cancel signal path.
	 * @param serviceConsumer the service consumer callback.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws TaskException if task occurs.
	 */
	public static void train(String dataDir, String gtDir, String outputDir,
			String device, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer,
			String cancelSignalPath,
			Consumer<Service> serviceConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		validateTrainingArguments(dataDir, gtDir, outputDir, validFraction, config);
		String normalizedDevice = normalizeDevice(device);
		File output = new File(outputDir);
		if (!output.isDirectory() && !output.mkdirs()) {
			throw new IOException("Could not create StarDist output directory: " + output.getAbsolutePath());
		}

		PixiEnvironmentSpec envSpec = resolvePytorchEnv();
		Environment env = Appose.pixi()
				.environment(envSpec.getSelectedEnvironment())
				.wrap(envSpec.getEnvironmentDirectory());
		Service python = env.python();
		if (serviceConsumer != null) {
			serviceConsumer.accept(python);
		}
		python.init("import numpy as np");
		if (logConsumer != null) {
			python.debug(line -> TrainingCodeUtils.logTrainingDebug(line, logConsumer));
		}
		try {
			Task task = python.task(buildTrainingCode(dataDir, gtDir, outputDir, normalizedDevice,
					imageChannels, labelColorMode, validFraction, config, cancelSignalPath));
			task.listen(event -> handleTrainingEvent(event, progressConsumer, previewConsumer, logConsumer));
			task.waitFor();
		} finally {
			if (python.isAlive()) {
				python.close();
			}
			if (serviceConsumer != null) {
				serviceConsumer.accept(null);
			}
		}
	}

	/**
	 * Returns the result of default training config.
	 *
	 * @param epochs the epochs.
	 * @return the resulting map.
	 */
	public static Map<String, Object> defaultTrainingConfig(int epochs) {
		Map<String, Object> config = defaultModelConfig2D();
		config.put("train_shape_completion", false);
		config.put("train_completion_crop", 32);
		config.put("train_patch_size", Arrays.asList(256, 256));
		config.put("train_background_reg", 1e-4d);
		config.put("train_foreground_only", 0.9d);
		config.put("train_sample_cache", true);
		config.put("train_dist_loss", "mae");
		config.put("train_loss_weights", Arrays.asList(1.0d, 0.2d));
		config.put("train_class_weights", Arrays.asList(1.0d, 1.0d));
		config.put("train_epochs", epochs);
		config.put("train_steps_per_epoch", 100);
		config.put("train_learning_rate", 0.0003d);
		config.put("train_batch_size", 4);
		config.put("train_n_val_patches", null);
		config.put("train_tensorboard", false);
		config.put("train_reduce_lr", reduceLrConfig());
		config.put("use_gpu", false);
		config.put("validation_preview_count", 20);
		return config;
	}

	/**
	 * Returns the result of default model config2 d.
	 *
	 * @return the resulting map.
	 */
	public static Map<String, Object> defaultModelConfig2D() {
		Map<String, Object> config = new LinkedHashMap<String, Object>();
		config.put("axes", "YXC");
		config.put("n_channel_in", 1);
		config.put("n_rays", 32);
		config.put("grid", Arrays.asList(1, 1));
		config.put("backbone", "unet");
		return config;
	}

	/**
	 * Returns the result of default model config3 d.
	 *
	 * @return the resulting map.
	 */
	public static Map<String, Object> defaultModelConfig3D() {
		Map<String, Object> config = new LinkedHashMap<String, Object>();
		config.put("axes", "ZYXC");
		config.put("n_channel_in", 1);
		config.put("n_rays", 96);
		config.put("grid", Arrays.asList(1, 1, 1));
		config.put("backbone", "unet");
		config.put("train_patch_size", Arrays.asList(128, 128, 128));
		config.put("train_batch_size", 1);
		return config;
	}

	private static Map<String, Object> reduceLrConfig() {
		Map<String, Object> config = new LinkedHashMap<String, Object>();
		config.put("factor", 0.5d);
		config.put("patience", 40);
		config.put("min_delta", 0.0d);
		return config;
	}

	private static String normalizeDevice(String device) {
		if (device == null) {
			return "cpu";
		}
		String normalized = device.trim().toLowerCase();
		return "cuda".equals(normalized) || "mps".equals(normalized) ? normalized : "cpu";
	}

	private static void validateTrainingArguments(String dataDir, String gtDir,
			String outputDir, double validFraction, Map<String, Object> config) {
		if (dataDir == null || !new File(dataDir).isDirectory()) {
			throw new IllegalArgumentException("The StarDist dataset directory does not exist: " + dataDir);
		}
		if (gtDir != null && !gtDir.trim().isEmpty() && !new File(gtDir).isDirectory()) {
			throw new IllegalArgumentException("The StarDist ground-truth directory does not exist: " + gtDir);
		}
		if (outputDir == null || outputDir.trim().isEmpty()) {
			throw new IllegalArgumentException("The StarDist output directory cannot be empty.");
		}
		if (validFraction < 0.0d || validFraction >= 1.0d) {
			throw new IllegalArgumentException("The StarDist validation fraction must be in [0, 1).");
		}
		if (config == null || config.isEmpty()) {
			throw new IllegalArgumentException("The StarDist training config cannot be empty.");
		}
		Object epochs = firstPresent(config, "train_epochs", "epochs");
		if (!(epochs instanceof Number) || ((Number) epochs).intValue() <= 0) {
			throw new IllegalArgumentException("The StarDist config must define train_epochs > 0.");
		}
	}

	/**
	 * Builds the training code.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param gpu whether to use GPU.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @return the created string.
	 */
	public static String buildTrainingCode(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config) {
		return buildTrainingCode(dataDir, gtDir, outputDir, gpu ? "cuda" : "cpu",
				imageChannels, labelColorMode, validFraction, config);
	}

	/**
	 * Builds the training code.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param device the device.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @return the created string.
	 */
	public static String buildTrainingCode(String dataDir, String gtDir, String outputDir,
			String device, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config) {
		return buildTrainingCode(dataDir, gtDir, outputDir, device, imageChannels, labelColorMode,
				validFraction, config, null);
	}

	/**
	 * Builds the training code.
	 *
	 * @param dataDir the data directory.
	 * @param gtDir the ground-truth directory.
	 * @param outputDir the output directory.
	 * @param device the device.
	 * @param imageChannels the image channels.
	 * @param labelColorMode the label color mode.
	 * @param validFraction the valid fraction.
	 * @param config the config.
	 * @param cancelSignalPath the cancel signal path.
	 * @return the created string.
	 */
	public static String buildTrainingCode(String dataDir, String gtDir, String outputDir,
			String device, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config, String cancelSignalPath) {
		String nl = System.lineSeparator();
		boolean hasGtDir = gtDir != null && !gtDir.trim().isEmpty();
		String gtDirCode = hasGtDir ? "gt_dir = r'" + TrainingCodeUtils.py(new File(gtDir).getAbsolutePath()) + "'" + nl : "";
		String safeImageChannels = imageChannels == null || imageChannels.trim().isEmpty()
				? "grayscale" : imageChannels.trim();
		String normalizedDevice = normalizeDevice(device);
		return ""
				+ "import contextlib, json, os, random, sys" + nl
				+ "from pathlib import Path" + nl
				+ "import numpy as np" + nl
				+ TrainingCodeUtils.apposeStdoutCapture()
				+ "import tensorflow as tf" + nl
				+ "_jdll_requested_device = '" + TrainingCodeUtils.py(normalizedDevice) + "'" + nl
				+ "_jdll_tf_device = '/CPU:0'" + nl
				+ "try:" + nl
				+ "  if _jdll_requested_device == 'cpu':" + nl
				+ "    try:" + nl
				+ "      tf.config.set_visible_devices([], 'GPU')" + nl
				+ "    except Exception:" + nl
				+ "      pass" + nl
				+ "  elif _jdll_requested_device in ('cuda', 'mps', 'gpu'):" + nl
				+ "    _jdll_gpus = tf.config.list_physical_devices('GPU')" + nl
				+ "    if _jdll_gpus:" + nl
				+ "      for _jdll_gpu in _jdll_gpus:" + nl
				+ "        try:" + nl
				+ "          tf.config.experimental.set_memory_growth(_jdll_gpu, True)" + nl
				+ "        except Exception:" + nl
				+ "          pass" + nl
				+ "      _jdll_tf_device = '/GPU:0'" + nl
				+ "except Exception:" + nl
				+ "  _jdll_tf_device = '/CPU:0'" + nl
				+ "from csbdeep.utils import normalize" + nl
				+ "from stardist.models import Config2D, StarDist2D" + nl
				+ "try:" + nl
				+ "  from tensorflow.keras.callbacks import Callback" + nl
				+ "except Exception:" + nl
				+ "  from keras.callbacks import Callback" + nl
				+ "try:" + nl
				+ "  from tifffile import imread" + nl
				+ "except Exception:" + nl
				+ "  from imageio.v3 import imread" + nl
				+ "data_dir = r'" + TrainingCodeUtils.py(new File(dataDir).getAbsolutePath()) + "'" + nl
				+ gtDirCode
				+ "output_dir = Path(r'" + TrainingCodeUtils.py(new File(outputDir).getAbsolutePath()) + "')" + nl
				+ "preview_dir = output_dir / 'previews'" + nl
				+ "preview_manifest_path = preview_dir / 'latest.json'" + nl
				+ "output_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "preview_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "stardist_log_path = output_dir / 'training.log'" + nl
				+ "config = json.loads(r'''" + TrainingCodeUtils.toJson(config) + "''')" + nl
				+ "cancel_signal_path = r'" + TrainingCodeUtils.py(cancelSignalPath == null ? "" : cancelSignalPath) + "'" + nl
				+ "preview_count = int(config.pop('validation_preview_count', 20))" + nl
				+ "is_accelerated = _jdll_requested_device != 'cpu'" + nl
				+ "progress_every_n_steps = 5 if is_accelerated else 1" + nl
				+ "log_every_n_steps = 50 if is_accelerated else 10" + nl
				+ "# StarDist use_gpu enables gputools/OpenCL preprocessing, not TensorFlow CUDA." + nl
				+ "config['use_gpu'] = False" + nl
				+ "if '" + TrainingCodeUtils.py(safeImageChannels).toLowerCase() + "' == 'rgb':" + nl
				+ "  config['axes'] = 'YXC'" + nl
				+ "  config['n_channel_in'] = 3" + nl
				+ "else:" + nl
				+ "  config.setdefault('axes', 'YXC')" + nl
				+ "  config.setdefault('n_channel_in', 1)" + nl
				+ "state = {'total_steps': int(config.get('train_epochs', 0)) * int(config.get('train_steps_per_epoch', 0)), 'total_epochs': int(config.get('train_epochs', 0))}" + nl
				+ TrainingCodeUtils.taskUpdateFunction("_task_update")
				+ TrainingCodeUtils.scalarFunction("_scalar", false)
				+ TrainingCodeUtils.cleanDictFunction("_clean", "_scalar")
				+ "def _cancel_requested():" + nl
				+ "  return bool(cancel_signal_path) and os.path.exists(cancel_signal_path)" + nl
				+ "def _atomic_npy_save(path, array):" + nl
				+ "  tmp_path = str(path) + '.tmp'" + nl
				+ "  with open(tmp_path, 'wb') as f:" + nl
				+ "    np.save(f, array)" + nl
				+ "  os.replace(tmp_path, path)" + nl
				+ "IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}" + nl
				+ "MASK_DIRS = ('masks', 'mask', 'labels', 'label', 'gt')" + nl
				+ "IMAGE_DIRS = ('images', 'image', 'imgs', 'img', 'data')" + nl
				+ "def _files(folder):" + nl
				+ "  return sorted([p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])" + nl
				+ "def _subdir(parent, names):" + nl
				+ "  for name in names:" + nl
				+ "    p = Path(parent) / name" + nl
				+ "    if p.is_dir():" + nl
				+ "      return p" + nl
				+ "  return Path(parent)" + nl
				+ "def _mask_key(path):" + nl
				+ "  stem = path.stem" + nl
				+ "  for suffix in ('_mask', '_masks', '_label', '_labels'):" + nl
				+ "    if stem.lower().endswith(suffix):" + nl
				+ "      return stem[:-len(suffix)]" + nl
				+ "  return stem" + nl
				+ "def _pairs(image_folder, mask_folder):" + nl
				+ "  masks = {_mask_key(p): p for p in _files(mask_folder)}" + nl
				+ "  out = []" + nl
				+ "  for img in _files(image_folder):" + nl
				+ "    key = img.stem[:-6] if img.stem.lower().endswith('_image') else img.stem" + nl
				+ "    if key in masks:" + nl
				+ "      out.append((img, masks[key]))" + nl
				+ "  return out" + nl
				+ "def _split_pairs(folder):" + nl
				+ "  return _pairs(_subdir(folder, IMAGE_DIRS), _subdir(folder, MASK_DIRS))" + nl
				+ "def _load_pairs(pairs):" + nl
				+ "  X, Y = [], []" + nl
				+ "  axes = str(config.get('axes', 'YXC')).upper()" + nl
				+ "  n_channels = int(config.get('n_channel_in', 1))" + nl
				+ "  for img_path, mask_path in pairs:" + nl
				+ "    x = np.asarray(imread(str(img_path)))" + nl
				+ "    y = np.asarray(imread(str(mask_path))).astype(np.int32, copy=False)" + nl
				+ "    if y.ndim > 2:" + nl
				+ "      y = y[..., 0]" + nl
				+ "    if n_channels == 1:" + nl
				+ "      if x.ndim == 3:" + nl
				+ "        x = x[..., 0]" + nl
				+ "      if 'C' in axes:" + nl
				+ "        x = x[..., None]" + nl
				+ "    elif x.ndim == 2:" + nl
				+ "      x = np.repeat(x[..., None], n_channels, axis=-1)" + nl
				+ "    elif x.shape[-1] != n_channels:" + nl
				+ "      x = x[..., :n_channels]" + nl
				+ "    norm_axis = (0, 1) if x.ndim == 3 else (0, 1)" + nl
				+ "    X.append(normalize(x, 1, 99.8, axis=norm_axis).astype(np.float32, copy=False))" + nl
				+ "    Y.append(y)" + nl
				+ "  return X, Y" + nl
				+ "def _dataset():" + nl
				+ "  root = Path(data_dir)" + nl
				+ "  if 'gt_dir' in globals():" + nl
				+ "    pairs = _pairs(root, Path(gt_dir))" + nl
				+ "    random.Random(42).shuffle(pairs)" + nl
				+ "    n_val = max(1, int(round(len(pairs) * " + validFraction + "))) if len(pairs) > 1 else 0" + nl
				+ "    return pairs[n_val:], pairs[:n_val]" + nl
				+ "  train_dir = root / 'train'" + nl
				+ "  val_dir = root / 'val'" + nl
				+ "  if not val_dir.is_dir():" + nl
				+ "    val_dir = root / 'validation'" + nl
				+ "  if train_dir.is_dir():" + nl
				+ "    train_pairs = _split_pairs(train_dir)" + nl
				+ "    val_pairs = _split_pairs(val_dir) if val_dir.is_dir() else []" + nl
				+ "  else:" + nl
				+ "    train_pairs = _split_pairs(root)" + nl
				+ "    val_pairs = []" + nl
				+ "  if not val_pairs:" + nl
				+ "    random.Random(42).shuffle(train_pairs)" + nl
				+ "    n_val = max(1, int(round(len(train_pairs) * " + validFraction + "))) if len(train_pairs) > 1 else 0" + nl
				+ "    val_pairs = train_pairs[:n_val]" + nl
				+ "    train_pairs = train_pairs[n_val:]" + nl
				+ "  if not train_pairs or not val_pairs:" + nl
				+ "    raise ValueError('Could not find matching StarDist training/validation image-mask pairs in ' + data_dir)" + nl
				+ "  return train_pairs, val_pairs" + nl
				+ "class JDLLProgressCallback(Callback):" + nl
				+ "  def __init__(self, model_ref, X_val):" + nl
				+ "    super().__init__()" + nl
				+ "    self.model_ref = model_ref" + nl
				+ "    self.X_val = X_val" + nl
				+ "    self.global_step = 0" + nl
				+ "  def _lr(self):" + nl
				+ "    try:" + nl
				+ "      return float(self.model.optimizer.learning_rate.numpy())" + nl
				+ "    except Exception:" + nl
				+ "      return None" + nl
				+ "  def on_train_begin(self, logs=None):" + nl
				+ "    info = {'type': 'progress', 'epoch': 0, 'step': 0, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': {}, 'metrics': {}}" + nl
				+ "    _task_update(message='StarDist training started', current=0, maximum=state['total_steps'], info=info)" + nl
				+ "  def on_train_batch_end(self, batch, logs=None):" + nl
				+ "    logs = logs or {}" + nl
				+ "    self.global_step += 1" + nl
				+ "    epoch = int((self.global_step - 1) // max(1, int(config.get('train_steps_per_epoch', 1))) + 1)" + nl
				+ "    should_update = self.global_step == 1 or self.global_step % progress_every_n_steps == 0 or self.global_step == state['total_steps']" + nl
				+ "    should_log = self.global_step == 1 or self.global_step % log_every_n_steps == 0 or self.global_step == state['total_steps']" + nl
				+ "    if should_update:" + nl
				+ "      losses = _clean({'train/total_loss': logs.get('loss'), 'train/prob_loss': logs.get('prob_loss'), 'train/dist_loss': logs.get('dist_loss')})" + nl
				+ "      metrics = _clean({'learning_rate': self._lr()})" + nl
				+ "      info = {'type': 'progress', 'epoch': epoch, 'step': self.global_step, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': losses, 'metrics': metrics}" + nl
				+ "      kwargs = {'current': self.global_step, 'maximum': state['total_steps'], 'info': info}" + nl
				+ "      if should_log:" + nl
				+ "        kwargs['message'] = 'StarDist training step %d/%d' % (self.global_step, state['total_steps'])" + nl
				+ "      _task_update(**kwargs)" + nl
				+ "    if should_log:" + nl
				+ "      print('step %05d/%d epoch=%d/%d loss=%s prob=%s dist=%s lr=%s' % (self.global_step, state['total_steps'], epoch, state['total_epochs'], logs.get('loss'), logs.get('prob_loss'), logs.get('dist_loss'), self._lr()), flush=True)" + nl
				+ "    if _cancel_requested():" + nl
				+ "      try:" + nl
				+ "        self.model.stop_training = True" + nl
				+ "      except Exception:" + nl
				+ "        pass" + nl
				+ "      _task_update(message='StarDist training cancellation requested', current=self.global_step, maximum=state['total_steps'], info={'type': 'cancelled', 'epoch': epoch, 'step': self.global_step})" + nl
				+ "  def on_epoch_end(self, epoch, logs=None):" + nl
				+ "    logs = logs or {}" + nl
				+ "    current_epoch = int(epoch) + 1" + nl
				+ "    step = min(state['total_steps'], current_epoch * int(config.get('train_steps_per_epoch', 1)))" + nl
				+ "    losses = _clean({'train/total_loss': logs.get('loss'), 'val/total_loss': logs.get('val_loss')})" + nl
				+ "    metrics = _clean({'learning_rate': self._lr()})" + nl
				+ "    info = {'type': 'progress', 'epoch': current_epoch, 'step': step, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': losses, 'metrics': metrics}" + nl
				+ "    _task_update(message='StarDist epoch %d/%d' % (current_epoch, state['total_epochs']), current=step, maximum=state['total_steps'], info=info)" + nl
				+ "    print('epoch %03d/%d step=%d/%d loss=%s val_loss=%s lr=%s' % (current_epoch, state['total_epochs'], step, state['total_steps'], logs.get('loss'), logs.get('val_loss'), self._lr()), flush=True)" + nl
				+ "    if _cancel_requested():" + nl
				+ "      try:" + nl
				+ "        self.model.stop_training = True" + nl
				+ "      except Exception:" + nl
				+ "        pass" + nl
				+ "      return" + nl
				+ "    samples = []" + nl
				+ "    for i, image in enumerate(self.X_val[:preview_count]):" + nl
				+ "      image_path = preview_dir / ('preview_%03d_image.npy' % i)" + nl
				+ "      pred_path = preview_dir / ('preview_%03d_prediction.npy' % i)" + nl
				+ "      prob_path = preview_dir / ('preview_%03d_prob.npy' % i)" + nl
				+ "      sample = {'index': i}" + nl
				+ "      _atomic_npy_save(image_path, image)" + nl
				+ "      sample['image_path'] = str(image_path)" + nl
				+ "      try:" + nl
				+ "        prediction, details = self.model_ref.predict_instances(image, axes=str(config.get('axes', 'YXC')), normalizer=None, n_tiles=None, show_tile_progress=False)" + nl
				+ "        _atomic_npy_save(pred_path, np.asarray(prediction, dtype=np.int32))" + nl
				+ "        sample['prediction_path'] = str(pred_path)" + nl
				+ "        prob, _dist = self.model_ref.predict(image, axes=str(config.get('axes', 'YXC')), normalizer=None, n_tiles=None, show_tile_progress=False)" + nl
				+ "        _atomic_npy_save(prob_path, prob)" + nl
				+ "        sample['prob_path'] = str(prob_path)" + nl
				+ "      except Exception:" + nl
				+ "        pass" + nl
				+ "      samples.append(sample)" + nl
				+ "    if samples:" + nl
				+ "      manifest = {'epoch': current_epoch, 'samples': samples}" + nl
				+ "      with open(preview_manifest_path, 'w', encoding='utf-8') as f:" + nl
				+ "        json.dump(manifest, f)" + nl
				+ "      _task_update(message='StarDist validation preview epoch %d' % current_epoch, current=current_epoch, maximum=state['total_epochs'], info={'type': 'preview', 'epoch': current_epoch, 'preview_path': str(preview_manifest_path)})" + nl
				+ "train_pairs, val_pairs = _dataset()" + nl
				+ "X_train, Y_train = _load_pairs(train_pairs)" + nl
				+ "X_val, Y_val = _load_pairs(val_pairs)" + nl
				+ "with open(stardist_log_path, 'a', encoding='utf-8') as stardist_log, contextlib.redirect_stdout(stardist_log), contextlib.redirect_stderr(stardist_log), tf.device(_jdll_tf_device):" + nl
				+ "  model_config = Config2D(**config)" + nl
					+ "  model = StarDist2D(model_config, name=output_dir.name, basedir=str(output_dir.parent))" + nl
					+ "  model.prepare_for_training()" + nl
					+ "  model.callbacks.append(JDLLProgressCallback(model, X_val))" + nl
					+ "  best_path = output_dir / str(config.get('train_checkpoint', 'weights_best.h5'))" + nl
					+ "  last_path = output_dir / str(config.get('train_checkpoint_last', 'weights_last.h5'))" + nl
					+ "  if best_path.exists():" + nl
					+ "    _task_update(message='Overwriting StarDist best checkpoint during training: ' + str(best_path), info={'type': 'checkpoint', 'kind': 'best', 'path': str(best_path), 'overwrite': True})" + nl
					+ "  if last_path.exists():" + nl
					+ "    _task_update(message='Overwriting StarDist last checkpoint during training: ' + str(last_path), info={'type': 'checkpoint', 'kind': 'last', 'path': str(last_path), 'overwrite': True})" + nl
					+ "  history = model.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=int(config.get('train_epochs', 1)), steps_per_epoch=int(config.get('train_steps_per_epoch', 100)), workers=0)" + nl
					+ "  if best_path.exists():" + nl
					+ "    _task_update(message='StarDist best checkpoint: ' + str(best_path), info={'type': 'checkpoint', 'kind': 'best', 'path': str(best_path)})" + nl
					+ "  last_existed = last_path.exists()" + nl
					+ "  _task_update(message=('%s StarDist last checkpoint: %s' % ('Overwriting' if last_existed else 'Saving', str(last_path))), info={'type': 'checkpoint', 'kind': 'last', 'path': str(last_path), 'overwrite': last_existed})" + nl
					+ "  try:" + nl
					+ "    model.keras_model.save_weights(str(last_path))" + nl
					+ "    _task_update(message='StarDist last checkpoint: ' + str(last_path), info={'type': 'checkpoint', 'kind': 'last', 'path': str(last_path)})" + nl
					+ "  except Exception as _jdll_save_error:" + nl
					+ "    _task_update(message='Could not save StarDist last checkpoint: ' + str(_jdll_save_error), info={'type': 'warning', 'message': str(_jdll_save_error)})" + nl
					+ "_task_update(message='Exported/final StarDist model directory: ' + str(output_dir), info={'type': 'checkpoint', 'kind': 'final', 'path': str(output_dir)})" + nl
					+ "task.outputs['result'] = str(output_dir)" + nl;
		}

	private static void handleTrainingEvent(TaskEvent event,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer) {
		if (!event.responseType.equals(ResponseType.UPDATE) || event.info == null) {
			return;
		}
		if (event.message != null && logConsumer != null) {
			logConsumer.accept(event.message);
		}
		Object type = event.info.get("type");
		if ("progress".equals(type) && progressConsumer != null) {
			progressConsumer.accept(new StardistTrainingProgress(
					TrainingCodeUtils.asInt(event.info.get("epoch"), (int) event.current),
					TrainingCodeUtils.asInt(event.info.get("step"), (int) event.current),
					TrainingCodeUtils.asInt(event.info.get("total_epochs"), 0),
					TrainingCodeUtils.asInt(event.info.get("total_steps"), (int) event.maximum),
					TrainingCodeUtils.asDoubleMap(event.info.get("losses")),
					TrainingCodeUtils.asDoubleMap(event.info.get("metrics"))));
		} else if ("preview".equals(type) && previewConsumer != null) {
			Object previewPath = event.info.get("preview_path");
			previewConsumer.accept(new StardistValidationPreview(
					TrainingCodeUtils.asInt(event.info.get("epoch"), (int) event.current),
					previewPath == null ? null : previewPath.toString()));
		}
	}

	private static Object firstPresent(Map<String, Object> config, String preferred, String fallback) {
		Object value = config.get(preferred);
		return value == null ? config.get(fallback) : value;
	}
	
	/**
	 * Runs this class from the command line.
	 *
	 * @param args command-line arguments.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 * @throws LoadModelException if the model cannot be loaded.
	 * @throws RunModelException if model inference cannot be run.
	 * @throws InterruptedException if the current thread is interrupted.
	 */
	public static void main(String[] args) throws IOException, BuildException, LoadModelException, RunModelException, InterruptedException {
		String path = "/home/carlos/git/deep-icy/models/stardist/kkeras";
        PixiEnvironmentManager.installRequirements(StarDist.resolvePytorchEnv(), (str) -> {System.out.println(str);});
		try (StarDist model = StarDist.fromFile(path, null)) {
			Tensor<FloatType> tensor = Tensor.build(
					"input",
					"bcyx",
					ArrayImgs.floats(new float[1 * 1 * 512 * 512], 1, 1, 512, 512)
				);

			model.inference(tensor);
		}
	}
}
