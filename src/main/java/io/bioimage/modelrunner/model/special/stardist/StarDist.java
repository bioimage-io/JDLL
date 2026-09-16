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
	private static final long DEFAULT_DENSE_TILE_Z = 64L;
	private static final long DEFAULT_DENSE_OUTPUT_HALO_XY = 96L;
	private static final long DEFAULT_DENSE_OUTPUT_HALO_Z = 16L;
	private static final double MAX_OBJECT_TILE_AREA_RATIO = 0.1d;
	private static final double MIN_OBJECT_TILE_AREA_RATIO = 0.00018692d;
	private static final double IDEAL_OBJECT_TILE_AREA_RATIO = 0.04d;
	private static final double MAX_OBJECT_TILE_VOLUME_RATIO = Math.pow(MAX_OBJECT_TILE_AREA_RATIO, 1.5d);
	private static final double MIN_OBJECT_TILE_VOLUME_RATIO = Math.pow(MIN_OBJECT_TILE_AREA_RATIO, 1.5d);
	private static final double IDEAL_OBJECT_TILE_VOLUME_RATIO = Math.pow(IDEAL_OBJECT_TILE_AREA_RATIO, 1.5d);
	private static final double MIN_METADATA_SCALE = 0.25d;
	private static final double MAX_METADATA_SCALE = 4.0d;

	private static final String LOAD_MODEL_CODE = ""
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
			+ "if 'StarDist2D' not in globals().keys() or 'StarDist3D' not in globals().keys():" + System.lineSeparator()
			+ "  from stardist.models import StarDist2D, StarDist3D, Config2D, Config3D" + System.lineSeparator()
			+ "  from stardist.nms import non_maximum_suppression" + System.lineSeparator()
			+ "  from stardist.geometry import polygons_to_label" + System.lineSeparator()
			+ "  task.export(StarDist2D=StarDist2D)" + System.lineSeparator()
			+ "  task.export(StarDist3D=StarDist3D)" + System.lineSeparator()
			+ "  task.export(Config2D=Config2D)" + System.lineSeparator()
			+ "  task.export(Config3D=Config3D)" + System.lineSeparator()
			+ "  task.export(non_maximum_suppression=non_maximum_suppression)" + System.lineSeparator()
			+ "  task.export(polygons_to_label=polygons_to_label)" + System.lineSeparator()
			+ "_jdll_stardist_ndim = %d" + System.lineSeparator()
			+ "def _jdll_stardist_config(values):" + System.lineSeparator()
			+ "  values = dict(values or {})" + System.lineSeparator()
			+ "  values.pop('n_dim', None)" + System.lineSeparator()
			+ "  return (Config3D if _jdll_stardist_ndim == 3 else Config2D)(**values)" + System.lineSeparator()
			+ "def _jdll_load_stardist(source, config):" + System.lineSeparator()
			+ "  model_class = StarDist3D if _jdll_stardist_ndim == 3 else StarDist2D" + System.lineSeparator()
			+ "  if source is None:" + System.lineSeparator()
			+ "    return model_class(_jdll_stardist_config(config), name=None, basedir=None)" + System.lineSeparator()
			+ "  source = os.path.abspath(source)" + System.lineSeparator()
			+ "  model_dir = source if os.path.isdir(source) else os.path.dirname(source)" + System.lineSeparator()
			+ "  model = model_class(None, name=os.path.basename(model_dir), basedir=os.path.dirname(model_dir))" + System.lineSeparator()
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
	private final ObjectScaleMetadata objectScaleMetadata;
	protected Rectangle objectSize;

	private Double threshold = null;
	private StarDist(String modelIdentity, Map<String, Object> config,
			Dimensionality dimensionality, Consumer<InferenceProgress> inferenceProgressConsumer, String device) throws IOException {
		this(modelIdentity, modelIdentity, config, dimensionality, inferenceProgressConsumer, device);
	}

	private StarDist(String modelIdentity, String modelSource, Map<String, Object> config,
			Dimensionality dimensionality, Consumer<InferenceProgress> inferenceProgressConsumer, String device) throws IOException {
		super(modelIdentity, modelIdentity, modelIdentity, modelIdentity, config, true, device);
		File identityFile = new File(modelIdentity);
		File parent = identityFile.getParentFile();
		modelFolder = parent == null ? identityFile.getAbsoluteFile().getParent() : parent.getAbsolutePath();
		this.mpkPath = modelSource;
		this.config = normalizedConfig(config);
		this.nChannels = inferNChannels(this.config);
		this.dimensionality = dimensionality;
		this.objectScaleMetadata = modelSource == null ? null : loadObjectScaleMetadata(modelSource, dimensionality);
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
		StarDist model = new StarDist(scratchIdentity().getAbsolutePath(), null, normalized,
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
		StarDist model = new StarDist(scratchIdentity().getAbsolutePath(), null, config,
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
		StarDist model = new StarDist(scratchIdentity().getAbsolutePath(), null, config,
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
		long originalDepth = axisSizeOrDefault(originalDims, originalAxes, 'z', 1L);
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
		long imageDepth = axisSizeOrDefault(referenceDims, referenceAxes, 'z', 1L);
		merger.addCallback(reconstructed -> runStardistNms(reconstructed, imageDepth, imageHeight, imageWidth));
		merger.addCallback(reconstructed -> restoreOriginalScale(
				reconstructed, originalDepth, originalHeight, originalWidth));
		merger.configure(tiledInputs);
		return merger;
	}

	private <R extends RealType<R> & NativeType<R>> List<Tensor<R>> runStardistNms(
			final List<Tensor<R>> reconstructed, final long imageDepth,
			final long imageHeight, final long imageWidth) {
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
			String probAxes = is3D() ? "bzyx" : "byx";
			String distAxes = is3D() ? "bzyxc" : "byxc";
			String spatialAxes = is3D() ? "zyx" : "yx";
			String spatialChannelAxes = is3D() ? "zyxc" : "yxc";
			String imageShape = is3D()
					? "(" + imageDepth + ", " + imageHeight + ", " + imageWidth + ")"
					: "(" + imageHeight + ", " + imageWidth + ")";
			code += probName + " = " + ConvertDims.getMethodName() + "(" + probName
					+ ", '" + probAxes + "', out_order='" + spatialAxes
					+ "', n_channels=1, output_type='numpy', contiguous=False)" + System.lineSeparator();
			code += distName + " = " + ConvertDims.getMethodName() + "(" + distName
					+ ", '" + distAxes + "', out_order='" + spatialChannelAxes + "', n_channels="
					+ configInt("n_rays", is3D() ? 96 : 32)
					+ ", output_type='numpy', contiguous=False)" + System.lineSeparator();
			code += "labels, _jdll_details = " + MODEL_VAR_NAME + "._instances_from_prediction("
					+ imageShape + ", " + probName + ", " + distName + ", "
					+ "prob_thresh=" + (threshold == null ? MODEL_VAR_NAME + ".thresholds.prob" : threshold.toString()) + ", "
					+ "nms_thresh=" + MODEL_VAR_NAME + ".thresholds.nms, verbose=False)" + System.lineSeparator();
	        code += String.format("handle_output(labels.astype(np.float32, copy=False), %s, %s, %s, %s)",
	        		SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY)  + System.lineSeparator();
			code += taskOutputsCode();
			Map<String, RandomAccessibleInterval<R>> labels = executeCode(code);
			if (labels.isEmpty()) {
				return reconstructed;
			}
			RandomAccessibleInterval<R> labelImage = labels.values().iterator().next();
			//return Arrays.asList(Tensor.build("labels", "yx", Cast.unchecked(toStableIntImage(labelImage))));
			return Arrays.asList(Tensor.build("labels", spatialAxes, labelImage));
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
		if (objectScaleMetadata != null) {
			double[] anisotropy = modelAnisotropy();
			double scale = metadataObjectScale(objectScaleMetadata.objectDiameter,
					objectSize.getWidth(), objectSize.getHeight(),
					is3D() ? anisotropy[1] : 1.0d,
					is3D() ? anisotropy[2] : 1.0d);
			return minimumUsableScale(reference, scale);
		}
		return is3D() ? legacyVolumeScale(reference) : legacyAreaScale(reference);
	}

	private <T extends RealType<T> & NativeType<T>> double legacyAreaScale(final Tensor<T> reference) {
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

	private <T extends RealType<T> & NativeType<T>> double legacyVolumeScale(final Tensor<T> reference) {
		String axes = reference.getAxesOrderString().toLowerCase();
		long[] dims = reference.getData().dimensionsAsLongArray();
		long x = axisSize(dims, axes, 'x');
		long y = axisSize(dims, axes, 'y');
		long z = axisSize(dims, axes, 'z');
		long tileX = Math.min(DEFAULT_DENSE_TILE_XY, x * 3L);
		long tileY = Math.min(DEFAULT_DENSE_TILE_XY, y * 3L);
		long tileZ = Math.min(configuredTileZ(), z * 3L);
		double[] anisotropy = modelAnisotropy();
		double scale = legacyVolumeObjectScale(objectSize.getWidth(), objectSize.getHeight(),
				tileX, tileY, tileZ, anisotropy);
		if (scale == 1.0d) {
			return 1.0d;
		}
		return minimumUsableScale(reference, scale);
	}

	static double metadataObjectScale(final double modelDiameter,
			final double objectWidth, final double objectHeight,
			final double anisotropyY, final double anisotropyX) {
		double diameter = 2.0d * Math.sqrt(objectWidth * objectHeight / Math.PI)
				* Math.sqrt(anisotropyY * anisotropyX);
		double scale = modelDiameter / diameter;
		return Math.max(MIN_METADATA_SCALE, Math.min(MAX_METADATA_SCALE, scale));
	}

	static double legacyVolumeObjectScale(final double objectWidth, final double objectHeight,
			final long tileX, final long tileY, final long tileZ, final double[] anisotropy) {
		double diameterXY = Math.sqrt(objectWidth * objectHeight);
		double depth = diameterXY * Math.sqrt(anisotropy[1] * anisotropy[2]) / anisotropy[0];
		double objectVolume = objectWidth * objectHeight * depth;
		double tileVolume = tileX * (double) tileY * tileZ;
		double ratio = objectVolume / tileVolume;
		if (ratio >= MIN_OBJECT_TILE_VOLUME_RATIO && ratio <= MAX_OBJECT_TILE_VOLUME_RATIO) {
			return 1.0d;
		}
		return Math.cbrt((IDEAL_OBJECT_TILE_VOLUME_RATIO * tileVolume) / objectVolume);
	}

	private <T extends RealType<T> & NativeType<T>> double minimumUsableScale(
			final Tensor<T> reference, final double proposedScale) {
		String axes = reference.getAxesOrderString().toLowerCase();
		long[] dims = reference.getData().dimensionsAsLongArray();
		long x = axisSize(dims, axes, 'x');
		long y = axisSize(dims, axes, 'y');
		long tileX = Math.min(DEFAULT_DENSE_TILE_XY, x * 3L);
		long tileY = Math.min(DEFAULT_DENSE_TILE_XY, y * 3L);
		long roiX = usableTileSize(tileX, DEFAULT_DENSE_OUTPUT_HALO_XY);
		long roiY = usableTileSize(tileY, DEFAULT_DENSE_OUTPUT_HALO_XY);
		if (!is3D()) {
			double imageArea = x * (double) y;
			double roiArea = roiX * (double) roiY;
			return imageArea * proposedScale * proposedScale < roiArea * 0.9d
					? Math.sqrt(roiArea / imageArea)
					: proposedScale;
		}
		long z = axisSize(dims, axes, 'z');
		long tileZ = Math.min(configuredTileZ(), z * 3L);
		long roiZ = usableTileSize(tileZ, DEFAULT_DENSE_OUTPUT_HALO_Z);
		double imageVolume = x * (double) y * z;
		double roiVolume = roiX * (double) roiY * roiZ;
		return imageVolume * proposedScale * proposedScale * proposedScale < roiVolume * 0.9d
				? Math.cbrt(roiVolume / imageVolume)
				: proposedScale;
	}

	private static long usableTileSize(final long tileSize, final long halo) {
		return Math.max(1L, tileSize - 2L * Math.min(halo, Math.max(0L, (tileSize - 1L) / 2L)));
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
		int zAxis = axes.indexOf('z');
		targetDims[xAxis] = Math.max(1L, Math.round(sourceDims[xAxis] * scale));
		targetDims[yAxis] = Math.max(1L, Math.round(sourceDims[yAxis] * scale));
		if (zAxis >= 0) {
			targetDims[zAxis] = Math.max(1L, Math.round(sourceDims[zAxis] * scale));
		}
		Img<T> target = new ArrayImgFactory<T>(Util.getTypeFromInterval(source)).create(targetDims);
		LinearAxis xMap = new LinearAxis(sourceDims[xAxis], targetDims[xAxis]);
		LinearAxis yMap = new LinearAxis(sourceDims[yAxis], targetDims[yAxis]);
		LinearAxis zMap = zAxis < 0 ? null : new LinearAxis(sourceDims[zAxis], targetDims[zAxis]);
		LongStream.range(0L, lineCount(targetDims, xAxis)).parallel().forEach(line -> {
			RandomAccess<T> targetAccess = target.randomAccess();
			RandomAccess<T> sourceAccess = source.randomAccess();
			long[] targetPosition = linePosition(line, targetDims, xAxis);
			long[] sourcePosition = targetPosition.clone();
			int yy = Math.toIntExact(targetPosition[yAxis]);
			int zz = zAxis < 0 ? 0 : Math.toIntExact(targetPosition[zAxis]);
			for (int x = 0; x < xMap.length(); x ++) {
				targetPosition[xAxis] = x;
				targetAccess.setPosition(targetPosition);
				targetAccess.get().setReal(zAxis < 0
						? bilinear(sourceAccess, sourcePosition, xAxis, yAxis, xMap, yMap, x, yy)
						: trilinear(sourceAccess, sourcePosition, xAxis, yAxis, zAxis,
								xMap, yMap, zMap, x, yy, zz));
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

	private static <T extends RealType<T> & NativeType<T>> double trilinear(
			final RandomAccess<T> access, final long[] position,
			final int xAxis, final int yAxis, final int zAxis,
			final LinearAxis xMap, final LinearAxis yMap, final LinearAxis zMap,
			final int x, final int y, final int z) {
		position[zAxis] = zMap.low[z];
		double low = bilinear(access, position, xAxis, yAxis, xMap, yMap, x, y);
		position[zAxis] = zMap.high[z];
		double high = bilinear(access, position, xAxis, yAxis, xMap, yMap, x, y);
		return low * zMap.lowWeight[z] + high * zMap.highWeight[z];
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
			final List<Tensor<R>> tensors, final long originalDepth,
			final long originalHeight, final long originalWidth) {
		List<Tensor<R>> restored = new ArrayList<Tensor<R>>(tensors.size());
		for (Tensor<R> tensor : tensors) {
			restored.add(hasSpatialAxes(tensor)
					? resizeTensorNearest(tensor, originalDepth, originalHeight, originalWidth)
					: tensor);
		}
		return restored;
	}

	private static <T extends RealType<T> & NativeType<T>> Tensor<T> resizeTensorNearest(
			final Tensor<T> input, final long depth, final long height, final long width) {
		String axes = input.getAxesOrderString().toLowerCase();
		RandomAccessibleInterval<T> source = input.getData();
		long[] sourceDims = source.dimensionsAsLongArray();
		long[] targetDims = sourceDims.clone();
		int xAxis = axisIndex(axes, 'x');
		int yAxis = axisIndex(axes, 'y');
		int zAxis = axes.indexOf('z');
		targetDims[xAxis] = width;
		targetDims[yAxis] = height;
		if (zAxis >= 0) {
			targetDims[zAxis] = depth;
		}
		if (Arrays.equals(sourceDims, targetDims)) {
			return input;
		}
		Img<T> target = new ArrayImgFactory<T>(Util.getTypeFromInterval(source)).create(targetDims);
		double xRatio = sourceDims[xAxis] / (double) targetDims[xAxis];
		double yRatio = sourceDims[yAxis] / (double) targetDims[yAxis];
		double zRatio = zAxis < 0 ? 1.0d : sourceDims[zAxis] / (double) targetDims[zAxis];
		LongStream.range(0L, lineCount(targetDims, xAxis)).parallel().forEach(line -> {
			RandomAccess<T> sourceAccess = source.randomAccess();
			RandomAccess<T> targetAccess = target.randomAccess();
			long[] targetPosition = linePosition(line, targetDims, xAxis);
			long[] sourcePosition = targetPosition.clone();
			sourcePosition[yAxis] = Math.min(sourceDims[yAxis] - 1L, (long) Math.floor(targetPosition[yAxis] * yRatio));
			if (zAxis >= 0) {
				sourcePosition[zAxis] = Math.min(
						sourceDims[zAxis] - 1L, (long) Math.floor(targetPosition[zAxis] * zRatio));
			}
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

	private <T extends RealType<T> & NativeType<T>> TileInfo createInputTileInfo(final Tensor<T> input) {
		String axes = input.getAxesOrderString().toLowerCase();
		long[] imageDims = input.getData().dimensionsAsLongArray();
		long[] tileDims = imageDims.clone();
		int xAxis = axisIndex(axes, 'x');
		int yAxis = axisIndex(axes, 'y');
		tileDims[xAxis] = Math.min(DEFAULT_DENSE_TILE_XY, imageDims[xAxis] * 3);
		tileDims[yAxis] = Math.min(DEFAULT_DENSE_TILE_XY, imageDims[yAxis] * 3);
		if (is3D()) {
			int zAxis = axisIndex(axes, 'z');
			tileDims[zAxis] = Math.min(configuredTileZ(), imageDims[zAxis] * 3);
		}
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
		int gridY = gridForAxis('y');
		int gridX = gridForAxis('x');
		int nRays = configInt("n_rays", is3D() ? 96 : 32);
		long outputTileY = ceilDiv(tileY, gridY);
		long outputTileX = ceilDiv(tileX, gridX);
		long outputY = ceilDiv(y, gridY);
		long outputX = ceilDiv(x, gridX);
		long haloY = safeOutputHalo(outputTileY, outputY, gridY, DEFAULT_DENSE_OUTPUT_HALO_XY);
		long haloX = safeOutputHalo(outputTileX, outputX, gridX, DEFAULT_DENSE_OUTPUT_HALO_XY);

		List<TileInfo> outputInfo = new ArrayList<TileInfo>();
		if (is3D()) {
			long z = axisSize(inputDims, axes, 'z');
			long tileZ = Math.min(configuredTileZ(), z * 3);
			int gridZ = gridForAxis('z');
			long outputTileZ = ceilDiv(tileZ, gridZ);
			long outputZ = ceilDiv(z, gridZ);
			long haloZ = safeOutputHalo(outputTileZ, outputZ, gridZ, DEFAULT_DENSE_OUTPUT_HALO_Z);
			String probAxes = "bzyx";
			String distAxes = "bzyxc";
			TileInfo prob = TileInfo.build("output_0",
					new long[] {batch, outputZ, outputY, outputX},
					probAxes,
					new long[] {1L, outputTileZ, outputTileY, outputTileX},
					probAxes);
			prob.setHalo(new long[] {0L, haloZ, haloY, haloX}, probAxes);
			TileInfo dist = TileInfo.build("output_1",
					new long[] {batch, outputZ, outputY, outputX, nRays},
					distAxes,
					new long[] {1L, outputTileZ, outputTileY, outputTileX, nRays},
					distAxes);
			dist.setHalo(new long[] {0L, haloZ, haloY, haloX, 0L}, distAxes);
			outputInfo.add(prob);
			outputInfo.add(dist);
			TileInfo.adaptHalos(outputInfo);
			return outputInfo;
		}

		TileInfo prob = TileInfo.build("output_0",
				new long[] {batch, outputY, outputX},
				"byx",
				new long[] {1L, outputTileY, outputTileX},
				"byx");
		prob.setHalo(new long[] {0L, haloY, haloX}, "byx");

		TileInfo dist = TileInfo.build("output_1",
				new long[] {batch, outputY, outputX, nRays},
				"byxc",
				new long[] {1L, outputTileY, outputTileX, nRays},
				"byxc");
		dist.setHalo(new long[] {0L, haloY, haloX, 0L}, "byxc");

		outputInfo.add(prob);
		outputInfo.add(dist);
		TileInfo.adaptHalos(outputInfo);
		return outputInfo;
	}

	private int gridForAxis(final char axis) {
		int index = is3D()
				? (axis == 'z' ? 0 : axis == 'y' ? 1 : 2)
				: (axis == 'y' ? 0 : 1);
		Object value = config.get("grid");
		if (value instanceof List<?>) {
			List<?> grid = (List<?>) value;
			if (grid.size() > index && grid.get(index) instanceof Number) {
				return Math.max(1, ((Number) grid.get(index)).intValue());
			}
		}
		return 1;
	}

	private long configuredTileZ() {
		Object patch = config.get("train_patch_size");
		if (patch instanceof List<?> && !((List<?>) patch).isEmpty()
				&& ((List<?>) patch).get(0) instanceof Number) {
			return Math.max(1L, ((Number) ((List<?>) patch).get(0)).longValue());
		}
		return DEFAULT_DENSE_TILE_Z;
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

	private static long safeOutputHalo(final long outputTileSize, final long outputImageSize,
			final int grid, final long requestedInputHalo) {
		if (outputTileSize >= outputImageSize) {
			return 0L;
		}
		long requested = ceilDiv(requestedInputHalo, grid);
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
			return is3D() ? "bzyx" : "byx";
		else if (outputCount == 1)
			return is3D() ? "bzyxc" : "byxc";
		else
			throw new IllegalArgumentException("StarDist only has 2 outputs.");
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
		return String.format(LOAD_MODEL_CODE, device, is3D() ? 3 : 2, source, configStr);
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
				+ "', out_order='" + (is3D() ? "zyxc" : "yxc")
				+ "', output_type='numpy', contiguous=False, n_channels="
				+ nChannels + ")" + System.lineSeparator();
			}
			code += "with tf.device(_jdll_tf_device):" + System.lineSeparator();
			code += "    _prob, _dist = " + MODEL_VAR_NAME + ".predict("
					+ names.get(0) + ", axes='" + (is3D() ? "ZYXC" : "YXC")
					+ "', normalizer=None, n_tiles=None, show_tile_progress=False)" + System.lineSeparator();
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

	private static File scratchIdentity() throws IOException {
		File identity = File.createTempFile("jdll-stardist-config-", ".h5");
		identity.deleteOnExit();
		return identity;
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

	private static ObjectScaleMetadata loadObjectScaleMetadata(
			final String modelSource, final Dimensionality dimensionality) {
		try {
			File path = new File(modelSource);
			File modelDir = path.isDirectory() ? path : path.getParentFile();
			File metadataFile = modelDir == null ? null : new File(modelDir, "model_metadata.json");
			if (metadataFile == null || !metadataFile.isFile()) {
				return null;
			}
			Map<String, Object> metadata = JSONUtils.load(metadataFile.getAbsolutePath());
			Object scaleValue = metadata.get("instance_scale");
			if (!(scaleValue instanceof Map<?, ?>)) {
				return null;
			}
			Map<?, ?> scale = (Map<?, ?>) scaleValue;
			Object dimensionsValue = scale.get("dimensions");
			int dimensions = dimensionsValue instanceof Number
					? ((Number) dimensionsValue).intValue()
					: Integer.parseInt(String.valueOf(dimensionsValue));
			if (dimensions != (dimensionality == Dimensionality.THREE_D ? 3 : 2)) {
				return null;
			}
			Object diameterValue = scale.get("model_object_diameter_px");
			if (!(diameterValue instanceof Number)) {
				return null;
			}
			double diameter = ((Number) diameterValue).doubleValue();
			if (!Double.isFinite(diameter) || diameter <= 0.0d) {
				return null;
			}
			return new ObjectScaleMetadata(diameter, numericArray(scale.get("anisotropy"), 3));
		} catch (Exception ignored) {
			return null;
		}
	}

	private double[] modelAnisotropy() {
		if (objectScaleMetadata != null && objectScaleMetadata.anisotropy != null) {
			return objectScaleMetadata.anisotropy.clone();
		}
		double[] anisotropy = numericArray(config.get("anisotropy"), 3);
		return anisotropy == null ? new double[] {1.0d, 1.0d, 1.0d} : anisotropy;
	}

	private static double[] numericArray(final Object value, final int expectedSize) {
		if (!(value instanceof List<?>) || ((List<?>) value).size() != expectedSize) {
			return null;
		}
		double[] result = new double[expectedSize];
		for (int i = 0; i < result.length; i ++) {
			Object entry = ((List<?>) value).get(i);
			if (!(entry instanceof Number)) {
				return null;
			}
			result[i] = ((Number) entry).doubleValue();
			if (!Double.isFinite(result[i]) || result[i] <= 0.0d) {
				return null;
			}
		}
		return result;
	}

	private static final class ObjectScaleMetadata {
		private final double objectDiameter;
		private final double[] anisotropy;

		private ObjectScaleMetadata(final double objectDiameter, final double[] anisotropy) {
			this.objectDiameter = objectDiameter;
			this.anisotropy = anisotropy == null ? null : anisotropy.clone();
		}
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
		String code = ""
				+ "import contextlib, json, os, random, sys, xml.etree.ElementTree as ET" + nl
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
				+ "from stardist import Rays_GoldenSpiral, calculate_extents" + nl
				+ "from stardist.models import Config2D, Config3D, StarDist2D, StarDist3D" + nl
				+ "try:" + nl
				+ "  from tensorflow.keras.callbacks import Callback" + nl
				+ "except Exception:" + nl
				+ "  from keras.callbacks import Callback" + nl
				+ "from PIL import Image" + nl
				+ "from tifffile import TiffFile, TiffFileError" + nl
				+ "data_dir = r'" + TrainingCodeUtils.py(new File(dataDir).getAbsolutePath()) + "'" + nl
				+ gtDirCode
				+ "output_dir = Path(r'" + TrainingCodeUtils.py(new File(outputDir).getAbsolutePath()) + "')" + nl
				+ "preview_dir = output_dir / 'previews'" + nl
				+ "preview_manifest_path = preview_dir / 'latest.json'" + nl
				+ "output_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "preview_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "stardist_log_path = output_dir / 'training.log'" + nl
				+ "config = json.loads(r'''" + TrainingCodeUtils.toJson(config) + "''')" + nl
				+ "fine_tune_weights = config.pop('_jdll_fine_tune_weights', None)" + nl
				+ "fine_tune_source_config = config.pop('_jdll_fine_tune_source_config', None)" + nl
				+ "n_dim = int(config.pop('n_dim', 2))" + nl
				+ "auto_anisotropy = bool(config.pop('_jdll_auto_anisotropy', False))" + nl
				+ "if n_dim == 3:" + nl
				+ "  config.pop('train_shape_completion', None)" + nl
				+ "  config.pop('train_completion_crop', None)" + nl
				+ "for _derived_key in ('n_channel_out', 'net_input_shape', 'net_mask_shape'):" + nl
				+ "  config.pop(_derived_key, None)" + nl
				+ "  if fine_tune_source_config is not None:" + nl
				+ "    fine_tune_source_config.pop(_derived_key, None)" + nl
				+ "cancel_signal_path = r'" + TrainingCodeUtils.py(cancelSignalPath == null ? "" : cancelSignalPath) + "'" + nl
				+ "preview_count = int(config.pop('validation_preview_count', 20))" + nl
				+ "is_accelerated = _jdll_requested_device != 'cpu'" + nl
				+ "progress_every_n_steps = 5 if is_accelerated else 1" + nl
				+ "log_every_n_steps = 50 if is_accelerated else 10" + nl
				+ "# StarDist use_gpu enables gputools/OpenCL preprocessing, not TensorFlow CUDA." + nl
				+ "config['use_gpu'] = False" + nl
				+ "if '" + TrainingCodeUtils.py(safeImageChannels).toLowerCase() + "' == 'rgb':" + nl
				+ "  config['axes'] = 'ZYXC' if n_dim == 3 else 'YXC'" + nl
				+ "  config['n_channel_in'] = 3" + nl
				+ "else:" + nl
				+ "  config['axes'] = 'ZYXC' if n_dim == 3 else 'YXC'" + nl
				+ "  config['n_channel_in'] = 1" + nl
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
				+ "IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}" + nl
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
				+ "  return _pairs(_subdir(folder, IMAGE_DIRS), _subdir(folder, MASK_DIRS))" + nl;
		return code + "_reader_formats = {'.tif': 'TIFF', '.tiff': 'TIFF', '.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.bmp': 'BMP'}" + nl
				+ "_reader_preferences, _image_ome_metadata = {}, {}" + nl
				+ "_reader_notices = set()" + nl
				+ "def _signature_format(path):" + nl
				+ "  with path.open('rb') as stream:" + nl
				+ "    header = stream.read(8)" + nl
				+ "  if header[:4] in (b'II\\x2a\\x00', b'MM\\x00\\x2a', b'II\\x2b\\x00', b'MM\\x00\\x2b'):" + nl
				+ "    return 'TIFF'" + nl
				+ "  if header == b'\\x89PNG\\r\\n\\x1a\\n':" + nl
				+ "    return 'PNG'" + nl
				+ "  if header[:3] == b'\\xff\\xd8\\xff':" + nl
				+ "    return 'JPEG'" + nl
				+ "  if header[:2] == b'BM':" + nl
				+ "    return 'BMP'" + nl
				+ "  return None" + nl
				+ "def _read_format(path, file_format, is_mask):" + nl
				+ "  if is_mask and file_format == 'JPEG':" + nl
				+ "    raise ValueError('JPEG cannot be used as an instance-label mask')" + nl
				+ "  if file_format == 'TIFF':" + nl
				+ "    with TiffFile(str(path)) as tif:" + nl
				+ "      if len(tif.series) != 1:" + nl
				+ "        raise ValueError('Multiple TIFF series; export one image series for training')" + nl
				+ "      series = tif.series[0]" + nl
				+ "      array = np.asarray(series.asarray())" + nl
				+ "      if n_dim == 3 and not is_mask:" + nl
				+ "        _image_ome_metadata[str(path)] = tif.ome_metadata" + nl
				+ "      return array, str(series.axes).upper()" + nl
				+ "  with Image.open(path, formats=[file_format]) as image:" + nl
				+ "    if getattr(image, 'n_frames', 1) != 1:" + nl
				+ "      raise ValueError('Multiple raster frames; export a TIFF stack with explicit axes')" + nl
				+ "    if image.mode == 'P' and not is_mask:" + nl
				+ "      image = image.convert('RGB')" + nl
				+ "    array = np.asarray(image)" + nl
				+ "    return array, 'YX' if array.ndim == 2 else 'YXC'" + nl
				+ "def _read_array(path, is_mask=False):" + nl
				+ "  path = Path(path).absolute()" + nl
				+ "  key = (str(path.parent), path.suffix.lower(), is_mask)" + nl
				+ "  expected = _reader_preferences.get(key, _reader_formats.get(key[1]))" + nl
				+ "  if expected is None:" + nl
				+ "    raise ValueError('Unsupported image extension: ' + str(path))" + nl
				+ "  try:" + nl
				+ "    return _read_format(path, expected, is_mask)" + nl
				+ "  except (OSError, ValueError, TiffFileError) as error:" + nl
				+ "    actual = _signature_format(path)" + nl
				+ "    if actual is None or actual == expected:" + nl
				+ "      raise OSError('Cannot read %s as %s: %s' % (path, expected, error)) from error" + nl
				+ "    try:" + nl
				+ "      result = _read_format(path, actual, is_mask)" + nl
				+ "    except (OSError, ValueError, TiffFileError) as format_error:" + nl
				+ "      raise OSError('Cannot read %s as %s: %s' % (path, actual, format_error)) from format_error" + nl
				+ "    _reader_preferences[key] = actual" + nl
				+ "    notice = (key, actual)" + nl
				+ "    if notice not in _reader_notices:" + nl
				+ "      _reader_notices.add(notice)" + nl
				+ "      _task_update(message='Detected %s data in %s files under %s; using the %s reader (each file is still validated).' % (actual, key[1], path.parent, actual), info={'type': 'warning', 'path': str(path)})" + nl
				+ "    return result" + nl
				+ "def _canonical_array(array, source_axes, is_mask):" + nl
				+ "  array = np.asarray(array)" + nl
				+ "  axes = list((source_axes or '').upper().replace('S', 'C'))" + nl
				+ "  if len(axes) != array.ndim:" + nl
				+ "    axes = list(('ZYX' if n_dim == 3 else 'YX') + ('' if is_mask or array.ndim == n_dim else 'C'))" + nl
				+ "  if n_dim == 3 and 'Z' not in axes:" + nl
				+ "    unknown = [i for i, axis in enumerate(axes) if axis not in ('X', 'Y', 'C', 'T')]" + nl
				+ "    if len(unknown) == 1:" + nl
				+ "      axes[unknown[0]] = 'Z'" + nl
				+ "  for index in range(len(axes) - 1, -1, -1):" + nl
				+ "    axis = axes[index]" + nl
				+ "    if axis == 'T' or axis not in ('X', 'Y', 'Z', 'C'):" + nl
				+ "      if array.shape[index] != 1:" + nl
				+ "        raise ValueError('Unsupported non-spatial axis %s with size %d in shape %s' % (axis, array.shape[index], array.shape))" + nl
				+ "      array = np.take(array, 0, axis=index)" + nl
				+ "      axes.pop(index)" + nl
				+ "  spatial = list('ZYX' if n_dim == 3 else 'YX')" + nl
				+ "  if any(axis not in axes for axis in spatial):" + nl
				+ "    raise ValueError('Expected %dD data but found axes %s and shape %s' % (n_dim, ''.join(axes), array.shape))" + nl
				+ "  if is_mask and 'C' in axes:" + nl
				+ "    channel = axes.index('C')" + nl
				+ "    array = np.take(array, 0, axis=channel)" + nl
				+ "    axes.pop(channel)" + nl
				+ "  target = spatial + ([] if is_mask else ['C'])" + nl
				+ "  if not is_mask and 'C' not in axes:" + nl
				+ "    array = np.expand_dims(array, axis=-1)" + nl
				+ "    axes.append('C')" + nl
				+ "  array = np.transpose(array, [axes.index(axis) for axis in target])" + nl
				+ "  return array" + nl
				+ "def _physical_spacing(path):" + nl
				+ "  if n_dim != 3:" + nl
				+ "    return None" + nl
				+ "  try:" + nl
				+ "    xml = _image_ome_metadata.get(str(Path(path).absolute()))" + nl
				+ "    if not xml:" + nl
				+ "      return None" + nl
				+ "    pixels = next(element for element in ET.fromstring(xml).iter() if element.tag.endswith('Pixels'))" + nl
				+ "    units = [pixels.attrib.get('PhysicalSize' + axis + 'Unit') for axis in ('Z', 'Y', 'X')]" + nl
				+ "    scales = {'nm': 0.001, 'um': 1.0, 'micrometer': 1.0, 'micrometre': 1.0, 'mm': 1000.0, 'cm': 10000.0, 'm': 1000000.0}" + nl
				+ "    normalized_units = [None if unit is None else str(unit).strip().lower().replace(chr(181), 'u').replace(chr(956), 'u') for unit in units]" + nl
				+ "    if any(unit is None for unit in normalized_units) and not all(unit is None for unit in normalized_units):" + nl
				+ "      return None" + nl
				+ "    factors = [1.0 if unit is None else scales.get(unit) for unit in normalized_units]" + nl
				+ "    if any(factor is None for factor in factors):" + nl
				+ "      return None" + nl
				+ "    spacing = tuple(float(pixels.attrib['PhysicalSize' + axis]) * factor for axis, factor in zip(('Z', 'Y', 'X'), factors))" + nl
				+ "    return spacing if all(value > 0 and np.isfinite(value) for value in spacing) else None" + nl
				+ "  except Exception:" + nl
				+ "    return None" + nl
				+ "def _load_pairs(pairs):" + nl
				+ "  X, Y = [], []" + nl
				+ "  n_channels = int(config.get('n_channel_in', 1))" + nl
				+ "  for img_path, mask_path in pairs:" + nl
				+ "    raw_x, x_axes = _read_array(img_path)" + nl
				+ "    raw_y, y_axes = _read_array(mask_path, is_mask=True)" + nl
				+ "    x = _canonical_array(raw_x, x_axes, False)" + nl
				+ "    y = _canonical_array(raw_y, y_axes, True).astype(np.int32, copy=False)" + nl
				+ "    if n_channels == 1:" + nl
				+ "      x = x[..., :1]" + nl
				+ "    elif x.shape[-1] == 1:" + nl
				+ "      x = np.repeat(x, n_channels, axis=-1)" + nl
				+ "    elif x.shape[-1] == 2:" + nl
				+ "      x = np.concatenate((x, np.zeros_like(x[..., :1])), axis=-1)" + nl
				+ "    elif x.shape[-1] > n_channels:" + nl
				+ "      x = x[..., :n_channels]" + nl
				+ "    spatial_axes = tuple(range(n_dim))" + nl
				+ "    empty_channels = np.all(x == 0, axis=spatial_axes)" + nl
				+ "    x = normalize(x, 1, 99.8, axis=spatial_axes).astype(np.float32, copy=False)" + nl
				+ "    x[..., empty_channels] = 0" + nl
				+ "    X.append(x)" + nl
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
				+ "def _valid_ratio(values):" + nl
				+ "  values = np.asarray(values, dtype=np.float64)" + nl
				+ "  if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):" + nl
				+ "    return None" + nl
				+ "  return tuple((values / np.min(values)).tolist())" + nl
				+ "def _resolve_anisotropy(train_pairs, labels):" + nl
				+ "  if n_dim != 3:" + nl
				+ "    return" + nl
				+ "  extent_ratio = None" + nl
				+ "  try:" + nl
				+ "    extents = np.asarray(calculate_extents(labels), dtype=np.float64)" + nl
				+ "    if extents.shape == (3,) and np.all(extents > 0):" + nl
				+ "      extent_ratio = tuple((np.max(extents) / extents).tolist())" + nl
				+ "  except Exception:" + nl
				+ "    extents = None" + nl
				+ "  spacing = [_physical_spacing(path) for path, _mask in train_pairs]" + nl
				+ "  spacing = [value for value in spacing if value is not None]" + nl
				+ "  metadata_ratio = _valid_ratio(np.median(np.asarray(spacing), axis=0)) if spacing else None" + nl
				+ "  if not auto_anisotropy and config.get('anisotropy') is not None:" + nl
				+ "    anisotropy = tuple(float(value) for value in config['anisotropy'])" + nl
				+ "    source = 'model configuration'" + nl
				+ "  elif metadata_ratio is not None:" + nl
				+ "    anisotropy = metadata_ratio" + nl
				+ "    source = 'OME physical voxel spacing'" + nl
				+ "  elif extent_ratio is not None:" + nl
				+ "    anisotropy = extent_ratio" + nl
				+ "    source = 'training-mask object extents (roughly isotropic-object fallback)'" + nl
				+ "  else:" + nl
				+ "    anisotropy = (1.0, 1.0, 1.0)" + nl
				+ "    source = 'isotropic fallback'" + nl
				+ "  config['anisotropy'] = list(anisotropy)" + nl
				+ "  if auto_anisotropy:" + nl
				+ "    config['grid'] = [1 if value > 1.5 else 2 for value in anisotropy]" + nl
				+ "  if auto_anisotropy or not config.get('rays_json'):" + nl
				+ "    config['rays_json'] = Rays_GoldenSpiral(int(config.get('n_rays', 96)), anisotropy=anisotropy).to_json()" + nl
				+ "  comparison = '' if extent_ratio is None else ', object_extent_ratio=' + str(tuple(round(v, 4) for v in extent_ratio))" + nl
				+ "  _task_update(message='Resolved StarDist3D anisotropy from %s: %s%s; grid=%s.' % (source, tuple(round(v, 4) for v in anisotropy), comparison, tuple(config['grid'])), info={'type': 'configuration', 'anisotropy': list(anisotropy), 'anisotropy_source': source, 'grid': list(config['grid'])})" + nl
				+ "def _measure_image_instances(label, image_index):" + nl
				+ "  label = np.asarray(label)" + nl
				+ "  ids, counts = np.unique(label, return_counts=True)" + nl
				+ "  border_ids = set()" + nl
				+ "  for axis in range(n_dim):" + nl
				+ "    for edge in (0, label.shape[axis] - 1):" + nl
				+ "      border_ids.update(int(value) for value in np.unique(np.take(label, edge, axis=axis)) if value > 0)" + nl
				+ "  candidates = [int(value) for value, count in zip(ids, counts) if value > 0 and count >= 4 and int(value) not in border_ids]" + nl
				+ "  available = len(candidates)" + nl
				+ "  if available > 21:" + nl
				+ "    candidates = random.Random(5489 + image_index).sample(candidates, 21)" + nl
				+ "  anisotropy = np.asarray(config.get('anisotropy', [1.0, 1.0, 1.0]), dtype=np.float64) if n_dim == 3 else None" + nl
				+ "  diameters, extents = [], []" + nl
				+ "  for instance_id in candidates:" + nl
				+ "    coordinates = np.argwhere(label == instance_id)" + nl
				+ "    if coordinates.size == 0:" + nl
				+ "      continue" + nl
				+ "    extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1" + nl
				+ "    count = float(coordinates.shape[0])" + nl
				+ "    diameter = (2.0 * np.sqrt(count / np.pi) if n_dim == 2 else 2.0 * (3.0 * count * np.prod(anisotropy) / (4.0 * np.pi)) ** (1.0 / 3.0))" + nl
				+ "    diameters.append(float(diameter))" + nl
				+ "    extents.append(extent.astype(np.float64))" + nl
				+ "  if not diameters:" + nl
				+ "    return None" + nl
				+ "  return {'diameter': float(np.median(diameters)), 'extent': np.median(np.asarray(extents), axis=0), 'sampled': len(diameters), 'available': available}" + nl
				+ "def _aggregate_object_statistics(labels):" + nl
				+ "  measured = [value for value in (_measure_image_instances(label, index) for index, label in enumerate(labels)) if value is not None]" + nl
				+ "  if not measured:" + nl
				+ "    return {'images_measured': 0, 'images_without_valid_instances': len(labels)}" + nl
				+ "  diameters = np.asarray([value['diameter'] for value in measured], dtype=np.float64)" + nl
				+ "  extents = np.asarray([value['extent'] for value in measured], dtype=np.float64)" + nl
				+ "  sampled = np.asarray([value['sampled'] for value in measured], dtype=np.int64)" + nl
				+ "  available = np.asarray([value['available'] for value in measured], dtype=np.int64)" + nl
				+ "  return {'images_measured': len(measured), 'images_without_valid_instances': len(labels) - len(measured)," + nl
				+ "    'median_object_diameter_px': float(np.median(diameters)), 'object_diameter_p10_px': float(np.percentile(diameters, 10))," + nl
				+ "    'object_diameter_p90_px': float(np.percentile(diameters, 90)), 'median_extent_px': [float(value) for value in np.median(extents, axis=0)]," + nl
				+ "    'median_instances_sampled_per_image': float(np.median(sampled)), 'minimum_instances_sampled_per_image': int(sampled.min())," + nl
				+ "    'maximum_instances_sampled_per_image': int(sampled.max()), 'median_available_instances_per_image': float(np.median(available))}" + nl
				+ "def _save_object_statistics(labels):" + nl
				+ "  statistics_path = output_dir / 'dataset_statistics.json'" + nl
				+ "  metadata_path = output_dir / 'model_metadata.json'" + nl
				+ "  if metadata_path.exists():" + nl
				+ "    metadata_path.unlink()" + nl
				+ "  training = _aggregate_object_statistics(labels)" + nl
				+ "  payload = {'instance_scale_statistics': {'training': training, 'settings': {'measure': 'equivalent_circle_diameter' if n_dim == 2 else 'equivalent_sphere_diameter', 'max_instances_per_image': 21, 'exclude_border_instances': True, 'min_instance_pixels_or_voxels': 4}}}" + nl
				+ "  with open(statistics_path, 'w', encoding='utf-8') as f:" + nl
				+ "    json.dump(payload, f, indent=2)" + nl
				+ "  diameter = training.get('median_object_diameter_px')" + nl
				+ "  if diameter is None:" + nl
				+ "    _task_update(message='Could not estimate a robust StarDist object size; legacy inference scaling will be used.', info={'type': 'warning', 'path': str(statistics_path)})" + nl
				+ "    return" + nl
				+ "  instance_scale = {'dimensions': n_dim, 'measure': 'equivalent_circle_diameter' if n_dim == 2 else 'equivalent_sphere_diameter'," + nl
				+ "    'size_unit': 'pixels' if n_dim == 2 else 'minimum_voxel_spacing_pixels', 'model_object_diameter_px': diameter," + nl
				+ "    'object_diameter_p10_px': training['object_diameter_p10_px'], 'object_diameter_p90_px': training['object_diameter_p90_px']," + nl
				+ "    'median_extent_px': training['median_extent_px']}" + nl
				+ "  if n_dim == 3:" + nl
				+ "    instance_scale['anisotropy'] = [float(value) for value in config.get('anisotropy', [1.0, 1.0, 1.0])]" + nl
				+ "  with open(metadata_path, 'w', encoding='utf-8') as f:" + nl
				+ "    json.dump({'schema_version': 1, 'model_family': 'stardist', 'instance_scale': instance_scale}, f, indent=2)" + nl
				+ "  _task_update(message='Measured StarDist median object diameter: %.3f %s. Saved model scale metadata at: %s' % (diameter, instance_scale['size_unit'], metadata_path), info={'type': 'configuration', 'object_diameter_px': diameter, 'path': str(metadata_path)})" + nl
				+ "def _adapt_input_kernel(kernel, target_shape):" + nl
				+ "  source_channels = int(kernel.shape[-2])" + nl
				+ "  target_channels = int(target_shape[-2])" + nl
				+ "  if source_channels == 1 and target_channels > 1:" + nl
				+ "    return np.repeat(kernel, target_channels, axis=-2) / float(target_channels)" + nl
				+ "  if target_channels == 1:" + nl
				+ "    return np.sum(kernel, axis=-2, keepdims=True)" + nl
				+ "  adapted = np.zeros(target_shape, dtype=kernel.dtype)" + nl
				+ "  copied = min(source_channels, target_channels)" + nl
				+ "  adapted[..., :copied, :] = kernel[..., :copied, :]" + nl
				+ "  return adapted" + nl
				+ "def _load_fine_tune_weights(model_ref):" + nl
				+ "  source_channels = int(fine_tune_source_config.get('n_channel_in', 1))" + nl
				+ "  target_channels = int(config.get('n_channel_in', 1))" + nl
				+ "  if source_channels == target_channels:" + nl
				+ "    model_ref.keras_model.load_weights(fine_tune_weights)" + nl
				+ "    return 'Loaded all source weights without channel adaptation.'" + nl
				+ "  with tf.device('/CPU:0'):" + nl
				+ "    source_type = StarDist3D if n_dim == 3 else StarDist2D" + nl
				+ "    config_type = Config3D if n_dim == 3 else Config2D" + nl
				+ "    source_model = source_type(config_type(**fine_tune_source_config), name=None, basedir=None)" + nl
				+ "    source_model.keras_model.load_weights(fine_tune_weights)" + nl
				+ "  source_layers = [layer for layer in source_model.keras_model.layers if layer.get_weights()]" + nl
				+ "  target_layers = [layer for layer in model_ref.keras_model.layers if layer.get_weights()]" + nl
				+ "  if len(source_layers) != len(target_layers):" + nl
				+ "    raise ValueError('Source and target StarDist architectures have different weighted layer counts.')" + nl
				+ "  adapted_first = False" + nl
				+ "  for source_layer, target_layer in zip(source_layers, target_layers):" + nl
				+ "    source_weights = source_layer.get_weights()" + nl
				+ "    target_weights = target_layer.get_weights()" + nl
				+ "    if len(source_weights) == len(target_weights) and all(a.shape == b.shape for a, b in zip(source_weights, target_weights)):" + nl
				+ "      target_layer.set_weights(source_weights)" + nl
				+ "      continue" + nl
				+ "    kernel_compatible = (not adapted_first and len(source_weights) == 2 and len(target_weights) == 2" + nl
				+ "      and source_weights[0].ndim in (4, 5) and source_weights[0].ndim == target_weights[0].ndim" + nl
				+ "      and source_weights[0].shape[:-2] == target_weights[0].shape[:-2]" + nl
				+ "      and source_weights[0].shape[-1] == target_weights[0].shape[-1]" + nl
				+ "      and source_weights[1].shape == target_weights[1].shape)" + nl
				+ "    if not kernel_compatible:" + nl
				+ "      raise ValueError('Source and target StarDist architectures differ beyond the input channels.')" + nl
				+ "    target_layer.set_weights([_adapt_input_kernel(source_weights[0], target_weights[0].shape), source_weights[1]])" + nl
				+ "    adapted_first = True" + nl
				+ "  del source_model" + nl
				+ "  if not adapted_first:" + nl
				+ "    raise ValueError('Could not locate the StarDist input convolution to adapt.')" + nl
				+ "  return 'Adapted the first convolution from %d to %d channels.' % (source_channels, target_channels)" + nl
				+ "def _baseline_validation(model_ref, X_val, Y_val):" + nl
				+ "  try:" + nl
				+ "    n_take = config.get('train_n_val_patches')" + nl
				+ "    n_take = len(X_val) if n_take is None else min(int(n_take), len(X_val))" + nl
				+ "    if n_dim == 3:" + nl
				+ "      from stardist.models.model3d import StarDistData3D" + nl
				+ "      from stardist.rays3d import rays_from_json" + nl
				+ "      validation = StarDistData3D(X_val, Y_val, batch_size=n_take, length=1," + nl
				+ "        rays=rays_from_json(model_ref.config.rays_json), patch_size=config['train_patch_size']," + nl
				+ "        grid=config['grid'], anisotropy=config.get('anisotropy'), use_gpu=False," + nl
				+ "        foreground_prob=config.get('train_foreground_only', 0.9)," + nl
				+ "        n_classes=config.get('n_classes'), sample_ind_cache=config.get('train_sample_cache', True))[0]" + nl
				+ "    else:" + nl
				+ "      from stardist.models.model2d import StarDistData2D" + nl
				+ "      validation = StarDistData2D(X_val, Y_val, batch_size=n_take, length=1," + nl
				+ "        n_rays=config['n_rays'], patch_size=config['train_patch_size'], grid=config['grid']," + nl
				+ "        shape_completion=config.get('train_shape_completion', False)," + nl
				+ "        b=(config.get('train_completion_crop', 32) if config.get('train_shape_completion', False) else 0), use_gpu=False," + nl
				+ "        foreground_prob=config.get('train_foreground_only', 0.9)," + nl
				+ "        n_classes=config.get('n_classes'), sample_ind_cache=config.get('train_sample_cache', True))[0]" + nl
				+ "    values = model_ref.keras_model.evaluate(validation[0], validation[1], verbose=0, return_dict=True)" + nl
				+ "    clean = _clean(values)" + nl
				+ "    _task_update(message='Fine-tuning baseline validation: ' + json.dumps(clean, sort_keys=True), info={'type': 'baseline', 'metrics': clean})" + nl
				+ "  except Exception as baseline_error:" + nl
				+ "    _task_update(message='Could not calculate fine-tuning baseline validation: ' + str(baseline_error), info={'type': 'warning', 'message': str(baseline_error)})" + nl
				+ "def _preview_region(image, label, full_volume):" + nl
				+ "  if n_dim != 3 or full_volume:" + nl
				+ "    return image, label" + nl
				+ "  patch = tuple(int(value) for value in config.get('train_patch_size', label.shape))" + nl
				+ "  coords = np.argwhere(label > 0)" + nl
				+ "  center = np.asarray(label.shape, dtype=np.int64) // 2 if len(coords) == 0 else np.median(coords, axis=0).astype(np.int64)" + nl
				+ "  slices = []" + nl
				+ "  for axis, size in enumerate(patch):" + nl
				+ "    size = min(int(size), int(label.shape[axis]))" + nl
				+ "    start = max(0, min(int(center[axis]) - size // 2, int(label.shape[axis]) - size))" + nl
				+ "    slices.append(slice(start, start + size))" + nl
				+ "  return image[tuple(slices) + (slice(None),)], label[tuple(slices)]" + nl
				+ "def _preview_tiles(image):" + nl
				+ "  patch = tuple(int(value) for value in config.get('train_patch_size', image.shape[:n_dim]))" + nl
				+ "  spatial = tuple(max(1, int(np.ceil(image.shape[axis] / float(patch[axis])))) for axis in range(n_dim))" + nl
				+ "  return spatial + ((1,) if image.ndim > n_dim else ())" + nl
				+ "def _initial_plane(label):" + nl
				+ "  if n_dim != 3:" + nl
				+ "    return 0" + nl
				+ "  counts = np.count_nonzero(label > 0, axis=(1, 2))" + nl
				+ "  return int(np.argmax(counts)) if np.any(counts) else int(label.shape[0] // 2)" + nl
				+ "class JDLLProgressCallback(Callback):" + nl
				+ "  def __init__(self, model_ref, X_val, Y_val):" + nl
				+ "    super().__init__()" + nl
				+ "    self.model_ref = model_ref" + nl
				+ "    self.X_val = X_val" + nl
				+ "    self.Y_val = Y_val" + nl
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
				+ "      except Exception as preview_error:" + nl
				+ "        _task_update(message='Could not generate StarDist validation prediction: ' + str(preview_error), info={'type': 'warning', 'message': str(preview_error)})" + nl
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
				+ "    full_volume = n_dim == 3 and current_epoch == state['total_epochs']" + nl
				+ "    for i, (source_image, source_label) in enumerate(zip(self.X_val[:preview_count], self.Y_val[:preview_count])):" + nl
				+ "      image, label = _preview_region(source_image, source_label, full_volume)" + nl
				+ "      image_path = preview_dir / ('preview_%03d_image.npy' % i)" + nl
				+ "      label_path = preview_dir / ('preview_%03d_label.npy' % i)" + nl
				+ "      pred_path = preview_dir / ('preview_%03d_prediction.npy' % i)" + nl
				+ "      prob_path = preview_dir / ('preview_%03d_prob.npy' % i)" + nl
				+ "      sample = {'index': i}" + nl
				+ "      _atomic_npy_save(image_path, image)" + nl
				+ "      _atomic_npy_save(label_path, np.asarray(label, dtype=np.int32))" + nl
				+ "      sample['image_path'] = str(image_path)" + nl
				+ "      sample['label_path'] = str(label_path)" + nl
				+ "      sample['axes'] = str(config.get('axes', 'ZYXC' if n_dim == 3 else 'YXC'))" + nl
				+ "      sample['initial_plane'] = {'axis': 'z', 'index': _initial_plane(label)} if n_dim == 3 else None" + nl
				+ "      sample['full_volume'] = bool(full_volume)" + nl
				+ "      try:" + nl
				+ "        preview_tiles = _preview_tiles(image)" + nl
				+ "        prediction, details = self.model_ref.predict_instances(image, axes=str(config.get('axes', 'YXC')), normalizer=None, n_tiles=preview_tiles, show_tile_progress=False)" + nl
				+ "        _atomic_npy_save(pred_path, np.asarray(prediction, dtype=np.int32))" + nl
				+ "        sample['prediction_path'] = str(pred_path)" + nl
				+ "        prob, _dist = self.model_ref.predict(image, axes=str(config.get('axes', 'YXC')), normalizer=None, n_tiles=preview_tiles, show_tile_progress=False)" + nl
				+ "        _atomic_npy_save(prob_path, prob)" + nl
				+ "        sample['prob_path'] = str(prob_path)" + nl
				+ "      except Exception:" + nl
				+ "        pass" + nl
				+ "      samples.append(sample)" + nl
				+ "    if samples:" + nl
				+ "      manifest = {'epoch': current_epoch, 'n_dim': n_dim, 'axes': str(config.get('axes')), 'anisotropy': config.get('anisotropy'), 'samples': samples}" + nl
				+ "      with open(preview_manifest_path, 'w', encoding='utf-8') as f:" + nl
				+ "        json.dump(manifest, f)" + nl
				+ "      _task_update(message='StarDist validation preview epoch %d' % current_epoch, current=current_epoch, maximum=state['total_epochs'], info={'type': 'preview', 'epoch': current_epoch, 'preview_path': str(preview_manifest_path)})" + nl
				+ "train_pairs, val_pairs = _dataset()" + nl
				+ "X_train, Y_train = _load_pairs(train_pairs)" + nl
				+ "X_val, Y_val = _load_pairs(val_pairs)" + nl
				+ "_resolve_anisotropy(train_pairs, Y_train)" + nl
				+ "_save_object_statistics(Y_train)" + nl
				+ "with open(stardist_log_path, 'a', encoding='utf-8') as stardist_log, contextlib.redirect_stdout(stardist_log), contextlib.redirect_stderr(stardist_log), tf.device(_jdll_tf_device):" + nl
				+ "  config_type = Config3D if n_dim == 3 else Config2D" + nl
				+ "  model_type = StarDist3D if n_dim == 3 else StarDist2D" + nl
				+ "  model_config = config_type(**config)" + nl
				+ "  model = model_type(model_config, name=output_dir.name, basedir=str(output_dir.parent))" + nl
				+ "  if fine_tune_weights is not None:" + nl
				+ "    transfer_message = _load_fine_tune_weights(model)" + nl
				+ "    _task_update(message=transfer_message, info={'type': 'fine_tune', 'message': transfer_message})" + nl
				+ "  model.prepare_for_training()" + nl
				+ "  if fine_tune_weights is not None:" + nl
				+ "    _baseline_validation(model, X_val, Y_val)" + nl
				+ "  model.callbacks.append(JDLLProgressCallback(model, X_val, Y_val))" + nl
				+ "  best_path = output_dir / str(config.get('train_checkpoint', 'weights_best.h5'))" + nl
				+ "  last_path = output_dir / str(config.get('train_checkpoint_last', 'weights_last.h5'))" + nl
				+ "  if best_path.exists():" + nl
				+ "    _task_update(message='Overwriting StarDist best checkpoint during training: ' + str(best_path), info={'type': 'checkpoint', 'kind': 'best', 'path': str(best_path), 'overwrite': True})" + nl
				+ "  if last_path.exists():" + nl
				+ "    _task_update(message='Overwriting StarDist last checkpoint during training: ' + str(last_path), info={'type': 'checkpoint', 'kind': 'last', 'path': str(last_path), 'overwrite': True})" + nl
				+ "  history = model.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=int(config.get('train_epochs', 1)), steps_per_epoch=int(config.get('train_steps_per_epoch', 100)), workers=0)" + nl
				+ "  if best_path.exists():" + nl
				+ "    _task_update(message='StarDist best checkpoint: ' + str(best_path), info={'type': 'checkpoint', 'kind': 'best', 'path': str(best_path)})" + nl
				+ "  if last_path.exists():" + nl
				+ "    _task_update(message='StarDist last checkpoint: ' + str(last_path), info={'type': 'checkpoint', 'kind': 'last', 'path': str(last_path)})" + nl
				+ "  if not _cancel_requested() and best_path.exists():" + nl
				+ "    try:" + nl
				+ "      model.keras_model.load_weights(str(best_path))" + nl
				+ "      _task_update(message='Optimizing StarDist probability and NMS thresholds on validation data.', info={'type': 'thresholds', 'status': 'started'})" + nl
				+ "      thresholds = model.optimize_thresholds(X_val, Y_val, save_to_json=True)" + nl
				+ "      thresholds_path = output_dir / 'thresholds.json'" + nl
				+ "      _task_update(message='Saved optimized StarDist thresholds at: ' + str(thresholds_path), info={'type': 'thresholds', 'status': 'finished', 'path': str(thresholds_path)})" + nl
				+ "    except Exception as threshold_error:" + nl
				+ "      _task_update(message='Could not optimize StarDist thresholds: ' + str(threshold_error), info={'type': 'warning', 'message': str(threshold_error)})" + nl
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
