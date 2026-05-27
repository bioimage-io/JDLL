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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.TaskException;
import org.apposed.appose.util.Messages;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentResolver;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Unified StarDist model entry point.
 */
public final class StarDist extends DLModelPytorchProtected {

	public enum Dimensionality {
		TWO_D,
		THREE_D
	}

	private static final String PIXI_TOML = "tomls/cellcast-pixi.toml";
	private static final String COMMON_CELLCAST_ENV_NAME = "cellcast-jdll";
	private static final String CELLCAST_WHEEL_RESOURCE_DIR = "wheels/cellcast";
	private static final String WHEELS_CACHE_DIR_NAME = "wheels";

	private static final String OUTPUT_MASK_KEY = "mask";
	private static final String SHM_NAME_KEY = "_shm_name";
	private static final String DTYPE_KEY = "_dtype";
	private static final String SHAPE_KEY = "_shape";
	private static final String KEYS_KEY = "keys";

	private static final String LOAD_MODEL_CODE_2D = ""
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  task.export(os=os)" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  task.export(shared_memory=shared_memory)" + System.lineSeparator()
			+ "if 'train' not in globals().keys():" + System.lineSeparator()
			+ "  import cellcast.training.stardist_2d as train" + System.lineSeparator()
			+ "  task.export(train=train)" + System.lineSeparator()
			+ "model = train.load_stardist_2d(source=%s, config=%s, gpu=%s)" + System.lineSeparator()
			+ "task.export(model=model)" + System.lineSeparator();

	private static final String RUN_MODEL_CODE = ""
			+ "output = model.predict_instances(im, return_predict=False)" + System.lineSeparator()
			+ "if type(output) == np.ndarray:" + System.lineSeparator()
			+ "  im[:] = output" + System.lineSeparator()
			+ "  if os.name == 'nt':" + System.lineSeparator()
			+ "    im_shm.close()" + System.lineSeparator()
			+ "    im_shm.unlink()" + System.lineSeparator()
			+ "if type(output) != list and type(output) != tuple:" + System.lineSeparator()
			+ "  raise TypeError('StarDist output should be a list or tuple.')" + System.lineSeparator()
			+ "if type(output[0]) != np.ndarray:" + System.lineSeparator()
			+ "  raise TypeError('The first StarDist output should be a np.ndarray.')" + System.lineSeparator()
			+ "if len(im.shape) == 3 and len(output[0].shape) == 2:" + System.lineSeparator()
			+ "  im[:, :, 0] = output[0]" + System.lineSeparator()
			+ "elif len(im.shape) == 4 and len(output[0].shape) == 3:" + System.lineSeparator()
			+ "  im[:, :, :, 0] = output[0]" + System.lineSeparator()
			+ "else:" + System.lineSeparator()
			+ "  im[:] = output[0]" + System.lineSeparator()
			+ "details = output[1] if len(output) > 1 else {}" + System.lineSeparator()
			+ "if type(details) != dict:" + System.lineSeparator()
			+ "  raise TypeError('If StarDist returns metadata, it must be a dict.')" + System.lineSeparator()
			+ "task.outputs['" + KEYS_KEY + "'] = list(details.keys())" + System.lineSeparator()
			+ "shm_list = []" + System.lineSeparator()
			+ "np_list = []" + System.lineSeparator()
			+ "for kk, vv in details.items():" + System.lineSeparator()
			+ "  if type(vv) != np.ndarray:" + System.lineSeparator()
			+ "    task.update('Output ' + kk + ' is not a np.ndarray. Only np.ndarrays supported.')" + System.lineSeparator()
			+ "    continue" + System.lineSeparator()
			+ "  if vv.nbytes == 0:" + System.lineSeparator()
			+ "    task.outputs[kk] = None" + System.lineSeparator()
			+ "  else:" + System.lineSeparator()
			+ "    task.outputs[kk + '" + SHAPE_KEY + "'] = vv.shape" + System.lineSeparator()
			+ "    task.outputs[kk + '" + DTYPE_KEY + "'] = str(vv.dtype)" + System.lineSeparator()
			+ "    shm = shared_memory.SharedMemory(create=True, size=vv.nbytes)" + System.lineSeparator()
			+ "    task.outputs[kk + '" + SHM_NAME_KEY + "'] = shm.name" + System.lineSeparator()
			+ "    shm_list.append(shm)" + System.lineSeparator()
			+ "    aa = np.ndarray(vv.shape, dtype=vv.dtype, buffer=shm.buf)" + System.lineSeparator()
			+ "    aa[:] = vv" + System.lineSeparator()
			+ "    np_list.append(aa)" + System.lineSeparator()
			+ "globals()['shm_list'] = shm_list" + System.lineSeparator()
			+ "globals()['np_list'] = np_list" + System.lineSeparator()
			+ "if os.name == 'nt':" + System.lineSeparator()
			+ "  im_shm.close()" + System.lineSeparator()
			+ "  im_shm.unlink()" + System.lineSeparator();

	private static final String CLOSE_SHM_CODE = ""
			+ "if 'np_list' in globals().keys():" + System.lineSeparator()
			+ "  for a in np_list:" + System.lineSeparator()
			+ "    del a" + System.lineSeparator()
			+ "if 'shm_list' in globals().keys():" + System.lineSeparator()
			+ "  for s in shm_list:" + System.lineSeparator()
			+ "    s.unlink()" + System.lineSeparator()
			+ "    del s" + System.lineSeparator();

	private final String mpkPath;
	private final int nChannels;
	private final Dimensionality dimensionality;
	private final Map<String, Object> config;

	private Service python;
	private SharedMemoryArray shma;
	private Double threshold = null;
	private Consumer<InferenceProgress> inferenceProgressConsumer;

	private StarDist(String modelIdentity, Map<String, Object> config,
			Dimensionality dimensionality) throws IOException {
		super(modelIdentity, modelIdentity, modelIdentity, modelIdentity, config, true);
		modelFolder = new File(modelIdentity).getParentFile().getAbsolutePath();
		this.mpkPath = modelIdentity;
		this.config = normalizedConfig(config);
		this.nChannels = inferNChannels(this.config);
		this.dimensionality = dimensionality;
		this.environmentSpec = resolvePytorchEnv();
	}

	public static StarDist fromFile(String modelPath)
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = loadModelConfig(modelPath);
		String modelIdentity = resolveModelIdentityFile(modelPath).getAbsolutePath();
		StarDist model = new StarDist(modelIdentity, config, inferDimensionality(config));
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
				inferDimensionality(normalized));
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
				Dimensionality.TWO_D);
		model.loadModel();
		return model;
	}

	public static StarDist fromDefault3D()
			throws IOException, BuildException, LoadModelException {
		Map<String, Object> config = defaultModelConfig3D();
		StarDist model = new StarDist("stardist-default-3d", config,
				Dimensionality.THREE_D);
		model.loadModel();
		return model;
	}

	public Dimensionality getDimensionality() {
		return dimensionality;
	}

	public Map<String, Object> getConfig() {
		return new LinkedHashMap<String, Object>(config);
	}

	public boolean is2D() {
		return dimensionality == Dimensionality.TWO_D;
	}

	public boolean is3D() {
		return dimensionality == Dimensionality.THREE_D;
	}

	public int getNChannels() {
		return nChannels;
	}

	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}

	public void setInferenceProgressConsumer(Consumer<InferenceProgress> consumer) {
		this.inferenceProgressConsumer = consumer;
	}

	public Consumer<InferenceProgress> getInferenceProgressConsumer() {
		return inferenceProgressConsumer;
	}

	@Override
	public void close() {
		if (!loaded) {
			return;
		}
		python.close();
		loaded = false;
		closed = true;
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run(List<Tensor<T>> inTensors, List<Tensor<R>> outTensors) throws RunModelException {
		throw new RunModelException("StarDist currently exposes RandomAccessibleInterval inference.");
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		throw new RunModelException("StarDist currently exposes RandomAccessibleInterval inference.");
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	Map<String, RandomAccessibleInterval<R>> run(RandomAccessibleInterval<T> img)
			throws InterruptedException, TaskException, IOException {
		if (!loaded || python == null) {
			try {
				loadModel();
			} catch (LoadModelException e) {
				throw new IOException("Could not load StarDist model before inference.", e);
			}
		}
		checkInput(img);
		shma = SharedMemoryArray.createSHMAFromRAI(img, false, false);
		String code = createEncodeImageScript() + System.lineSeparator();
		if (this.threshold != null) {
			code += String.format("model.thresholds = dict(prob=%s, nms=model.thresholds.nms)", threshold)
					+ System.lineSeparator();
		}
		code += RUN_MODEL_CODE + System.lineSeparator();

		Task task = python.task(code);
		task.waitFor();
		ensureTaskSucceeded(task);
		return reconstructOutputs(task);
	}

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

	public void train(String dataDir, String gtDir, String outputDir, int epochs,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, true, "grayscale", "grayscale", 0.15d,
				defaultTrainingConfig(epochs), progressConsumer, previewConsumer, logConsumer);
	}

	public void train(String dataDir, String outputDir, int epochs,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, null, outputDir, epochs, progressConsumer, previewConsumer, logConsumer);
	}

	@Override
	protected String buildModelCode() {
		String gpu = "True"; // TODO
		String source = mpkPath != null ? "r'" + mpkPath + "'" : "None";
		String configStr = TrainingCodeUtils.toJson(config).replace("null", "None").replace("true", "True").replace("false", "False");
		return String.format(LOAD_MODEL_CODE_2D, source, configStr, gpu);
	}

	private <T extends RealType<T> & NativeType<T>> void checkInput(RandomAccessibleInterval<T> image) {
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

	private <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() {
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

	private String createEncodeImageScript() {
		String code = "";
		code += "im_shm = shared_memory.SharedMemory(name='"
				+ shma.getNameForPython() + "', size=" + shma.getSize()
				+ ")" + System.lineSeparator();
		long nElems = 1;
		for (long elem : shma.getOriginalShape()) {
			nElems *= elem;
		}
		code += "im = np.ndarray(" + nElems + ", dtype='"
				+ CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
				+ "', buffer=im_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i++) {
			code += shma.getOriginalShape()[i] + ", ";
		}
		code += "])" + System.lineSeparator();
		return code;
	}

	protected <T extends RealType<T> & NativeType<T>>
	Map<String, RandomAccessibleInterval<T>> reconstructOutputs(Task task)
			throws IOException {
		Map<String, RandomAccessibleInterval<T>> outs = new LinkedHashMap<String, RandomAccessibleInterval<T>>();
		outs.put(OUTPUT_MASK_KEY, reconstructMask());

		if (task.outputs.get(KEYS_KEY) != null) {
			for (String kk : (List<String>) task.outputs.get(KEYS_KEY)) {
				outs.put(kk, reconstruct(task, kk));
			}
		}

		if (PlatformDetection.isWindows()) {
			python.task(CLOSE_SHM_CODE);
		}
		return outs;
	}

	private <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> reconstruct(Task task, String key) throws IOException {
		String shmName = (String) task.outputs.get(key + SHM_NAME_KEY);
		String dtype = (String) task.outputs.get(key + DTYPE_KEY);
		List<Number> shape = (List<Number>) task.outputs.get(key + SHAPE_KEY);
		if (shape == null) {
			return null;
		}
		long[] dims = new long[shape.size()];
		for (int i = 0; i < dims.length; i++) {
			dims[i] = shape.get(i).longValue();
		}
		SharedMemoryArray shmOut = SharedMemoryArray.readOrCreate(shmName, dims,
				Cast.unchecked(CommonUtils.getImgLib2DataType(dtype)), false, false);
		RandomAccessibleInterval<T> rai = shmOut.getSharedRAI();
		RandomAccessibleInterval<T> copy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai),
				Util.getTypeFromInterval(Cast.unchecked(rai)));
		shmOut.close();
		return copy;
	}

	private static void ensureTaskSucceeded(Task task) {
		if (task.status == TaskStatus.CANCELED) {
			throw new RuntimeException("Task canceled");
		}
		if (task.status == TaskStatus.FAILED || task.status == TaskStatus.CRASHED) {
			throw new RuntimeException(task.error);
		}
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
			File mpk = findMpkFile(file);
			if (mpk != null) {
				return mpk.getAbsoluteFile();
			}
		}
		throw new IllegalArgumentException("Path provided does not point to a StarDist .mpk model: " + modelPath);
	}

	private static File findMpkFile(File modelPath) {
		if (modelPath == null) {
			return null;
		}
		if (modelPath.isFile()
				&& modelPath.getName().toLowerCase().endsWith(StardistModelRegistry.STARDIST_WEIGHTS_EXTENSION)) {
			return modelPath;
		}
		if (!modelPath.isDirectory()) {
			return null;
		}
		java.nio.file.Path mpk = StardistModelRegistry.findMpk(modelPath.getAbsolutePath());
		return mpk == null ? null : mpk.toFile();
	}

	private static File createPlaceholderModelFile(String modelName) throws IOException {
		File cacheDir = new File(PixiEnvironmentResolver.userCacheDir("jdll"),
				"stardist" + File.separator + "placeholders");
		if (!cacheDir.isDirectory() && !cacheDir.mkdirs()) {
			throw new IOException("Could not create StarDist placeholder directory: "
					+ cacheDir.getAbsolutePath());
		}
		String safeName = modelName == null ? "stardist" : modelName.replaceAll("[^A-Za-z0-9._-]", "_");
		File marker = new File(cacheDir, safeName + ".placeholder");
		if (!marker.isFile() && !marker.createNewFile()) {
			throw new IOException("Could not create StarDist placeholder file: " + marker.getAbsolutePath());
		}
		return marker;
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

	public static PixiEnvironmentSpec resolvePytorchEnv() {
		String wheelResource = PixiEnvironmentResolver.selectResourceByCurrentPlatform(
				CELLCAST_WHEEL_RESOURCE_DIR + "/cellcast-0.2.1.dev0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
				CELLCAST_WHEEL_RESOURCE_DIR + "/cellcast-0.2.1.dev0-cp37-abi3-manylinux_2_28_aarch64.whl",
				CELLCAST_WHEEL_RESOURCE_DIR + "/cellcast-0.2.1.dev0-cp37-abi3-macosx_10_12_x86_64.whl",
				CELLCAST_WHEEL_RESOURCE_DIR + "/cellcast-0.2.1.dev0-cp37-abi3-macosx_11_0_arm64.whl",
				CELLCAST_WHEEL_RESOURCE_DIR + "/cellcast-0.2.1.dev0-cp37-abi3-win_amd64.whl",
				"CellCast");
		File cellcastWheel = PixiEnvironmentResolver.cacheClasspathResource(wheelResource, WHEELS_CACHE_DIR_NAME);
		return PixiEnvironmentResolver.fromTemplate(COMMON_CELLCAST_ENV_NAME, PIXI_TOML,
				COMMON_CELLCAST_ENV_NAME,
				PixiEnvironmentResolver.currentPixiPlatform(),
				PixiEnvironmentResolver.toPixiPath(cellcastWheel));
	}

	public static boolean isInstalled() {
		try {
			return PixiEnvironmentManager.isInstalled(resolvePytorchEnv());
		} catch (Exception e) {
			return false;
		}
	}

	public static void installDefaultRequirements() throws InterruptedException, BuildException {
		installDefaultRequirements(null);
	}

	public static void installDefaultRequirements(Consumer<String> consumer)
			throws InterruptedException, BuildException {
		PixiEnvironmentSpec spec = resolvePytorchEnv();
		PixiEnvironmentManager.installRequirements(spec, consumer == null ? s -> { } : consumer);
		if (!isInstalled()) {
			throw new RuntimeException("Not all the required packages were installed correctly. Please try again."
					+ " If the error persists, please post an issue at: https://github.com/bioimage-io/JDLL/issues");
		}
	}

	public static void train(int epochs, String dataDir, String gtDir, String outputDir,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		train(dataDir, gtDir, outputDir, true, "grayscale", "grayscale", 0.15d,
				defaultTrainingConfig(epochs), progressConsumer, previewConsumer, logConsumer);
	}

	public static void train(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		validateTrainingArguments(dataDir, gtDir, outputDir, validFraction, config);
		File output = new File(outputDir);
		if (!output.isDirectory() && !output.mkdirs()) {
			throw new IOException("Could not create StarDist output directory: " + output.getAbsolutePath());
		}

		PixiEnvironmentSpec envSpec = resolvePytorchEnv();
		Environment env = Appose.pixi()
				.environment(envSpec.getSelectedEnvironment())
				.wrap(envSpec.getEnvironmentDirectory());
		Service python = env.python();
		python.init("import numpy as np");
		if (logConsumer != null) {
			python.debug(logConsumer);
		}
		try {
			Task task = python.task(buildTrainingCode(dataDir, gtDir, outputDir, gpu,
					imageChannels, labelColorMode, validFraction, config));
			task.listen(event -> handleTrainingEvent(event, progressConsumer, previewConsumer, logConsumer));
			task.waitFor();
		} finally {
			if (python.isAlive()) {
				python.close();
			}
		}
	}

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
		config.put("train_tensorboard", true);
		config.put("train_reduce_lr", reduceLrConfig());
		config.put("use_gpu", false);
		config.put("validation_preview_count", 20);
		return config;
	}

	public static Map<String, Object> defaultModelConfig2D() {
		Map<String, Object> config = new LinkedHashMap<String, Object>();
		config.put("axes", "YXC");
		config.put("n_channel_in", 1);
		config.put("n_rays", 32);
		config.put("grid", Arrays.asList(1, 1));
		config.put("backbone", "unet");
		return config;
	}

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

	public static String buildTrainingCode(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config) {
		String nl = System.lineSeparator();
		boolean hasGtDir = gtDir != null && !gtDir.trim().isEmpty();
		String gtDirCode = hasGtDir ? "gt_dir = r'" + TrainingCodeUtils.py(new File(gtDir).getAbsolutePath()) + "'" + nl : "";
		String gtDirArgument = hasGtDir ? "gt_dir=gt_dir, " : "";
		String safeImageChannels = imageChannels == null || imageChannels.trim().isEmpty()
				? "grayscale" : imageChannels.trim();
		String safeLabelColorMode = labelColorMode == null || labelColorMode.trim().isEmpty()
				? "grayscale" : labelColorMode.trim();
		return ""
				+ "import contextlib, json, os, sys" + nl
				+ "from pathlib import Path" + nl
				+ "import numpy as np" + nl
				+ TrainingCodeUtils.apposeStdoutCapture()
				+ "import cellcast.training.stardist_2d as train" + nl
				+ "data_dir = r'" + TrainingCodeUtils.py(new File(dataDir).getAbsolutePath()) + "'" + nl
				+ gtDirCode
				+ "output_dir = Path(r'" + TrainingCodeUtils.py(new File(outputDir).getAbsolutePath()) + "')" + nl
				+ "preview_dir = output_dir / 'previews'" + nl
				+ "preview_manifest_path = preview_dir / 'latest.json'" + nl
				+ "output_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "preview_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "stardist_log_path = output_dir / 'training.log'" + nl
				+ "config = json.loads(r'''" + TrainingCodeUtils.toJson(config) + "''')" + nl
				+ "_aliases = {'epochs': 'train_epochs', 'steps_per_epoch': 'train_steps_per_epoch', 'patch_size': 'train_patch_size', 'batch_size': 'train_batch_size', 'learning_rate': 'train_learning_rate', 'foreground_probability': 'train_foreground_only', 'background_reg': 'train_background_reg', 'loss_prob_weight': None, 'loss_dist_weight': None}" + nl
				+ "for _old, _new in list(_aliases.items()):" + nl
				+ "  if _new and _new in config and _old not in config:" + nl
				+ "    config[_old] = config[_new]" + nl
				+ "if 'loss_prob_weight' not in config and 'train_loss_weights' in config:" + nl
				+ "  config['loss_prob_weight'] = config['train_loss_weights'][0]" + nl
				+ "if 'loss_dist_weight' not in config and 'train_loss_weights' in config and len(config['train_loss_weights']) > 1:" + nl
				+ "  config['loss_dist_weight'] = config['train_loss_weights'][1]" + nl
				+ "state = {'total_steps': int(config.get('epochs', 0)) * int(config.get('steps_per_epoch', 0)), 'total_epochs': int(config.get('epochs', 0))}" + nl
				+ TrainingCodeUtils.taskUpdateFunction("_task_update")
				+ TrainingCodeUtils.scalarFunction("_scalar", false)
				+ TrainingCodeUtils.cleanDictFunction("_clean", "_scalar")
				+ "def _atomic_npy_save(path, array):" + nl
				+ "  tmp_path = str(path) + '.tmp'" + nl
				+ "  with open(tmp_path, 'wb') as f:" + nl
				+ "    np.save(f, array)" + nl
				+ "  os.replace(tmp_path, path)" + nl
				+ "def on_train_begin(plan):" + nl
				+ "  state['total_steps'] = int(plan.get('total_steps') or state['total_steps'])" + nl
				+ "  state['total_epochs'] = int(plan.get('epochs') or state['total_epochs'])" + nl
				+ "  info = {'type': 'progress', 'epoch': 0, 'step': 0, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': {}, 'metrics': _clean({'batch_size': plan.get('batch_size'), 'train_samples': plan.get('train_samples'), 'valid_samples': plan.get('valid_samples')})}" + nl
				+ "  _task_update(message='StarDist training started', current=0, maximum=state['total_steps'], info=info)" + nl
				+ "def on_step_end(step):" + nl
				+ "  global_step = int(step.get('global_step', 0))" + nl
				+ "  epoch = int(step.get('epoch', 0))" + nl
				+ "  losses = _clean({'train/total_loss': step.get('loss_total'), 'train/prob_loss': step.get('loss_prob'), 'train/dist_loss': step.get('loss_dist')})" + nl
				+ "  metrics = _clean({'learning_rate': step.get('learning_rate')})" + nl
				+ "  info = {'type': 'progress', 'epoch': epoch, 'step': global_step, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': losses, 'metrics': metrics}" + nl
				+ "  _task_update(message='StarDist training step %d/%d' % (global_step, state['total_steps']), current=global_step, maximum=state['total_steps'], info=info)" + nl
				+ "def on_validation_end(event):" + nl
				+ "  epoch = int(event.get('epoch', 0))" + nl
				+ "  step = min(state['total_steps'], epoch * int(config.get('steps_per_epoch', 1)))" + nl
				+ "  losses = _clean({'train/total_loss': event.get('train_total'), 'val/total_loss': event.get('valid_total')})" + nl
				+ "  metrics = _clean({'learning_rate': event.get('learning_rate')})" + nl
				+ "  info = {'type': 'progress', 'epoch': epoch, 'step': step, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': losses, 'metrics': metrics}" + nl
				+ "  _task_update(message='StarDist epoch %d/%d' % (epoch, state['total_epochs']), current=step, maximum=state['total_steps'], info=info)" + nl
				+ "  samples = []" + nl
				+ "  for i, preview in enumerate(event.get('previews', [])[:20]):" + nl
				+ "    image_path = preview_dir / ('preview_%03d_image.npy' % i)" + nl
				+ "    pred_path = preview_dir / ('preview_%03d_prediction.npy' % i)" + nl
				+ "    prob_path = preview_dir / ('preview_%03d_prob.npy' % i)" + nl
				+ "    sample = {'index': i}" + nl
				+ "    image = preview.get('image')" + nl
				+ "    prediction = preview.get('prediction')" + nl
				+ "    prob = preview.get('prob')" + nl
				+ "    if image is not None:" + nl
				+ "      _atomic_npy_save(image_path, image)" + nl
				+ "      sample['image_path'] = str(image_path)" + nl
				+ "    if prediction is not None:" + nl
				+ "      _atomic_npy_save(pred_path, np.asarray(prediction, dtype=np.int32))" + nl
				+ "      sample['prediction_path'] = str(pred_path)" + nl
				+ "    if prob is not None:" + nl
				+ "      _atomic_npy_save(prob_path, prob)" + nl
				+ "      sample['prob_path'] = str(prob_path)" + nl
				+ "    if 'image_path' in sample and ('prediction_path' in sample or 'prob_path' in sample):" + nl
				+ "      samples.append(sample)" + nl
				+ "  if samples:" + nl
				+ "    manifest = {'epoch': epoch, 'samples': samples}" + nl
				+ "    with open(preview_manifest_path, 'w', encoding='utf-8') as f:" + nl
				+ "      json.dump(manifest, f)" + nl
				+ "    _task_update(message='StarDist validation preview epoch %d' % epoch, current=epoch, maximum=state['total_epochs'], info={'type': 'preview', 'epoch': epoch, 'preview_path': str(preview_manifest_path)})" + nl
				+ "with open(stardist_log_path, 'a', encoding='utf-8') as stardist_log, contextlib.redirect_stdout(stardist_log), contextlib.redirect_stderr(stardist_log):" + nl
				+ "  result = train.train_stardist_2d_folder(data_dir=data_dir, " + gtDirArgument + "output_dir=str(output_dir), gpu=" + (gpu ? "True" : "False") + ", image_channels='" + TrainingCodeUtils.py(safeImageChannels) + "', label_color_mode='" + TrainingCodeUtils.py(safeLabelColorMode) + "', valid_fraction=" + validFraction + ", config=config, on_train_begin=on_train_begin, on_step_end=on_step_end, on_validation_end=on_validation_end)" + nl
				+ "task.output(result=str(result.get('output_dir', str(output_dir))))" + nl;
	}

	private static void handleTrainingEvent(TaskEvent event,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer) {
		if (event.message != null && logConsumer != null) {
			logConsumer.accept(event.message);
		}
		if (!event.responseType.equals(ResponseType.UPDATE) || event.info == null) {
			return;
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
	
	public static void main(String[] args) throws IOException, BuildException, LoadModelException, RunModelException {
		String path = "/home/carlos/git/deep-icy/models/stardist/hundred";
		try (StarDist model = StarDist.fromFile(path)) {
			model.inference(null);
		}
	}
}
