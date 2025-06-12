/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Environment;
import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.transformations.ScaleRangeTransformation;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

/**
 * Implementation of an API to run Stardist 2D models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *
 *@author Carlos Garcia
 */
public abstract class StardistAbstract extends BaseModel {
	
	private final String modelDir;
	
	protected final String name;
	
	protected final String basedir;
	
	protected Double threshold = null;
	
	protected final int nChannels;
	
	protected Map<String, Object> config;
	
	protected SharedMemoryArray shma;
	
	private ModelDescriptor descriptor;
		
	private Service python;
	
	/**
	 * Value used to scale the image wihtin the [0, 1] range.
	 * Using minimum percentile 0 is equivalent to use the minimum of the image as the max
	 * Every pixels is transformed as follows: new_pixel = (pixel - min) / (max - min)
	 */
	public double scaleRangeMaxPercentile = 99.8;
	
	/**
	 * Value used to scale the image wihtin the [0, 1] range.
	 * Using maximum percentile 100 is equivalent to use the maximum of the image as the max
	 * Every pixels is transformed as follows: new_pixel = (pixel - min) / (max - min)
	 */
	public double scaleRangeMinPercentile = 1;
	
	/**
	 * Channels along which the scalin is performed.
	 * Imagine a xyc image, if the axes specified are xyc, the image will be scaled all together.
	 * However if the axes specified are xy, each channel will be scaled differently.
	 * By default all the image is scaled together
	 * 
	 */
	public String scaleRangeAxes = null;
	
	private static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	private static final List<String> STARDIST_DEPS = Arrays.asList(new String[] {"python=3.10", "stardist", "numpy", "appose"});
	
	private static final List<String> STARDIST_DEPS_PIP;
	static {
		if (PlatformDetection.isMacOS() 
				&& (PlatformDetection.getArch().equals(PlatformDetection.ARCH_ARM64) || PlatformDetection.isUsingRosseta()))
			STARDIST_DEPS_PIP = Arrays.asList(new String[] {"tensorflow-macos<2.11"});
		else
			STARDIST_DEPS_PIP = Arrays.asList(new String[] {"tensorflow<2.11"});
	}
	
	private static final List<String> STARDIST_CHANNELS = Arrays.asList(new String[] {"conda-forge", "default"});

	
	private static final String OUTPUT_MASK_KEY = "mask";
	
	private static final String SHM_NAME_KEY = "_shm_name";
	
	private static final String DTYPE_KEY = "_dtype";
	
	private static final String SHAPE_KEY = "_shape";
	
	private static final String KEYS_KEY = "keys";
	
	protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
			+ "if '%s' not in globals().keys():" + System.lineSeparator()
			+ "  from stardist.models import %s" + System.lineSeparator()
			+ "  globals()['%s'] = %s" + System.lineSeparator()
			+ "if 'np' not in globals().keys():" + System.lineSeparator()
			+ "  import numpy as np" + System.lineSeparator()
			+ "  globals()['np'] = np" + System.lineSeparator()
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  globals()['os'] = os" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
			+ "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"-1\"" + System.lineSeparator()
			+ "model = %s(None, name='%s', basedir=r\"%s\")" + System.lineSeparator()
			+ "globals()['model'] = model" + System.lineSeparator();
	
	private static final String RUN_MODEL_CODE = ""
			+ "output = model.predict_instances(im, return_predict=False)" + System.lineSeparator()
			+ "if type(output) == np.ndarray:" + System.lineSeparator()
			+ "  im[:] = output" + System.lineSeparator()
			+ "  im[:] = output" + System.lineSeparator()
			+ "  if os.name == 'nt':" + System.lineSeparator()
			+ "    im_shm.close()" + System.lineSeparator()
			+ "    im_shm.unlink()" + System.lineSeparator()
			+ "if type(output) != list and type(output) != tuple:" + System.lineSeparator()
			+ "  raise TypeError('StarDist output should be a list of a np.ndarray')" + System.lineSeparator()
			+ "if type(output[0]) != np.ndarray:" + System.lineSeparator()
			+ "  raise TypeError('If the StarDist output is a list, the first entry should be a np.ndarray')" + System.lineSeparator()
			+ "if len(im.shape) == 3 and len(output[0].shape) == 2:" + System.lineSeparator()
			+ "  im[:, :, 0] = output[0]" + System.lineSeparator()
			+ "elif len(im.shape) == 4 and len(output[0].shape) == 3:" + System.lineSeparator()
			+ "  im[:, :, :, 0] = output[0]" + System.lineSeparator()
			+ "else:" + System.lineSeparator()
			+ "  im[:] = output[0]" + System.lineSeparator()
			+ "if len(output) > 1 and type(output[1]) != dict:" + System.lineSeparator()
			+ "  raise TypeError('If the StarDist output is a list, the second entry needs to be a dict.')" + System.lineSeparator()
			+ "task.outputs['" + KEYS_KEY + "'] = list(output[1].keys())" + System.lineSeparator()
			+ "shm_list = []" + System.lineSeparator()
			+ "np_list = []" + System.lineSeparator()
			+ "for kk, vv in output[1].items():" + System.lineSeparator()
			+ "  if type(vv) != np.ndarray:" + System.lineSeparator()
			+ "    task.update('Output ' + kk + ' is not a np.ndarray. Only np.ndarrays supported.')" + System.lineSeparator()
			+ "    continue" + System.lineSeparator()
			+ "  if output[1][kk].nbytes == 0:" + System.lineSeparator()
			+ "    task.outputs[kk] = None" + System.lineSeparator()
			+ "  else:" + System.lineSeparator()
			+ "    task.outputs[kk + '" + SHAPE_KEY + "'] = output[1][kk].shape" + System.lineSeparator()
			+ "    task.outputs[kk + '"+ DTYPE_KEY + "'] = str(output[1][kk].dtype)" + System.lineSeparator()
			+ "    shm = shared_memory.SharedMemory(create=True, size=output[1][kk].nbytes)" + System.lineSeparator()
			+ "    task.outputs[kk + '"+ SHM_NAME_KEY + "'] = shm.name" + System.lineSeparator()
			+ "    shm_list.append(shm)" + System.lineSeparator()
			+ "    aa = np.ndarray(output[1][kk].shape, dtype=output[1][kk].dtype, buffer=shm.buf)" + System.lineSeparator()
			+ "    aa[:] = output[1][kk]" + System.lineSeparator()
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
	
	protected abstract String createImportsCode();
	
	protected abstract <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image);
	
	protected abstract <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() throws IOException;

	/**
	 * 
	 * @return whether the model is stardist2d or not
	 */
	public abstract boolean is2D();
	
	/**
	 * 
	 * @return whether the model is stardist3d or not
	 */
	public abstract boolean is3D();
	
	protected StardistAbstract(String modelName, String baseDir, Map<String, Object> config) throws IOException {
		this.name = modelName;
		this.basedir = baseDir;
		modelDir = new File(baseDir, modelName).getAbsolutePath();
		checkFilesPresent(modelDir);
		this.nChannels = ((Number) config.get("n_channel_in")).intValue();
    	createPythonService();
	}
	
	protected StardistAbstract(String modelName, String baseDir) throws IOException {
		this.name = modelName;
		this.basedir = baseDir;
		modelDir = new File(baseDir, modelName).getAbsolutePath();
		checkFilesPresent(modelDir);
		config = JSONUtils.load(new File(modelDir, "config.json").getAbsolutePath());
		this.nChannels = ((Number) config.get("n_channel_in")).intValue();
    	createPythonService();
	}
	
	public static void checkFilesPresent(String modelDir) throws IOException {
		if (new File(modelDir, "config.json").isFile() == false && new File(modelDir, Constants.RDF_FNAME).isFile() == false)
			throw new IllegalArgumentException("No 'config.json' file found in the model directory");
		else if (new File(modelDir, "config.json").isFile() == false)
			createConfigFromBioimageio(null, modelDir);
		if (new File(modelDir, "thresholds.json").isFile() == false && new File(modelDir, Constants.RDF_FNAME).isFile() == false)
			throw new IllegalArgumentException("No 'thresholds.json' file found in the model directory");
		else if (new File(modelDir, "thresholds.json").isFile() == false)
			createThresholdsFromBioimageio(null, modelDir);
	}
	
	protected StardistAbstract(ModelDescriptor descriptor) throws IOException {
		this.descriptor = descriptor;
		this.name = new File(descriptor.getModelPath()).getName();
		this.basedir = new File(descriptor.getModelPath()).getParentFile().getAbsolutePath();
		modelDir = descriptor.getModelPath();
		if (new File(modelDir, "config.json").isFile() == false)
			createConfigFromBioimageio(descriptor, modelDir);
		if (new File(modelDir, "thresholds.json").isFile() == false)
			createThresholdsFromBioimageio(descriptor, modelDir);
		config = JSONUtils.load(new File(modelDir, "config.json").getAbsolutePath());
		this.nChannels = ((Number) config.get("n_channel_in")).intValue();
    	createPythonService();
	}
	
	@SuppressWarnings("unchecked")
	private static  void createConfigFromBioimageio(ModelDescriptor descriptor, String modelDir) throws IOException {
		if (descriptor == null)
			descriptor = ModelDescriptorFactory.readFromLocalFile(modelDir + File.separator + Constants.RDF_FNAME);
    	Map<String, Object> stardistMap = (Map<String, Object>) descriptor.getConfig().getSpecMap().get("stardist");
    	Map<String, Object> stardistConfig = (Map<String, Object>) stardistMap.get("config");
    	JSONUtils.writeJSONFile(new File(modelDir, "config.json").getAbsolutePath(), stardistConfig);
	}	
	
	private static void createThresholdsFromBioimageio(ModelDescriptor descriptor, String modelDir) throws IOException {
		if (descriptor == null)
			descriptor = ModelDescriptorFactory.readFromLocalFile(modelDir + File.separator + Constants.RDF_FNAME);
    	Map<String, Object> stardistMap = (Map<String, Object>) descriptor.getConfig().getSpecMap().get("stardist");
    	Map<String, Object> stardistThres = (Map<String, Object>) stardistMap.get("thresholds");
    	JSONUtils.writeJSONFile(new File(modelDir, "thresholds.json").getAbsolutePath(), stardistThres);
	}	
	
	private void createPythonService() throws IOException {
		Environment env = new Environment() {
			@Override public String base() { return new Mamba(INSTALLATION_DIR).getEnvsDir() + File.separator + "stardist"; }
			};
		python = env.python();
		python.debug(System.err::println);
	}
	
	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}
	
	protected String createEncodeImageScript() {
		String code = "";
		// This line wants to recreate the original numpy array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = np.ndarray(size, dtype="float64", buffer=input0_appose_shm.buf).reshape([64, 64])
		code += "im_shm = shared_memory.SharedMemory(name='"
							+ shma.getNameForPython() + "', size=" + shma.getSize() 
							+ ")" + System.lineSeparator();
		long nElems = 1;
		for (long elem : shma.getOriginalShape()) nElems *= elem;
		code += "im = np.ndarray(" + nElems  + ", dtype='" + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
			  + "', buffer=im_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i ++)
			code += shma.getOriginalShape()[i] + ", ";
		code += "])" + System.lineSeparator();
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void preprocess(List<Tensor<T>> inputs) {
		if (descriptor != null) {
			Processing processing = Processing.init(descriptor);
			List<Tensor<T>> inputsProcessed = processing.preprocess(inputs, false);
			inputs.set(0, inputsProcessed.get(0));
		} else {
			ScaleRangeTransformation transform = new ScaleRangeTransformation();
			transform.setMaxPercentile(scaleRangeMaxPercentile);
			transform.setMinPercentile(scaleRangeMinPercentile);
			transform.setAxes(scaleRangeAxes);
			inputs.set(0, Cast.unchecked(transform.apply(inputs.get(0))));
		}
	}
	
	public int getNChannels() {
		return nChannels;
	}
	
	@Override
	public void close() {
		if (!loaded)
			return;
		python.close();
		loaded = false;
		closed = true;
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run( List< Tensor < T > > inTensors, List< Tensor < R > > outTensors ) throws RunModelException {
		if (inTensors.size() > 1)
			throw new RunModelException("Stardist needs just one input image");
		preprocess(inTensors);
		try {
			Map<String, RandomAccessibleInterval<R>> outputs = run(inTensors.get(0).getData());
			for (Tensor<R> tensor : outTensors) {
				Entry<String, RandomAccessibleInterval<R>> entry = outputs.entrySet().stream()
						.filter(ee -> tensor.getName().equals(ee.getKey())
								&& Arrays.equals(tensor.getData().dimensionsAsLongArray(), ee.getValue().dimensionsAsLongArray()))
						.findFirst().orElse(null);
				if (entry == null 
						&& Arrays.equals(tensor.getData().dimensionsAsLongArray(), outputs.get(OUTPUT_MASK_KEY).dimensionsAsLongArray()))
					tensor.setData(outputs.get(OUTPUT_MASK_KEY));
				else if (entry != null)
					tensor.setData(entry.getValue());
			}
		} catch (IOException | InterruptedException e) {
			throw new RunModelException(Types.stackTrace(e));
		}
	}
	
	@Override
	public void loadModel() throws LoadModelException {
		if (closed)
			throw new RuntimeException("Cannot load model after it has been closed");
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
		Task task;
		try {
			task = python.task(code);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException("Task canceled");
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED)
				throw new RuntimeException(task.error);
			loaded = true;
		} catch (IOException | InterruptedException e) {
			throw new LoadModelException(Types.stackTrace(e));
		}
	}
	
	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) 
			throws RunModelException {
		if (inputTensors.size() > 1)
			throw new RunModelException("Stardist needs just one input image");
		preprocess(inputTensors);
		try {
			Map<String, RandomAccessibleInterval<T>> outputs = run(inputTensors.get(0).getData());
			List<Tensor<T>> outTensors = new ArrayList<Tensor<T>>();
			for (Entry<String, RandomAccessibleInterval<T>> entry : outputs.entrySet()) {
				if (entry.getValue() == null)
					continue;
				String axesOrder = "xy";
				if (entry.getValue().dimensionsAsLongArray().length > 2 && this.is2D())
					axesOrder += "c";
				else if (entry.getValue().dimensionsAsLongArray().length == 3 && this.is3D())
					axesOrder += "z";
				else if (entry.getValue().dimensionsAsLongArray().length > 3 && this.is3D())
					axesOrder += "zc";
				else if (entry.getValue().dimensionsAsLongArray().length == 1)
					axesOrder = "i";
				Tensor<T> tt = Tensor.build(entry.getKey(), axesOrder, entry.getValue());
				// TODO
				if (tt.getName() != "mask")
					continue;
				outTensors.add(tt);
			}
			return outTensors;
		} catch (IOException | InterruptedException e) {
			throw new RunModelException(Types.stackTrace(e));
		}
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> run(RandomAccessibleInterval<T> img) throws IOException, InterruptedException {
		checkInput(img);
		shma = SharedMemoryArray.createSHMAFromRAI(img, false, false);
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
		code += createEncodeImageScript() + System.lineSeparator();
		if (this.threshold != null) {
			code += String.format("model.thresholds = dict (prob=%s, nms=model.thresholds.nms)", threshold) + System.lineSeparator();
		}
		code += RUN_MODEL_CODE + System.lineSeparator();
		
		Task task = python.task(code);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException("Task canceled");
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException(task.error);
		else if (task.status == TaskStatus.CRASHED)
			throw new RuntimeException(task.error);
		loaded = true;
		
		
		return reconstructOutputs(task);
	}
	
	private <T extends RealType<T> & NativeType<T>> 
	Map<String, RandomAccessibleInterval<T>> reconstructOutputs(Task task) 
			throws IOException, InterruptedException {
		
		Map<String, RandomAccessibleInterval<T>> outs = new LinkedHashMap<String, RandomAccessibleInterval<T>>();
		outs.put(OUTPUT_MASK_KEY, reconstructMask());
		
		if (task.outputs.get(KEYS_KEY) != null) {
			for (String kk : (List<String>) task.outputs.get(KEYS_KEY)) {
				outs.put(kk, reconstruct(task, kk));
			}
		}
		
		if (PlatformDetection.isWindows()) {
			Task closeSHMTask = python.task(CLOSE_SHM_CODE);
			closeSHMTask.waitFor();
		}
		return outs;
	}
	
	private <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> reconstruct(Task task, String key) throws IOException {

		String shm_name = (String) task.outputs.get(key + SHM_NAME_KEY);
		String coords_dtype = (String) task.outputs.get(key + DTYPE_KEY);
		List<Number> coords_shape = (List<Number>) task.outputs.get(key + SHAPE_KEY);
		if (coords_shape == null)
			return null;
		
		long[] coordsSh = new long[coords_shape.size()];
		for (int i = 0; i < coordsSh.length; i ++)
			coordsSh[i] = coords_shape.get(i).longValue();
		SharedMemoryArray shmCoords = SharedMemoryArray.readOrCreate(shm_name, coordsSh, 
				Cast.unchecked(CommonUtils.getImgLib2DataType(coords_dtype)), false, false);
		
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> coordsRAI = shmCoords.getSharedRAI();
		RandomAccessibleInterval<T> coordsCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(coordsRAI), 
				Util.getTypeFromInterval(Cast.unchecked(coordsRAI)));
		
		shmCoords.close();
		
		return coordsCopy;
	}
	
	public static StardistAbstract init(String modelDir) throws IOException {
		File modelDirFile = new File(modelDir);
		String modelName = modelDirFile.getName();
		String baseDir = modelDirFile.getParentFile().getAbsolutePath();
		checkFilesPresent(modelDir);
		Map<String, Object> configMap = JSONUtils.load(new File(modelDir, "config.json").getAbsolutePath());
		String axes = ((String) configMap.get("axes")).toUpperCase();
		if (axes.contains("Z"))
			return new Stardist3D(modelName, baseDir, configMap);
		else
			return new Stardist2D(modelName, baseDir, configMap);
	}
	
	public static StardistAbstract init(String modelName, String baseDir) throws IOException {
		String modelDir = new File(baseDir, modelName).getAbsolutePath();
		checkFilesPresent(modelDir);
		Map<String, Object> configMap = JSONUtils.load(new File(modelDir, "config.json").getAbsolutePath());
		String axes = ((String) configMap.get("axes")).toUpperCase();
		if (axes.contains("Z"))
			return new Stardist3D(modelName, baseDir, configMap);
		else
			return new Stardist2D(modelName, baseDir, configMap);
	}
	
	public static StardistAbstract fromBioimageioModel(ModelDescriptor descriptor) throws IOException {
		if (!descriptor.getConfig().getSpecMap().keySet().contains("stardist"))
			throw new IllegalArgumentException("This Bioimage.io model does not correspond to a StarDist model.");
		if (!descriptor.getModelFamily().equals(ModelDescriptor.STARDIST))
			throw new RuntimeException("Please first install StarDist with 'StardistAbstract.installRequirements()'");
		if (descriptor.getInputTensors().get(0).getAxesOrder().contains("z"))
			return Stardist3D.fromBioimageioModel(descriptor);
		else 
			return Stardist2D.fromBioimageioModel(descriptor);
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled() {
		Mamba mamba = new Mamba(INSTALLATION_DIR);
		try {
			return mamba.checkAllDependenciesInEnv("stardist", STARDIST_DEPS);
		} catch (MambaInstallException e) {
			return false;
		}
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @param envPath
	 * 	path to the directory where the StarDist env is installed
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled(String envPath) {
		Mamba mamba = new Mamba(INSTALLATION_DIR);
		try {
			return mamba.checkAllDependenciesInEnv(envPath, STARDIST_DEPS);
		} catch (MambaInstallException e) {
			return false;
		}
	}
	
	/**
	 * Check whether the requirements needed to run Stardist are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for Stardist post processing is fine too.
	 * 
	 * If anything is not installed, this method also installs it
	 * 
	 * @throws IOException if there is any error downloading the DL engine or installing the micromamba environment
	 * @throws InterruptedException if the installation is stopped
	 * @throws RuntimeException if there is any unexpected error in the micromamba environment installation
	 * @throws MambaInstallException if there is any error downloading or installing micromamba
	 * @throws ArchiveException if there is any error decompressing the micromamba installer
	 * @throws URISyntaxException if the URL to the micromamba installation is not correct
	 */
	public static void installRequirements() throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ArchiveException, URISyntaxException {
		installRequirements(null);
	}
	
	/**
	 * Check whether the requirements needed to run Stardist are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for Stardist post processing is fine too.
	 * 
	 * If anything is not installed, this method also installs it
	 * 
	 * @param consumer
	 * 	String consumer that reads the installation log
	 * 
	 * @throws IOException if there is any error downloading the DL engine or installing the micromamba environment
	 * @throws InterruptedException if the installation is stopped
	 * @throws RuntimeException if there is any unexpected error in the micromamba environment installation
	 * @throws MambaInstallException if there is any error downloading or installing micromamba
	 * @throws ArchiveException if there is any error decompressing the micromamba installer
	 * @throws URISyntaxException if the URL to the micromamba installation is not correct
	 */
	public static void installRequirements(Consumer<String> consumer) throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ArchiveException, URISyntaxException {
		
		Mamba mamba = new Mamba(INSTALLATION_DIR);
		if (consumer != null) {
			mamba.setConsoleOutputConsumer(consumer);
			mamba.setErrorOutputConsumer(consumer);
		}
		boolean stardistPythonInstalled = false;
		try {
			List<String> deps = new ArrayList<String>();
			for (String dd : STARDIST_DEPS)
				deps.add(dd.equals("tensorflow-macos<2.11") ? dd.replace("-macos", "") : dd);
			stardistPythonInstalled = mamba.checkAllDependenciesInEnv("stardist", deps);
		} catch (MambaInstallException e) {
			mamba.installMicromamba();
		}
		if (!stardistPythonInstalled) {
			mamba.create("stardist", true, STARDIST_CHANNELS, STARDIST_DEPS.stream()
					.map(dd -> dd.contains("<") | dd.contains(">") ? "\"" + dd + "\"": dd)
					.collect(Collectors.toList()));
			mamba.pipInstallIn("stardist", STARDIST_DEPS_PIP.stream()
					.map(dd -> (PlatformDetection.isWindows() && (dd.contains("<") | dd.contains(">"))) ? "\"" + dd + "\"": dd)
					.collect(Collectors.toList()).toArray(new String[STARDIST_DEPS_PIP.size()]));
		};
	}
	
	/**
	 * Set the directory where the StarDist Python environment will be installed
	 * @param installationDir
	 * 	directory where the StarDist Python environment will be created
	 */
	public static void setInstallationDir(String installationDir) {
		INSTALLATION_DIR = installationDir;
	}
	
	/**
	 * 
	 * @return the directory where the StarDist Python environment will be created
	 */
	public static String getInstallationDir() {
		return INSTALLATION_DIR;
	}
}
