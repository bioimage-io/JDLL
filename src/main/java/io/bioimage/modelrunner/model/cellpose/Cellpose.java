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
package io.bioimage.modelrunner.model.cellpose;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
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
import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.model.stardist.Stardist2D;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

/**
 * Implementation of an API to run Cellpose models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *
 *@author Carlos Garcia
 */
public abstract class Cellpose implements Closeable {
	
	private final String modelDir;
	
	protected final String weightsPath;
	
	protected int[] channels;
	
	private int diameter = 30;
	
	private boolean loaded = false;
	
	protected SharedMemoryArray shma;
	
	private ModelDescriptor descriptor;
		
	private Service python;
	
	private String envPath = DEFAULT_ENV_DIR;
	
	private static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	private static String DEFAULT_ENV_DIR = INSTALLATION_DIR + File.separator + "envs" + DLModelPytorch.COMMON_PYTORCH_ENV_NAME;
	
	private static final List<String> CELLPOSE_DEPS = Arrays.asList(new String[] {"cellpose==3.1.1.1"});
	
	private static final List<String> PRETRAINED_CELLPOSE_MODELS = Arrays.asList(new String[] {"cyto", "cyto2", "cyto3", "nuclei"});
	
	private static final String CELLPOSE_URL = "https://www.cellpose.org/models/%s";
	
	private static final Map<String, String[]> MODEL_REQ;
	static {
		MODEL_REQ = new HashMap<String, String[]>();
		MODEL_REQ.put("cyto2", new String[] {"", ""});
		MODEL_REQ.put("cyto3", new String[] {"", ""});
		MODEL_REQ.put("cyto", new String[] {"", ""});
		MODEL_REQ.put("nuclei", new String[] {"", ""});
	}
	
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
			+ "model = %s(None, name='%s', basedir='%s')" + System.lineSeparator()
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
	
	protected Cellpose(String weightsPath) throws IOException, ModelSpecsException {
		this.weightsPath = weightsPath;
		this.modelDir = new File(weightsPath).getParentFile().getAbsolutePath();
		if (new File(modelDir, "config.json").isFile() == false && new File(modelDir, Constants.RDF_FNAME).isFile() == false)
			throw new IllegalArgumentException("No 'config.json' file found in the model directory");
		else if (new File(modelDir, "config.json").isFile() == false)
			createConfigFromBioimageio();
    	createPythonService();
	}
	
	protected Cellpose(ModelDescriptor descriptor) throws IOException, ModelSpecsException {
		this.descriptor = descriptor;
		modelDir = descriptor.getModelPath();
		this.weightsPath = descriptor.getModelPath() + File.separator + "";
		if (new File(modelDir, "config.json").isFile() == false)
			createConfigFromBioimageio();
    	createPythonService();
	}
	
	@SuppressWarnings("unchecked")
	private void createConfigFromBioimageio() throws IOException, ModelSpecsException {
		if (descriptor == null)
			descriptor = ModelDescriptorFactory.readFromLocalFile(modelDir + File.separator + Constants.RDF_FNAME);
    	Map<String, Object> stardistMap = (Map<String, Object>) descriptor.getConfig().getSpecMap().get("stardist");
    	Map<String, Object> stardistConfig = (Map<String, Object>) stardistMap.get("config");
    	JSONUtils.writeJSONFile(new File(modelDir, "config.json").getAbsolutePath(), stardistConfig);
	}
	
	public void setEnvPath(String envPath) throws IOException {
		if (!isInstalled(envPath))
			throw new IOException("Missing Cellpose requirements in the specified dir. These are the cellpose requirements: "
					+  CELLPOSE_DEPS);
		this.envPath = envPath;
		this.python.close();
		createPythonService();
	}
	
	public void setChannels(int[] channels) {
		this.channels = channels;
	}
	
	public void setDiameter(int diameter) {
		this.diameter = diameter;
	}
	
	public int getDiameter() {
		return this.diameter;
	}
	
	private void createPythonService() throws IOException {
		Environment env = new Environment() {
			@Override public String base() { return envPath; }
			};
		python = env.python();
		python.debug(System.err::println);
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
	
	public boolean isLoaded() {
		return loaded;
	}
	
	public void close() {
		if (!loaded)
			return;
		python.close();
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> predict(RandomAccessibleInterval<T> img) throws IOException, InterruptedException {
		checkInput(img);
		shma = SharedMemoryArray.createSHMAFromRAI(img, false, false);
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
		code += createEncodeImageScript() + System.lineSeparator();
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
		
		Map<String, RandomAccessibleInterval<T>> outs = new HashMap<String, RandomAccessibleInterval<T>>();
		outs.put("mask", reconstructMask());
		
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
	
	/**
	 * Initialize one of the "official" pretrained Stardist 2D models.
	 * By default, the model will be installed in the "models" folder inside the application
	 * @param pretrainedModel
	 * 	the name of the pretrained model. 
	 * @param forceDownload
	 * 	whether to force the download or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 */
	public static Cellpose fromPretained(String pretrainedModel, boolean forceDownload) throws IOException, InterruptedException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), forceDownload);
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param descriptor
	 * 	the bioimage.io model descriptor
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 */
	public static Cellpose fromBioimageioModel(ModelDescriptor descriptor) throws IOException {
		if (descriptor.getTags().stream().filter(tt -> tt.toLowerCase().equals("cellpose")).findFirst().orElse(null) == null
				&& !descriptor.getName().toLowerCase().contains("cellpose"))
			throw new RuntimeException("This model does not seem to be a cellpose model from the Bioimage.io");
		return new Cellpose(descriptor);
	}
	
	/**
	 * Initialize one of the "official" pretrained cellpose ("cyto2", "cyto3"...) models or
	 * those available in the bioimage.io
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param installDir
	 * 	the directory where the model wants to be installed
	 * @param forceInstall
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 */
	public static Cellpose fromPretained(String pretrainedModel, String installDir, boolean forceInstall) throws IOException, 
																					InterruptedException {
		if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel) && !forceInstall) {
			ModelDescriptor md = ModelDescriptorFactory.getModelsAtLocalRepo().stream()
					.filter(mm ->mm.getName().equals("StarDist H&E Nuclei Segmentation")).findFirst().orElse(null);
			if (md != null) return new Stardist2D(md);
			String path = BioimageioRepo.connect().downloadByName("StarDist H&E Nuclei Segmentation", installDir);
			return Cellpose.fromPretained(pretrainedModel);
		} else if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel)) {
			String path = BioimageioRepo.connect().downloadByName("StarDist H&E Nuclei Segmentation", installDir);
			return Cellpose.fromPretained(pretrainedModel);
		}
		
		BioimageioRepo br = BioimageioRepo.connect();

		ModelDescriptor descriptor = br.selectByName(pretrainedModel);
		if (descriptor == null)
			descriptor = br.selectByID(pretrainedModel);
		if (descriptor == null)
			throw new IllegalArgumentException("The model does not correspond to on of the available pretrained cellpose models."
					+ " To find a list of available cellpose models, please run Cellpose.getPretrainedList()");

		String path = BioimageioRepo.downloadModel(descriptor, installDir);
		descriptor.addModelPath(Paths.get(path));
		return Cellpose.fromBioimageioModel(descriptor);
	}
	
	public static List<String> getPretrainedList() {
		List<String> list = new ArrayList<String>();
		try {
			BioimageioRepo br = BioimageioRepo.connect();
			Map<String, ModelDescriptor> models = br.listAllModels(false);
			list = models.entrySet().stream()
					.filter(ee -> ee.getValue().getModelFamily().equals(ModelDescriptor.CELLPOSE))
					.map(ee -> ee.getValue().getName()).collect(Collectors.toList());
		} catch (InterruptedException e) {
		}
		list.addAll(PRETRAINED_CELLPOSE_MODELS);
		return list;
	}
	
	public static String donwloadPretrained(String modelName, String downloadDir) {
		return donwloadPretrained(modelName, downloadDir, null);
	}
	
	public static String donwloadPretrained(String modelName, String downloadDir, Consumer<Double> progressConsumer) {
		
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled() {
		// TODO
		return isInstalled(null);
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled(String envPath) {
		// TODO
		return false;
	}
	
	/**
	 * Check whether the requirements needed to run cellpose are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for cellpose post processing is fine too.
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
	 * Check whether the requirements needed to run cellpose are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for cellpose post processing is fine too.
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
		if (!DLModelPytorch.isInstalled(DEFAULT_ENV_DIR)) {
			String pytorchDir = DLModelPytorch.getInstallationDir();
			DLModelPytorch.setInstallationDir(INSTALLATION_DIR);
			DLModelPytorch.installRequirements(consumer);
			DLModelPytorch.setInstallationDir(pytorchDir);
		}
			
		Mamba mamba = new Mamba(INSTALLATION_DIR);
		if (consumer != null) {
			mamba.setConsoleOutputConsumer(consumer);
			mamba.setErrorOutputConsumer(consumer);
		}
		boolean cellposePythonInstalled = false;
		try {
			cellposePythonInstalled = mamba.checkAllDependenciesInEnv(DLModelPytorch.COMMON_PYTORCH_ENV_NAME, CELLPOSE_DEPS);
		} catch (MambaInstallException e) {
			mamba.installMicromamba();
		}
		if (!cellposePythonInstalled) {
			mamba.pipInstallIn(CLOSE_SHM_CODE, CELLPOSE_DEPS.toArray(new String[CELLPOSE_DEPS.size()]));
		};
	}
	
	/**
	 * Set the directory where the cellpose Python environment will be installed
	 * @param installationDir
	 * 	directory where the cellpose Python environment will be created
	 */
	public static void setInstallationDir(String installationDir) {
		INSTALLATION_DIR = installationDir;
		DEFAULT_ENV_DIR = INSTALLATION_DIR + File.separator + "envs" + DLModelPytorch.COMMON_PYTORCH_ENV_NAME;
	}
	
	/**
	 * 
	 * @return the directory where the cellpose Python environment will be created
	 */
	public static String getInstallationDir() {
		return INSTALLATION_DIR;
	}
}
