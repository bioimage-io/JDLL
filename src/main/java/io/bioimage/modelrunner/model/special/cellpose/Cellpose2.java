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
package io.bioimage.modelrunner.model.special.cellpose;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.download.MultiFileDownloader;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.model.special.SpecialModelBase;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Implementation of an API to run Cellpose models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *
 *@author Carlos Garcia
 */
public class Cellpose2 extends SpecialModelBase {
	
	private final String modelDir;
	
	protected final String weightsPath;
	
	protected final String sizeWeigthsPath;
	
	protected final String modelType;
	
	private String axes = "";
	
	protected int[] channels;
	
	private Integer diameter;
	
	private boolean is3D = false;
	
	private boolean loaded = false;
	
	protected SharedMemoryArray shma;
	
	protected SharedMemoryArray shmaFl;
	
	protected SharedMemoryArray shmaDn;
	
	protected SharedMemoryArray shmaSt;
	
	private ModelDescriptor descriptor;
		
	private Service python;
	
	private static int DEFAULT_DIAMETER = 30;
	
	private static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	private static String DEFAULT_ENV_DIR = INSTALLATION_DIR + File.separator + "envs" + DLModelPytorch.COMMON_PYTORCH_ENV_NAME;
	
	private static final List<String> CELLPOSE_DEPS = Arrays.asList(new String[] {"cellpose==3.1.1.1"});
	
	private static final List<String> PRETRAINED_CELLPOSE_MODELS = Arrays.asList(new String[] {"cyto", "cyto2", "cyto3", "nuclei"});
	
	private static final String CELLPOSE_URL = "https://www.cellpose.org/models/%s";
	
	private static final Map<String, String[]> MODEL_REQ;
	static {
		MODEL_REQ = new HashMap<String, String[]>();
		MODEL_REQ.put("cyto2", new String[] {"cyto2torch_0", "size_cyto2torch_0.npy"});
		MODEL_REQ.put("cyto3", new String[] {"cyto3", "size_cyto3.npy"});
		MODEL_REQ.put("cyto", new String[] {"cytotorch_0", "size_cytotorch_0.npy"});
		MODEL_REQ.put("nuclei", new String[] {"nucleitorch_0", "size_nucleitorch_0.npy"});
	}
	
	private static final Map<String, String> ALIAS;
	static {
		ALIAS = new HashMap<String, String>();
		ALIAS.put("cyto2", "cyto2torch_0");
		ALIAS.put("cyto3", "cyto3");
		ALIAS.put("cyto", "cytotorch_0");
		ALIAS.put("nuclei", "nucleitorch_0");
	}
	
	private static final Map<String, Long> MODEL_SIZE;
	static {
		MODEL_SIZE = new HashMap<String, Long>();
		MODEL_SIZE.put("cyto2torch_0", 26_563_614L);
		MODEL_SIZE.put("cyto3", 26_566_255L);
		MODEL_SIZE.put("cytotorch_0", 26_563_614L);
		MODEL_SIZE.put("nucleitorch_0", 26_563_614L);
	}
	
	private static final String SHM_NAME_KEY = "_shm_name";
	
	private static final String DTYPE_KEY = "_dtype";
	
	private static final String SHAPE_KEY = "_shape";
	
	private static final String KEYS_KEY = "keys";

	protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
			+ "if 'denoise' not in globals().keys():" + System.lineSeparator()
			+ "  from cellpose import denoise" + System.lineSeparator()
			+ "  globals()['denoise'] = denoise" + System.lineSeparator()
			+ "if 'np' not in globals().keys():" + System.lineSeparator()
			+ "  import numpy as np" + System.lineSeparator()
			+ "  globals()['np'] = np" + System.lineSeparator()
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  globals()['os'] = os" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
			+ "model = denoise.CellposeDenoiseModel(gpu=True, model_type='%s')" + System.lineSeparator()
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
	
	protected Cellpose2(String weightsPath) throws IOException {
		this.weightsPath = weightsPath;
		this.modelDir = new File(weightsPath).getParentFile().getAbsolutePath();
		this.axes = "yxc";
		this.modelType = ALIAS.entrySet().stream()
				.filter(ee -> ee.getValue().equals(new File(this.weightsPath).getName()))
				.map(ee -> ee.getValue()).findFirst().get();
		this.sizeWeigthsPath = modelDir + File.separator + MODEL_REQ.get(modelType)[1];
		this.envPath = DEFAULT_ENV_DIR;
    	createPythonService();
	}
	
	protected Cellpose2(ModelDescriptor descriptor) throws IOException {
		this.descriptor = descriptor;
		this.sizeWeigthsPath = null;
		this.modelType = "bioimage.io";
		modelDir = descriptor.getModelPath();
		this.weightsPath = descriptor.getModelPath() + File.separator + "";
		if (new File(modelDir, "config.json").isFile() == false)
			createConfigFromBioimageio();
		envPath = DEFAULT_ENV_DIR;
    	createPythonService();
	}
	
	@SuppressWarnings("unchecked")
	private void createConfigFromBioimageio() throws IOException {
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
	
	protected String createEncodeImageScript() {
		String code = "";
		code += codeToConvertShmaToPython(shma, "im") + System.lineSeparator();
		code += codeToConvertShmaToPython(shmaFl, "fl") + System.lineSeparator();
		code += codeToConvertShmaToPython(shmaSt, "st") + System.lineSeparator();
		code += codeToConvertShmaToPython(shmaDn, "dn") + System.lineSeparator();
		return code;
	}
	
	public boolean isLoaded() {
		return loaded;
	}
	
	public void close() {
		if (!loaded)
			return;
		python.close();
		loaded = false;
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
	 * @throws ExecutionException 
	 */
	public static Cellpose2 fromPretained(String pretrainedModel, boolean forceDownload) throws IOException, InterruptedException, ExecutionException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), forceDownload);
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param descriptor
	 * 	the bioimage.io model descriptor
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 */
	public static Cellpose2 fromBioimageioModel(ModelDescriptor descriptor) throws IOException {
		if (descriptor.getTags().stream().filter(tt -> tt.toLowerCase().equals("cellpose")).findFirst().orElse(null) == null
				&& !descriptor.getName().toLowerCase().contains("cellpose"))
			throw new RuntimeException("This model does not seem to be a cellpose model from the Bioimage.io");
		return new Cellpose2(descriptor);
	}
	
	/**
	 * Initialize one of the "official" pretrained cellpose ("cyto2", "cyto3"...) models or
	 * those available in the bioimage.io
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param modelsDir
	 * 	the directory where the model wants to be installed
	 * @param forceInstall
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ExecutionException 
	 */
	public static Cellpose2 fromPretained(String pretrainedModel, String modelsDir, boolean forceInstall) throws IOException, 
																					InterruptedException, ExecutionException {
		if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel) && !forceInstall) {
			String weightsPath = fileIsCellpose(pretrainedModel, modelsDir);
			if (weightsPath != null) return new Cellpose2(weightsPath);
			String path = donwloadPretrainedOfficial(pretrainedModel, modelsDir, null);
			return new Cellpose2(path);
		} else if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel)) {
			String path = donwloadPretrainedOfficial(pretrainedModel, modelsDir, null);
			return new Cellpose2(path);
		}
		if (!forceInstall) {
			List<ModelDescriptor> localModels = ModelDescriptorFactory.getModelsAtLocalRepo();
			ModelDescriptor model = localModels.stream()
					.filter(md -> md.getModelID().equals(pretrainedModel) 
							|| md.getName().toLowerCase().equals(pretrainedModel.toLowerCase()))
					.findFirst().orElse(null);
			if (model != null)
				return Cellpose2.fromBioimageioModel(model);
		}
		
		BioimageioRepo br = BioimageioRepo.connect();
		ModelDescriptor descriptor = br.selectByName(pretrainedModel);
		if (descriptor == null)
			descriptor = br.selectByID(pretrainedModel);
		if (descriptor == null)
			throw new IllegalArgumentException("The model does not correspond to on of the available pretrained cellpose models."
					+ " To find a list of available cellpose models, please run Cellpose.getPretrainedList()");
		String path = BioimageioRepo.downloadModel(descriptor, modelsDir);
		descriptor.addModelPath(Paths.get(path));
		return Cellpose2.fromBioimageioModel(descriptor);
	}
	
	private static String fileIsCellpose(String pretrainedModel, String modelsDir) {
		File pretrainedFile = new File(pretrainedModel);
		 if (pretrainedFile.isFile() && isCellposeFile(pretrainedFile))
			 return pretrainedFile.getAbsolutePath();
		 if (ALIAS.keySet().contains(pretrainedModel) || MODEL_SIZE.containsKey(pretrainedModel)) {
			 String path = lookForModelInDir(pretrainedModel, modelsDir);
			 if (path != null)
				 return path;
		 }
		 return null;
	}
	
	private static boolean isCellposeFile(File pretrainedFile) {
		return MODEL_SIZE.keySet().contains(pretrainedFile.getName()) && MODEL_SIZE.get(pretrainedFile.getName()) == pretrainedFile.length();
	}
	
	private static String lookForModelInDir(String modelName, String modelsDir) {
		File dir = new File(modelsDir);
		if (!dir.isDirectory())
			return null;
		String name;
		if (MODEL_SIZE.keySet().contains(modelName))
			name = ALIAS.entrySet().stream().filter(ee -> ee.getValue().equals(modelName))
			.map(ee -> ee.getKey()).findFirst().get();
		else 
			name = modelName;
		File modelDir = Arrays.stream(dir.listFiles())
				.filter(ff -> ff.isDirectory() && ff.getName().startsWith(name + "_"))
				.findFirst().orElse(null);
		if (modelDir == null)
			return null;
		String weightsPath = modelDir.getAbsolutePath() + File.separator + ALIAS.get(name);
		File weigthsFile = new File(weightsPath);
		if (weigthsFile.isFile() && weigthsFile.length() == MODEL_SIZE.get(ALIAS.get(name)))
			return weightsPath;
		return null;
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
	
	public static String donwloadPretrained(String modelName, String downloadDir) 
			throws ExecutionException, InterruptedException, IOException {
		return donwloadPretrained(modelName, downloadDir, null);
	}
	
	public static String donwloadPretrained(String modelName, String downloadDir, Consumer<Double> progressConsumer) 
			throws ExecutionException, InterruptedException, IOException {
		String path = donwloadPretrainedOfficial(modelName, downloadDir, progressConsumer);
		if (path == null)
			path = donwloadPretrainedBioimageio(modelName, downloadDir, progressConsumer);
		if (path == null)
			throw new IllegalArgumentException("The model does not correspond to on of the available pretrained cellpose models."
					+ " To find a list of available cellpose models, please run Cellpose.getPretrainedList()");
		return path;
	}
	
	private static String donwloadPretrainedBioimageio(String modelName, String downloadDir, Consumer<Double> progressConsumer) 
			throws InterruptedException, IOException {
		
		BioimageioRepo br = BioimageioRepo.connect();

		ModelDescriptor descriptor = br.selectByName(modelName);
		if (descriptor == null)
			descriptor = br.selectByID(modelName);
		if (descriptor == null)
			return null;
		String path = BioimageioRepo.downloadModel(descriptor, downloadDir);
		return path + File.separator + ""; // TODO
	}
	
	private static String donwloadPretrainedOfficial(String modelName, String downloadDir, Consumer<Double> progressConsumer) throws ExecutionException, InterruptedException {
		List<URL> urls = new ArrayList<URL>();
		if (!MODEL_REQ.keySet().contains(modelName))
			return null;
		for (String str : MODEL_REQ.get(modelName)) {
			try {
				urls.add(new URL(String.format(CELLPOSE_URL, str)));
			} catch (MalformedURLException e) {
			}
		}
		MultiFileDownloader mfd = new MultiFileDownloader(urls, new File(downloadDir));
		if (progressConsumer != null)
			mfd.setPartialProgressConsumer(progressConsumer);
		mfd.download();
		return downloadDir + File.separator + MODEL_REQ.get(modelName)[0];
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
			mamba.pipInstallIn(DLModelPytorch.COMMON_PYTORCH_ENV_NAME, CELLPOSE_DEPS.toArray(new String[CELLPOSE_DEPS.size()]));
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

	@Override
	protected String createImportsCode() {
		String modelName = new File(weightsPath).getName();
		String modelType = ALIAS.entrySet().stream()
				.filter(ee -> ee.getValue().equals(modelName))
				.map(ee -> ee.getKey()).findFirst().get();
		return String.format(LOAD_MODEL_CODE_ABSTRACT, modelType);
	}

	@Override
	// TODO add 3d
	protected <T extends RealType<T> & NativeType<T>> void checkInput(RandomAccessibleInterval<T> image) {
		long[] dims = image.dimensionsAsLongArray();
		if (channels != null && !is3D && dims.length == 2 && (channels.length < 2 || channels[0] != 0 || channels[1] != 0))
			throw new IllegalArgumentException("To process a 2d grayscale image, the channels parameter must be [0, 0]");
		else if (channels != null && !is3D && dims.length == 3
				&& Arrays.stream(dims).anyMatch(num -> num == 1) && (channels.length < 2 || channels[0] != 0 || channels[1] != 0))
			throw new IllegalArgumentException("To process a 2d grayscale image, the channels parameter must be [0, 0]");
		else if (channels != null && !is3D && dims.length == 3
				&& (channels.length < 2 || channels[0] == 0 || channels[1] == 0))
			throw new IllegalArgumentException("To process a 2d RGB image, the channels parameter must be [2, 3]"
					+ " or [2, 1], depending whether the blue or the red channels contains nuclei.");
		else if (channels != null && !is3D && dims.length == 3 && isRedChannelEmpty(image)
				&& (channels.length < 2 || channels[0] != 2 || channels[1] != 3))
			throw new IllegalArgumentException("To process a 2d RGB image, the channels parameter must be [2, 3]"
					+ " if the cytoplasm is green and the nuclei are blue.");
		else if (channels != null && !is3D && dims.length == 3 && !isRedChannelEmpty(image)
				&& (channels.length < 2 || channels[0] != 2 || channels[1] != 1))
			throw new IllegalArgumentException("To process a 2d RGB image, the channels parameter must be [2, 1]"
					+ " if the cytoplasm is green and the nuclei are red.");
		else if (dims.length == 2 || (dims.length == 3 && Arrays.stream(dims).anyMatch(num -> num == 1) ))
			this.channels = new int[2];
		else if (dims.length == 3 && isRedChannelEmpty(image))
			this.channels = new int[] {2, 3};
		else if (dims.length == 3 && isRedChannelEmpty(image))
			this.channels = new int[] {2, 1};
	}
	
	private static <T extends RealType<T> & NativeType<T>> boolean isRedChannelEmpty(RandomAccessibleInterval<T> image) {
		// TODO
		return true;
	}

	@SuppressWarnings("unchecked")
	@Override
	protected <T extends RealType<T> & NativeType<T>> Map<String, RandomAccessibleInterval<T>> reconstructOutputs(
			Task task) throws IOException {

		Map<String, RandomAccessibleInterval<T>> outs = new LinkedHashMap<String, RandomAccessibleInterval<T>>();
		outs.put("mask", reconstructMask());
		shma.close();
		outs.put("flows", SpecialModelBase.copy((RandomAccessibleInterval<T>) shmaFl.getSharedRAI()));
		shmaFl.close();
		outs.put("denoised", SpecialModelBase.copy((RandomAccessibleInterval<T>) shmaDn.getSharedRAI()));
		shmaDn.close();
		outs.put("styles", SpecialModelBase.copy((RandomAccessibleInterval<T>) shmaSt.getSharedRAI()));
		shmaSt.close();
		return outs;
	}
	
	private <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() {
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> mask = shma.getSharedRAI();
		if (shma.getOriginalShape().length == 2 
				|| axes.indexOf("c") == -1
				|| shma.getOriginalShape()[axes.indexOf("c")] == 1) {
			return SpecialModelBase.copy(mask);
		} else {
			long[] maxPos = mask.maxAsLongArray();
			maxPos[2] = 0;
			IntervalView<T> maskInterval = Views.interval(mask, mask.minAsLongArray(), maxPos);
			return SpecialModelBase.copy(maskInterval);
		}
	}

	protected String createRunModelCode() {
		String code = "";
		if (shma.getOriginalShape().length == 2 && shma.getOriginalShape()[axes.indexOf("c")] == -1) {
			code = "im[:]";
		} else if (shma.getOriginalShape().length == 3 && shma.getOriginalShape()[axes.indexOf("c")] == 0) {
			code = "im[0, :, :]";
		} else if (shma.getOriginalShape().length == 3 && shma.getOriginalShape()[axes.indexOf("c")] == 1) {
			code = "im[:, 0, :]";
		} else if (shma.getOriginalShape().length == 3 && shma.getOriginalShape()[axes.indexOf("c")] == 2) {
			code = "im[:, :, 0]";
		} else if (shma.getOriginalShape().length == 4 && shma.getOriginalShape()[axes.indexOf("c")] == 0) {
			code = "im[0, :, :, :]";
		} else if (shma.getOriginalShape().length == 4 && shma.getOriginalShape()[axes.indexOf("c")] == 1) {
			code = "im[:, 0, :, :]";
		} else if (shma.getOriginalShape().length == 4 && shma.getOriginalShape()[axes.indexOf("c")] == 2) {
			code = "im[:, :, 0, :]";
		} else if (shma.getOriginalShape().length == 4 && shma.getOriginalShape()[axes.indexOf("c")] == 3) {
			code = "im[:, :, :, 0]";
		} else {
			throw new RuntimeException("shape: " + Arrays.toString(shma.getOriginalShape()) + ", axes: " + axes);
		}
		code += ", fl[:], st[:], dn[:] = model.eval(im, diameter=";
		if (this.diameter == null && new File(sizeWeigthsPath).isFile()) {
			code += "None";
		} else if (diameter != null) {
			code += this.diameter;
		} else {
			code += DEFAULT_DIAMETER;
		}
		code += ""
				+ ", channels=["
				+ channels[0] + "," + channels[1] + "])" + System.lineSeparator()
				+ "if os.name == 'nt':" + System.lineSeparator()
				+ "    im_shm.close()" + System.lineSeparator()
				+ "    im_shm.unlink()" + System.lineSeparator()
				+ "    fl_shm.close()" + System.lineSeparator()
				+ "    fl_shm.unlink()" + System.lineSeparator()
				+ "    st_shm.close()" + System.lineSeparator()
				+ "    st_shm.unlink()" + System.lineSeparator()
				+ "    dn_shm.close()" + System.lineSeparator()
				+ "    dn_shm.unlink()" + System.lineSeparator();
		return code;
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> run(RandomAccessibleInterval<T> img) throws IOException, InterruptedException {
		checkInput(img);
		shma = SharedMemoryArray.createSHMAFromRAI(img, false, false);
		long[] dnShape = new long[] {};
		shmaDn = SharedMemoryArray.create(dnShape, new FloatType(0), false, false);
		long[] flShape = new long[] {};
		shmaFl = SharedMemoryArray.create(flShape, new FloatType(0), false, false);
		long[] stShape = new long[] {};
		shmaSt = SharedMemoryArray.create(stShape, new FloatType(0), false, false);
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
		code += createEncodeImageScript() + System.lineSeparator();
		code += createRunModelCode() + System.lineSeparator();
		
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

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> void run(
			List<Tensor<T>> inTensors, List<Tensor<R>> outTensors) throws RunModelException {
		if (inTensors.size() > 1)
			throw new IllegalArgumentException("Cellpose can only take one argument.");
		if (this.modelType.equals("bioimage.io") && !inTensors.get(0).getAxesOrderString().equals(axes))
			throw new IllegalArgumentException("Input axes should be " + axes);
		else if (!inTensors.get(0).getAxesOrderString().equals(axes) 
				&& !inTensors.get(0).getAxesOrderString().replace("c", "").equals(axes.replace("c", "")))
			throw new IllegalArgumentException("Input axes should be " + axes);
		if (!modelType.equals("bioimage.io") 
				&& inTensors.get(0).getAxesOrderString().replace("c", "").equals(axes.replace("c", "")))
				axes = inTensors.get(0).getAxesOrderString();
		try {
			Map<String, RandomAccessibleInterval<R>> outputs = run(inTensors.get(0).getData());
			for (Tensor<R> tensor : outTensors) {
				Entry<String, RandomAccessibleInterval<R>> entry = outputs.entrySet().stream()
						.filter(ee -> tensor.getName().equals(ee.getKey())
								&& Arrays.equals(tensor.getData().dimensionsAsLongArray(), ee.getValue().dimensionsAsLongArray()))
						.findFirst().orElse(null);
				if (entry != null)
					tensor.setData(entry.getValue());
			}
		} catch (IOException | InterruptedException e) {
			throw new RunModelException(Types.stackTrace(e));
		}
		
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		if (inputTensors.size() > 1)
			throw new IllegalArgumentException("Cellpose can only take one argument.");
		if (this.modelType.equals("bioimage.io") && !inputTensors.get(0).getAxesOrderString().equals(axes))
			throw new IllegalArgumentException("Input axes should be " + axes);
		else if (!inputTensors.get(0).getAxesOrderString().equals(axes) 
				&& !inputTensors.get(0).getAxesOrderString().replace("c", "").equals(axes.replace("c", "")))
			throw new IllegalArgumentException("Input axes should be " + axes);
		if (!modelType.equals("bioimage.io") 
				&& inputTensors.get(0).getAxesOrderString().replace("c", "").equals(axes.replace("c", "")))
				axes = inputTensors.get(0).getAxesOrderString();
		try {
			Map<String, RandomAccessibleInterval<T>> outputs = run(inputTensors.get(0).getData());
			List<Tensor<T>> outTensors = new ArrayList<Tensor<T>>();
			for (Entry<String, RandomAccessibleInterval<T>> entry : outputs.entrySet()) {
				if (entry.getValue() == null)
					continue;
				String axesOrder = "xy";
				if (entry.getValue().dimensionsAsLongArray().length > 2 && this.is3D)
					axesOrder += "c";
				else if (entry.getValue().dimensionsAsLongArray().length == 3 && this.is3D)
					axesOrder += "z";
				else if (entry.getValue().dimensionsAsLongArray().length > 3 && this.is3D)
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
}
