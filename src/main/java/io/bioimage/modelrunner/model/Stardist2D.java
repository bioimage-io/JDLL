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
package io.bioimage.modelrunner.model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.model.stardist_java_deprecate.StardistConfig;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.runmode.ops.GenericOp;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.JSONUtils;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Implementation of an API to run Stardist 2D models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *TODO add support for Mac arm
 *
 *@author Carlos Garcia
 */
public class Stardist2D {
	
	private String modelDir;
	
	private final String name;
	
	private final String basedir;
	
	private boolean loaded = false;
	
	private SharedMemoryArray shma;
	
	private ModelDescriptor descriptor;
	
	private final int channels;
	
	private Environment env;
	
	private Service python;
	
	private static final List<String> STARDIST_DEPS = Arrays.asList(new String[] {"python=3.10", "stardist", "numpy", "appose"});
	
	private static final List<String> STARDIST_CHANNELS = Arrays.asList(new String[] {"conda-forge", "default"});

	
	private static final String COORDS_DTYPE_KEY = "coords_dtype";
	
	private static final String COORDS_SHAPE_KEY = "coords_shape";
	
	private static final String POINTS_DTYPE_KEY = "points_dtype";
	
	private static final String POINTS_SHAPE_KEY = "points_shape";
	
	private static final String POINTS_KEY = "points";
	
	private static final String COORDS_KEY = "coords";
	
	private static final String LOAD_MODEL_CODE = ""
			+ "if 'StarDist2D' not in globals().keys():" + System.lineSeparator()
			+ "  from stardist.models import StarDist2D" + System.lineSeparator()
			+ "  globals()['StarDist2D'] = StarDist2D" + System.lineSeparator()
			+ "if 'np' not in globals().keys():" + System.lineSeparator()
			+ "  import numpy as np" + System.lineSeparator()
			+ "  globals()['np'] = np" + System.lineSeparator()
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  globals()['os'] = os" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
			+ "model = StarDist2D(None, name='%s', basedir='%s')" + System.lineSeparator()
			+ "globals()['model'] = model";
	
	private static final String RUN_MODEL_CODE = ""
			+ "shm_coords_id = task.inputs['shm_coords_id']" + System.lineSeparator()
			+ "shm_points_id = task.inputs['shm_points_id']" + System.lineSeparator()
			+ "output = model.predict_instances(im, returnPredcit=False)" + System.lineSeparator()
			+ "im[:] = output[0]" + System.lineSeparator()
			+ "task.outputs['coords_shape'] = output[1]['coords'].shape" + System.lineSeparator()
			+ "task.outputs['coords_dtype'] = output[1]['coords'].dtype" + System.lineSeparator()
			+ "task.outputs['points_shape'] = output[1]['points'].shape" + System.lineSeparator()
			+ "task.outputs['points_dtype'] = output[1]['points'].dtype" + System.lineSeparator()
			+ "coords_shm = "
			+ "shared_memory.SharedMemory(create=True, name=os.path.basename(shm_points_id), size=output[1]['coords'].nbytes)" + System.lineSeparator()
			+ "shared_coords = np.ndarray(output[1]['coords'].shape, dtype=output[1]['coords'].dtype, buffer=coords_shm.buf)" + System.lineSeparator()
			+ "points_shm = "
			+ "shared_memory.SharedMemory(create=True, name=os.path.basename(shm_points_id), size=output[1]['points'].nbytes)" + System.lineSeparator()
			+ "shared_points = np.ndarray(output[1]['points'].shape, dtype=output[1]['points'].dtype, buffer=points_shm.buf)" + System.lineSeparator()
			+ "globals()['shared_points'] = shared_points" + System.lineSeparator()
			+ "globals()['shared_coords'] = shared_coords" + System.lineSeparator()
			+ "globals()['shared_coords'] = shared_coords" + System.lineSeparator()
			+ "if os.name == 'nt':" + System.lineSeparator()
			+ "  im_shm.close()" + System.lineSeparator()
			+ "  im_shm.unlink()" + System.lineSeparator();
	
	private static final String CLOSE_SHM_CODE = ""
			+ "points_shm.close()" + System.lineSeparator()
			+ "points_shm.unlink()" + System.lineSeparator()
			+ "coords_shm.close()" + System.lineSeparator()
			+ "coords_shm.unlink()" + System.lineSeparator();
	
	private Stardist2D(String modelName, String baseDir) {
		this.name = modelName;
		this.basedir = baseDir;
		modelDir = new File(baseDir, modelName).getAbsolutePath();
		if (new File(modelDir, "config.json").isFile() == false && new File(modelDir, Constants.RDF_FNAME).isFile() == false)
			throw new IllegalArgumentException("No 'config.json' file found in the model directory");
		else if (new File(modelDir, "config.json").isFile() == false)
			createConfigFromBioimageio();
    	Map<String, Object> stardistMap = (Map<String, Object>) descriptor.getConfig().getSpecMap().get("stardist");
    	Map<String, Object> stardistConfig = (Map<String, Object>) stardistMap.get("config");
    	Map<String, Object> stardistThres = (Map<String, Object>) stardistMap.get("thresholds");
		this.channels = (int) stardistConfig.get("n_channel_in");
		
	}
	
	private void createConfigFromBioimageio() {
		
	}
	
	private void loadModel() throws IOException, InterruptedException {
		if (loaded)
			return;
		String code =  String.format(LOAD_MODEL_CODE, this.name, this.basedir);
		Task task = python.task(code);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException("Task canceled");
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException(task.error);
		else if (task.status == TaskStatus.CRASHED)
			throw new RuntimeException(task.error);
		loaded = true;
	}
	
	
	protected String createEncodeImageScript() {
		String code = "";
		// This line wants to recreate the original numpy array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = np.ndarray(size, dtype="float64", buffer=input0_appose_shm.buf).reshape([64, 64])
		code += "im_shm = shared_memory.SharedMemory(name='"
							+ shma.getNameForPython() + "', size=" + shma.getSize() 
							+ ")" + System.lineSeparator();
		code += "im = np.ndarray(" + shma.getSize()  + ", dtype='" + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
			  + "', buffer=im_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i ++)
			code += shma.getOriginalShape()[i] + ", ";
		code += "])" + System.lineSeparator();
		return code;
	}
	
	public void close() {
		if (!loaded)
			return;
		python.close();
	}
	
	public <T extends RealType<T> & NativeType<T>> 
	Map<String, RandomAccessibleInterval<T>> predict(RandomAccessibleInterval<T> img) throws IOException, InterruptedException {
		
		shma = SharedMemoryArray.createSHMAFromRAI(img);
		String code = "";
		if (!loaded) {
			code += String.format(LOAD_MODEL_CODE, this.name, this.basedir) + System.lineSeparator();
		}
		
		code += createEncodeImageScript() + System.lineSeparator();
		code += RUN_MODEL_CODE + System.lineSeparator();

		
		Map<String, Object> inputs = new HashMap<String, Object>();
		String shm_coords_id = SharedMemoryArray.createShmName();
		String shm_points_id = SharedMemoryArray.createShmName();
		inputs.put("shm_coords_id", shm_coords_id);
		inputs.put("shm_points_id", shm_points_id);
		
		Task task = python.task(code, inputs);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException("Task canceled");
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException(task.error);
		else if (task.status == TaskStatus.CRASHED)
			throw new RuntimeException(task.error);
		loaded = true;
		
		
		return reconstructOutputs(task, shm_coords_id, shm_points_id);
	}
	
	private <T extends RealType<T> & NativeType<T>> 
	Map<String, RandomAccessibleInterval<T>> reconstructOutputs(Task task, String shm_coords_id, String shm_points_id) 
			throws IOException, InterruptedException {
		
		String coords_dtype = (String) task.outputs.get("coords_dtype");
		List<Number> coords_shape = (List<Number>) task.outputs.get("coords_shape");
		String points_dtype = (String) task.outputs.get("points_dtype");
		List<Number> points_shape = (List<Number>) task.outputs.get("points_shape");
		
		long[] coordsSh = new long[coords_shape.size()];
		for (int i = 0; i < coordsSh.length; i ++)
			coordsSh[i] = coords_shape.get(i).longValue();
		SharedMemoryArray shmCoords = SharedMemoryArray.readOrCreate(shm_coords_id, coordsSh, 
				Cast.unchecked(CommonUtils.getImgLib2DataType(coords_dtype)), false, false);
		
		long[] pointsSh = new long[points_shape.size()];
		for (int i = 0; i < pointsSh.length; i ++)
			pointsSh[i] = points_shape.get(i).longValue();
		SharedMemoryArray shmPoints = SharedMemoryArray.readOrCreate(shm_points_id, pointsSh, 
				Cast.unchecked(CommonUtils.getImgLib2DataType(points_dtype)), false, false);
		
		Map<String, RandomAccessibleInterval<T>> outs = new HashMap<String, RandomAccessibleInterval<T>>();
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(shma.getSharedRAI()), 
				Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
		outs.put("mask", maskCopy);
		RandomAccessibleInterval<T> pointsRAI = shmPoints.getSharedRAI();
		RandomAccessibleInterval<T> pointsCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(pointsRAI), 
				Util.getTypeFromInterval(Cast.unchecked(pointsRAI)));
		outs.put("points", pointsCopy);
		RandomAccessibleInterval<T> coordsRAI = shmCoords.getSharedRAI();
		RandomAccessibleInterval<T> coordsCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(shmCoords), 
				Util.getTypeFromInterval(Cast.unchecked(shmCoords)));
		outs.put("coords", coordsCopy);
		
		shma.close();
		shmCoords.close();
		shmPoints.close();
		
		if (PlatformDetection.isWindows()) {
			Task closeSHMTask = python.task(CLOSE_SHM_CODE);
			closeSHMTask.waitFor();
		}
		return outs;
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param modelPath
	 * 	path to the Bioimage.io model
	 * @return an instance of a Stardist2D model ready to be used
     * @throws ModelSpecsException If there is any error in the configuration of the specs rdf.yaml file of the Bioimage.io.
     * @throws FileNotFoundException If the model file is not found.
     * @throws IOException If there's an I/O error.
	 */
	public static Stardist2D fromBioimageioModel(String modelPath) throws ModelSpecsException, FileNotFoundException, IOException {
		ModelDescriptor descriptor = ModelDescriptorFactory.readFromLocalFile(modelPath + File.separator + Constants.RDF_FNAME);
		return new Stardist2D(modelPath);
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 2D models.
	 * By default, the model will be installed in the "models" folder inside the application
	 * @param pretrainedModel
	 * 	the name of the pretrained model. 
	 * @param forceInstall
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ModelSpecsException if the model downloaded is not well specified in the config file
	 */
	public static Stardist2D fromPretained(String pretrainedModel, boolean forceInstall) throws IOException, InterruptedException, ModelSpecsException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), forceInstall);
	}
	
	/**
	 * TODO add support for 2D_paper_dsb2018
	 * Initialize one of the "official" pretrained Stardist 2D models
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param installDir
	 * 	the directory where the model wants to be installed
	 * @param forceInstall
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ModelSpecsException if the model downloaded is not well specified in the config file
	 */
	public static Stardist2D fromPretained(String pretrainedModel, String installDir, boolean forceInstall) throws IOException, 
																					InterruptedException, 
																					ModelSpecsException {
		if ((pretrainedModel.equals("StarDist H&E Nuclei Segmentation")
				|| pretrainedModel.equals("2D_versatile_he")) && !forceInstall) {
			ModelDescriptor md = ModelDescriptorFactory.getModelsAtLocalRepo().stream()
					.filter(mm ->mm.getName().equals("StarDist H&E Nuclei Segmentation")).findFirst().orElse(null);
			if (md != null) return new Stardist2D(md);
			String path = BioimageioRepo.connect().downloadByName("StarDist H&E Nuclei Segmentation", installDir);
			return Stardist2D.fromBioimageioModel(path);
		} else if (pretrainedModel.equals("StarDist H&E Nuclei Segmentation")
				|| pretrainedModel.equals("2D_versatile_he")) {
			String path = BioimageioRepo.connect().downloadByName("StarDist H&E Nuclei Segmentation", installDir);
			return Stardist2D.fromBioimageioModel(path);
		} else if ((pretrainedModel.equals("StarDist Fluorescence Nuclei Segmentation")
				|| pretrainedModel.equals("2D_versatile_fluo")) && !forceInstall) {
			ModelDescriptor md = ModelDescriptorFactory.getModelsAtLocalRepo().stream()
					.filter(mm ->mm.getName().equals("StarDist Fluorescence Nuclei Segmentation")).findFirst().orElse(null);
			if (md != null) return new Stardist2D(md);
			String path = BioimageioRepo.connect().downloadByName("StarDist Fluorescence Nuclei Segmentation", installDir);
			return Stardist2D.fromBioimageioModel(path);
		} else if (pretrainedModel.equals("StarDist Fluorescence Nuclei Segmentation")
				|| pretrainedModel.equals("2D_versatile_fluo")) {
			String path = BioimageioRepo.connect().downloadByName("StarDist Fluorescence Nuclei Segmentation", installDir);
			return Stardist2D.fromBioimageioModel(path);
		} else {
			throw new IllegalArgumentException("There is no Stardist2D model called: " + pretrainedModel);
		}
	}
	
	private <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image) {
		if (image.dimensionsAsLongArray().length == 2 && this.channels != 1)
			throw new IllegalArgumentException("Stardist2D needs an image with three dimensions: XYC");
		else if (image.dimensionsAsLongArray().length != 3 && this.channels != 1)
			throw new IllegalArgumentException("Stardist2D needs an image with three dimensions: XYC");
		else if (image.dimensionsAsLongArray().length != 2 && image.dimensionsAsLongArray()[2] != channels)
			throw new IllegalArgumentException("This Stardist2D model requires " + channels + " channels.");
		else if (image.dimensionsAsLongArray().length > 3 || image.dimensionsAsLongArray().length < 2)
			throw new IllegalArgumentException("Stardist2D model requires an image with dimensions XYC.");
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 */
	public void checkRequirementsInstalled() {
		// TODO
	}
	
	/**
	 * Check whether the requirements needed to run Stardist 2D are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for Stardist2D post processing is fine too.
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
		boolean installed = InstalledEngines.buildEnginesFinder()
				.checkEngineWithArgsInstalledForOS("tensorflow", "1.15.0", null, null).size() != 0;
		if (!installed)
			EngineInstall.installEngineWithArgs("tensorflow", "1.15.0", true, true);
		
		Mamba mamba = new Mamba();
		boolean stardistPythonInstalled = false;
		try {
			stardistPythonInstalled = mamba.checkAllDependenciesInEnv("stardist", STARDIST_DEPS);
		} catch (MambaInstallException e) {
			mamba.installMicromamba();
		}
		if (!stardistPythonInstalled) {
			// TODO add logging for environment installation
			mamba.create("stardist", true, STARDIST_CHANNELS, STARDIST_DEPS);
		};
	}
	
	/**
	 * Main method to check functionality
	 * @param args
	 * 	nothing
	 * @throws IOException nothing
	 * @throws InterruptedException nothing
	 * @throws RuntimeException nothing
	 * @throws MambaInstallException nothing
	 * @throws ModelSpecsException nothing
	 * @throws LoadEngineException nothing
	 * @throws RunModelException nothing
	 * @throws ArchiveException nothing
	 * @throws URISyntaxException nothing
	 * @throws LoadModelException nothing
	 */
	public static void main(String[] args) throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ModelSpecsException, LoadEngineException, 
													RunModelException, ArchiveException, 
													URISyntaxException, LoadModelException {
		Stardist2D.installRequirements();
		Stardist2D model = Stardist2D.fromPretained("2D_versatile_fluo", false);
		
		RandomAccessibleInterval<FloatType> img = ArrayImgs.floats(new long[] {512, 512});
		
		RandomAccessibleInterval<FloatType> res = model.predict(img);
		System.out.println(true);
	}
}
