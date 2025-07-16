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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.tiling.TileCalculator;
import io.bioimage.modelrunner.download.MultiFileDownloader;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.python.BioimageIoModelPytorchProtected;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
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
public class Cellpose extends BioimageIoModelPytorchProtected {
		
	protected boolean isBMZ;
	
	protected int[] channels;
	
	private Float diameter;
	
	protected String setDiameterCode = "";
	
	private String rdfString;
	
	private boolean is3D = false;
	
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
			+ "gpu_available = False" + System.lineSeparator() // TODO GPU
			+ ((IS_ARM) 
					? "" 
					: "from torch.backends import mps" + System.lineSeparator()
					+ "if mps.is_built() and mps.is_available():" + System.lineSeparator()
					+ "  gpu_available = True" + System.lineSeparator())
			+ MODEL_VAR_NAME + " = denoise.CellposeDenoiseModel(gpu=gpu_available, pretrained_model=r'%s')" + System.lineSeparator()
			+ "globals()['" + MODEL_VAR_NAME + "'] = " + MODEL_VAR_NAME + System.lineSeparator();
	
	protected static final String PATH_TO_RDF = "special_models/cellpose/rdf.yaml";
	
	protected static final URL RDF_URL = Cellpose.class.getClassLoader().getResource(PATH_TO_RDF);
	
	private static final String ONE_CHANNEL_STR = "ch_0";
	
	private static final String TWO_CHANNEL_STR = "ch_0, ch_1";
	
	private static final String THREE_CHANNEL_STR = "ch_0, ch_1, ch_3";

	protected Cellpose(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor) throws IOException {
		super(modelFile, callable, null, weightsPath, kwargs, descriptor, true);
    	createPythonService();
	}
	
	/**
	 * Set the channels array required to run Cellpose. It always has to be an array of length 2.
	 * If the image is gray scale [0, 0].
	 * For RGB images: R=1, G=2, B=3, the first value is the channel where cytoplasm is and the second the nuclei.
	 * If green cyto and red nuclei: [2, 1]
	 * 
	 * @param channels
	 * 	channels paramter for cellpose
	 */
	public void setChannels(int[] channels) {
		/**
		 * TODO remove
		if (channels.length != 2)
			throw new IllegalArgumentException("The channels arrays can only be [0, 0], [2, 3] or [2, 1]."
					+ " For grayscale images [0, 0], for 3 channels images where the cytoplasm is in the second channel "
					+ "(green) and the nuclei are in the first channel (red), [2, 1]; and when the nuclei are in the "
					+ "third channel (blue), [2, 3].");
		 */
		this.channels = channels;
	}
	
	/**
	 * Set the mean diameter of the cells in the image. If not set and the diameter model (an .npy file) is
	 * present in the same folder as the Cellpose model, the diameter is calculated on the fly, if not set and 
	 * the duameter model is not present, it is set to the default value (30).
	 * @param diameter
	 * 	the mean diameter of cells in the image
	 */
	public void setDiameter(float diameter) {
		this.diameter = diameter;
		setDiameterCode = "diameter=" + diameter;
	}
	
	/**
	 * 
	 * @return the diameter that has been set by the user. Cannot return the diameter calculated by the diameter model.
	 */
	public Float getDiameter() {
		return this.diameter;
	}
	
	private static <T extends RealType<T> & NativeType<T>> boolean isRedChannelEmpty(RandomAccessibleInterval<T> image) {
		// TODO
		return true;
	}
	
	// TODO add 3D
	protected <R extends RealType<R> & NativeType<R>> 
	List<Tensor<R>> checkInputTensors(List<Tensor<R>> inputTensors) {
		if (inputTensors.size() > 1)
			throw new IllegalArgumentException("The input tensor list should contain just one tensor");
		if (!inputTensors.get(0).getAxesOrderString().equals("xy") && !inputTensors.get(0).getAxesOrderString().equals("xyc"))
			throw new IllegalArgumentException("The input axes should be 'xyc'");

		long[] dims = inputTensors.get(0).getData().dimensionsAsLongArray();
		if (dims.length == 2) {
			FinalInterval interval = new FinalInterval(new long[3], new long[] {dims[0], dims[1], 1});
			IntervalView<R> nData = Views.interval(inputTensors.get(0).getData(), interval);
			inputTensors.set(0, Tensor.build(inputTensors.get(0).getName(), "xyc", nData));
		} else if (dims.length == 3 && dims[2] != 3 && dims[2] != 1)
			throw new IllegalArgumentException("Only 1 and 3 channel images supported. The provided input has " + dims[2]);
		return inputTensors;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	List<Tensor<T>> checkOutputTensors(List<Tensor<T>> outputTensors) {
		// TODO 
		return outputTensors;
	}
	
	private <R extends RealType<R> & NativeType<R>> 
	void createCustomDescriptor(List<Tensor<R>> inputTensors) {
		int nChannels = 1;
		String axesOrder = inputTensors.get(0).getAxesOrderString().toLowerCase();
		if (axesOrder.contains("c")
				&& inputTensors.get(0).getData().dimensionsAsLongArray()[axesOrder.indexOf("c")] == 3)
			nChannels = 3;
		else if (axesOrder.contains("c")
				&& inputTensors.get(0).getData().dimensionsAsLongArray()[axesOrder.indexOf("c")] != 1)
			throw new IllegalArgumentException("Inputs to cellpose model can only have either 1 or 3 channels.");
		String adaptedRdfString;
		String weightsName = new File(this.weightsPath).getName();
		if (nChannels == 1)
			adaptedRdfString = String.format(rdfString, ONE_CHANNEL_STR, ONE_CHANNEL_STR, weightsName);
		else
			adaptedRdfString = String.format(rdfString, THREE_CHANNEL_STR, TWO_CHANNEL_STR, weightsName);
			
		this.descriptor = ModelDescriptorFactory.readFromYamlTextString(adaptedRdfString);
		descriptor.addModelPath(Paths.get(new File(this.weightsPath).getParentFile().getAbsolutePath()));
		this.tileCalculator = TileCalculator.init(descriptor);
	}
	
	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @return the resulting tensors 
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		createCustomDescriptor(inputTensors);
		return super.run(checkInputTensors(inputTensors));
	}

	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @param outputTensors
	 * 	list of output tensors that are expected to be returned by the model
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		createCustomDescriptor(inputTensors);
		super.run(checkInputTensors(inputTensors), checkOutputTensors(outputTensors));
	}
	
	protected String buildModelCode() throws IOException {
		if (this.isBMZ)
			return super.buildModelCode();
		String code = String.format(LOAD_MODEL_CODE_ABSTRACT, 
				//"False", // TODO GPU 
				this.weightsPath);
		code += ""
				+ "diameter = None" + System.lineSeparator()
				+ "globals()['diameter'] = diameter" + System.lineSeparator();
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	String createInputsCode(List<RandomAccessibleInterval<T>> inRais, List<String> names) {
		if (this.isBMZ)
			return super.createInputsCode(inRais, names);
		String code = setDiameterCode + System.lineSeparator();
		setDiameterCode = "";
		code += "created_shms = []" + System.lineSeparator();
		code += "try:" + System.lineSeparator();
		for (int i = 0; i < inRais.size(); i ++) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(inRais.get(i), false, false);
			code += codeToConvertShmaToPython(shma, names.get(i));
			inShmaList.add(shma);
		}
		String nameList = "[";
		String channelList = "[";
		for (int i = 0; i < inRais.size(); i ++) {
			nameList += names.get(i) + ", ";
			channelList += createChannelsArgCode(inRais.get(i)) + ", ";
		}
		nameList += "]";
		channelList += "]";
		code += createDiamCode(nameList, channelList);
		code += "  print(diameter)" + System.lineSeparator();
		code += "  " + OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + ".eval(" + nameList + ", channels=" + channelList + ", ";
		code += "diameter=diameter)" + System.lineSeparator();;
		String closeEverythingWin = closeSHMWin();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "except Exception as e:" + System.lineSeparator();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "  raise e" + System.lineSeparator();
		code += ""
				+ SHMS_KEY + " = []" + System.lineSeparator()
				+ SHM_NAMES_KEY + " = []" + System.lineSeparator()
				+ DTYPES_KEY + " = []" + System.lineSeparator()
				+ DIMS_KEY + " = []" + System.lineSeparator()
				+ "globals()['" + SHMS_KEY + "'] = " + SHMS_KEY + System.lineSeparator()
				+ "globals()['" + SHM_NAMES_KEY + "'] = " + SHM_NAMES_KEY + System.lineSeparator()
				+ "globals()['" + DTYPES_KEY + "'] = " + DTYPES_KEY + System.lineSeparator()
				+ "globals()['" + DIMS_KEY + "'] = " + DIMS_KEY + System.lineSeparator();
		code += "handle_output_list(" + OUTPUT_LIST_KEY + ")" + System.lineSeparator();
		code += taskOutputsCode();
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>> String createChannelsArgCode(RandomAccessibleInterval<T> rai) {
		long[] dims = rai.dimensionsAsLongArray();
		if (channels == null && dims.length == 2)
			return "[0, 0]";
		else if (channels == null && dims.length == 3 && dims[2] == 1)
			return "[0, 0]";
		else if (channels == null && dims.length == 3 && dims[2] == 1)
			return "[0, 0]";
		else if (channels == null && dims.length == 3 && dims[2] == 3 && isRedChannelEmpty(rai))
			return "[2, 3]";
		else if (channels == null && dims.length == 3 && dims[2] == 3)
			return "[2, 1]";
		else if (channels != null)
			return Arrays.toString(channels);
		else
			throw new IllegalArgumentException("Bad configuration, dims=" + Arrays.toString(dims) 
			+ ", channels=" + Arrays.toString(channels));
	}
	
	protected String createDiamCode(String nameList, String channelList) {
		String code = ""
				+ "  if diameter is None:" + System.lineSeparator()
				+ "    from cellpose.models import SizeModel" + System.lineSeparator()
				+ "    from pathlib import Path" + System.lineSeparator()
				+ "    p = Path(r'" + this.modelFolder + "')" + System.lineSeparator()
				+ "    pretrained_list = [f for f in p.glob(\"size_*.npy\") if f.is_file()]" + System.lineSeparator()
				+ "    if len(pretrained_list) > 0:" + System.lineSeparator()
				+ String.format("      sz = SizeModel(pretrained_size=pretrained_list[0], cp_model=%s.cp)", MODEL_VAR_NAME) + System.lineSeparator()
				+ "      diameter = sz.eval(";
		
		code += nameList + ", channels=" + channelList +")[0]" + System.lineSeparator();
		return code;
	}
	
	/**
	 * Initialize a Cellpose model with the path to the model weigths.
	 * @param weightsPath
	 * 	path to the weights of a pretrained cellpose model
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 */
	public static Cellpose init(String weightsPath) throws IOException {
		File wFile = new File(weightsPath);
		if (wFile.isDirectory() && new File(wFile, Constants.RDF_FNAME).isFile())
			return init(ModelDescriptorFactory.readFromLocalFile(new File(wFile, Constants.RDF_FNAME).getAbsolutePath()));
		if (!wFile.isFile())
			throw new IllegalArgumentException("The path provided does not correspond to an existing file: " + weightsPath);		        
        Cellpose cellpose = new Cellpose(null, null, weightsPath, null, null);
		try (InputStream in = RDF_URL.openStream();
		     ByteArrayOutputStream baos = new ByteArrayOutputStream()) {

		    byte[] buffer = new byte[8192];
		    int len;
		    while ((len = in.read(buffer)) != -1) {
		        baos.write(buffer, 0, len);
		    }
		    cellpose.rdfString = baos.toString(StandardCharsets.UTF_8.name());
		} catch (IOException e) {
        }
		return cellpose;
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param descriptor
	 * 	the bioimage.io model descriptor
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 */
	public static Cellpose init(ModelDescriptor descriptor) throws IOException {
		if (descriptor.getTags().stream().filter(tt -> tt.toLowerCase().equals("cellpose")).findFirst().orElse(null) == null
				&& !descriptor.getName().toLowerCase().contains("cellpose"))
			throw new RuntimeException("This model does not seem to be a cellpose model from the Bioimage.io");
		if (descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID()) == null)
			throw new IllegalArgumentException("The model provided does not have weights in the required format, "
					+ ModelWeight.getPytorchID() + ".");
		WeightFormat pytorchWeights = descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID());
		String modelFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getArchitecture().getSource();
		String callable = pytorchWeights.getArchitecture().getCallable();
		String weightsFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getSource();
		Map<String, Object> kwargs = pytorchWeights.getArchitecture().getKwargs();
		Cellpose model =  new Cellpose(modelFile, callable, weightsFile, kwargs, descriptor);
		model.isBMZ = true;
		return model;
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 2D models.
	 * By default, the model will be installed in the "models" folder inside the application
	 * @param pretrainedModel
	 * 	the name of the pretrained model. 
	 * @param install
	 * 	whether to force the download or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ExecutionException if there is an error downloading the model
	 */
	public static Cellpose fromPretained(String pretrainedModel, boolean install) throws IOException, InterruptedException, ExecutionException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), install);
	}
	
	/**
	 * Initialize one of the "official" pretrained cellpose ("cyto2", "cyto3"...) models or
	 * those available in the bioimage.io
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param modelsDir
	 * 	the directory where the model wants to be installed
	 * @param install
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ExecutionException if there is an error downloading the model
	 */
	public static Cellpose fromPretained(String pretrainedModel, String modelsDir, boolean install) throws IOException, 
																					InterruptedException, ExecutionException {
		if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel) && !install) {
			String weightsPath = fileIsCellpose(pretrainedModel, modelsDir);
			if (weightsPath != null) return init(weightsPath);
			return null;
		} else if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel)) {
			String path = donwloadPretrainedOfficial(pretrainedModel, modelsDir, null);
			return init(path);
		}
		if (!install) {
			List<ModelDescriptor> localModels = ModelDescriptorFactory.getModelsAtLocalRepo(modelsDir);
			ModelDescriptor model = localModels.stream()
					.filter(md -> md.getModelID().equals(pretrainedModel) 
							|| md.getName().toLowerCase().equals(pretrainedModel.toLowerCase()))
					.findFirst().orElse(null);
			if (model != null)
				return Cellpose.init(model);
			else 
				return null;
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
		return Cellpose.init(descriptor);
	}
	
	/**
	 * Finds whether a pretrained Cellpose model is installed in the wanted directory
	 * 
	 * @param modelName
	 * 	the name of the model, it can be either the name of one of the official Cellpose models (cyto, cyto2, cyto3...)
	 * 	or a path to the weigths
	 * @param modelsDir
	 * 	the directory where we want to know whether the model is installed or not
	 * @return the path to the model if if exists, null otherwise
	 */
	public static String findPretrainedModelInstalled(String modelName, String modelsDir) {
		if (modelName.endsWith(".pth"))
			modelName = modelName.substring(0, modelName.length() - 4);
		else if (modelName.endsWith(".pt"))
				modelName = modelName.substring(0, modelName.length() - 3);
		if (ALIAS.keySet().contains(modelName) || MODEL_SIZE.containsKey(modelName)) {
			 for (String dir : findDirectoriesWithPattern(modelsDir, modelName)) {
				 String path = lookForModelInDir(modelName, dir);
				 if (path != null)
					 return path;
			 }
		 } else {
			 throw new IllegalArgumentException("Only supported pretrained models are: " + ALIAS.keySet());
		 }
		 return null;
	}
	
	/**
	 * Find if the String argument 'pretrainedModel' corresponds to a cellpose model that exists
	 * in the local computer or not.
	 * For example if we provide simply the String 'cyto3' it will look for files within the modelsDir
	 * subfolders that might contain the model. It only checks two levels of subfolders.
	 * 
	 * We can also provide the full path to the cellpose model
	 * @param pretrainedModel
	 * 	a String referring to a Cellpose model that might exist in our local computer or not. It can be the full path
	 * 	to our model or just the name of the pretrained cellpose model (cyto, cyto2, cyto3....)
	 * @param modelsDir
	 * 	the directory where we will look for a cellpose model if the whole path to the model is not given
	 * @return the full path to a Cellpose model if it exists or null if it does not exists in the paths specified
	 */
	public static String fileIsCellpose(String pretrainedModel, String modelsDir) {
		File pretrainedFile = new File(pretrainedModel);
		 if (pretrainedFile.isFile() && isCellposeFile(pretrainedFile))
			 return pretrainedFile.getAbsolutePath();
		 String path = findPretrainedModelInstalled(pretrainedModel, modelsDir);
		 if (path != null)
			 return path;
		 return lookForModelInDir(pretrainedModel, modelsDir);
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
		String weightsPath = dir.getAbsolutePath() + File.separator + ALIAS.get(name);
		File weigthsFile = new File(weightsPath);
		if (weigthsFile.isFile() && weigthsFile.length() == MODEL_SIZE.get(ALIAS.get(name)))
			return weightsPath;
		return null;
	}
	
	/**
	 * 
	 * @return a list of the available pretrained Cellpose models that can be run with JDLL.
	 * 	It returns both the official cellpose models and the custom bioimage.io ones
	 */
	public static List<String> getPretrainedList() {
		List<String> list = new ArrayList<String>();
		BioimageioRepo br = BioimageioRepo.connect();
		Map<String, ModelDescriptor> models = br.listAllModels(false);
		list = models.entrySet().stream()
				.filter(ee -> ee.getValue().getModelFamily().equals(ModelDescriptor.CELLPOSE))
				.map(ee -> ee.getValue().getName()).collect(Collectors.toList());
		list.addAll(PRETRAINED_CELLPOSE_MODELS);
		return list;
	}
	
	/**
	 * Download a pretrained cellpose model. It is able to download both official 
	 * Cellpose releases (cyto, cyto2, cyto3 or nuclei) and Cellpose variants available 
	 * in the Bioimge.io model zoo.
	 * 
	 * @param modelName
	 * 	name of the pretrained cellpose model (cyto, cyto2, cyto3 or nuclei for official Cellpose releases)
	 * @param downloadDir
	 * 	directory where the model is going to be downloaded
	 * @return the folder of the model downloaded
	 * @throws ExecutionException if there is any error downloading the model
	 * @throws InterruptedException if the download is interrupted
	 * @throws IOException if there is any error writing the downloaded files
	 */
	public static String donwloadPretrained(String modelName, String downloadDir) 
			throws ExecutionException, InterruptedException, IOException {
		return donwloadPretrained(modelName, downloadDir, null);
	}
	
	/**
	 * Download a pretrained cellpose model. It is able to download both official 
	 * Cellpose releases (cyto, cyto2, cyto3 or nuclei) and Cellpose variants available 
	 * in the Bioimge.io model zoo.
	 * 
	 * @param modelName
	 * 	name of the pretrained cellpose model (cyto, cyto2, cyto3 or nuclei for official Cellpose releases)
	 * @param downloadDir
	 * 	directory where the model is going to be downloaded
	 * @param progressConsumer
	 * 	consumer that will notify the download progress
	 * @return the folder of the model downloaded
	 * @throws ExecutionException if there is any error downloading the model
	 * @throws InterruptedException if the download is interrupted
	 * @throws IOException if there is any error writing the downloaded files
	 */
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
		String path = BioimageioRepo.downloadModel(descriptor, downloadDir, progressConsumer);
		return path;
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
		String fname = MultiFileDownloader.addTimeStampToFileName(modelName, true);
		downloadDir = downloadDir + File.separator + fname;
		MultiFileDownloader mfd = new MultiFileDownloader(urls, new File(downloadDir));
		if (progressConsumer != null)
			mfd.setPartialProgressConsumer(progressConsumer);
		mfd.download();
		return downloadDir + File.separator + MODEL_REQ.get(modelName)[0];
	}
	
	private static List<String> findDirectoriesWithPattern(String folderPath, String keyword) {
        // Regex pattern to match: keyword_ddMMyyyy_HHmmss
        String regex = "^" + Pattern.quote(keyword) + "_\\d{8}_\\d{6}$";
        Pattern pattern = Pattern.compile(regex);
        if (new File(folderPath).isDirectory() == false)
        	return new ArrayList<String>();
        return Arrays.stream(new File(folderPath).listFiles())
        		.filter(File::isDirectory)
        		.filter(ff -> pattern.matcher(ff.getName()).matches())
        		.map(ff -> ff.getAbsolutePath())
        		.collect(Collectors.toList());
    }
	
	
	/**
	 * Example code that shows how to run a model with cellpose
	 * @param <T>
	 * 	method param
	 * @param args
	 * 	method param
	 * @throws IOException exception
	 * @throws InterruptedException exception
	 * @throws ExecutionException exception
	 * @throws LoadModelException exception
	 * @throws RunModelException exception
	 */
	public static <T extends RealType<T> & NativeType<T>>
	void main(String[] args) throws IOException, InterruptedException, ExecutionException, LoadModelException, RunModelException {
		Cellpose model = Cellpose.fromPretained("cyto2", false);
		model.loadModel();
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(new long[] {512, 512, 3});
		List<RandomAccessibleInterval<FloatType>> rais = new ArrayList<RandomAccessibleInterval<FloatType>>();
		rais.add(rai);
		long tt = System.currentTimeMillis();
		List<RandomAccessibleInterval<T>> res = model.inference(rais);
		System.out.println(System.currentTimeMillis() - tt);
		tt = System.currentTimeMillis();
		List<RandomAccessibleInterval<T>> rees = model.inference(rais);
		System.out.println(System.currentTimeMillis() - tt);
		model.close();
		System.out.println(false);
	}
}
