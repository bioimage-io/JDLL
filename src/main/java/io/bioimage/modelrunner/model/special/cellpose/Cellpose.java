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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
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
	
	private Integer diameter;
	
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
			+ MODEL_VAR_NAME + " = denoise.CellposeDenoiseModel(gpu=%s, pretrained_model='%s')" + System.lineSeparator()
			+ "globals()['" + MODEL_VAR_NAME + "'] = " + MODEL_VAR_NAME + System.lineSeparator();
	
	protected static final String PATH_TO_RDF = "special_models/cellpose/rdf.yaml";
	
	protected static final URL RDF_URL = Cellpose.class.getClassLoader().getResource(PATH_TO_RDF);
	
	private static final String ONE_CHANNEL_STR = "ch_0";
	
	private static final String TWO_CHANNEL_STR = "ch_0, ch_1";
	
	private static final String THREE_CHANNEL_STR = "ch_0, ch_1, ch_3";

	protected Cellpose(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor) throws IOException {
		super(modelFile, callable, weightsPath, kwargs, descriptor, true);
    	createPythonService();
	}
	
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
	
	public void setDiameter(int diameter) {
		this.diameter = diameter;
	}
	
	public int getDiameter() {
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

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		createCustomDescriptor(inputTensors);
		super.run(checkInputTensors(inputTensors), checkOutputTensors(outputTensors));
	}
	
	protected String buildModelCode() {
		if (this.isBMZ)
			return super.buildModelCode();
		String code = String.format(LOAD_MODEL_CODE_ABSTRACT, 
				"False", // TODO GPU 
				this.weightsPath);
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	String createInputsCode(List<RandomAccessibleInterval<T>> inRais, List<String> names) {
		if (this.isBMZ)
			return super.createInputsCode(inRais, names);
		String code = "";
		for (int i = 0; i < inRais.size(); i ++) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(inRais.get(i), false, false);
			code += codeToConvertShmaToPython(shma, names.get(i));
			inShmaList.add(shma);
		}
		code += "print(type(" + names.get(0)  + "))" + System.lineSeparator();
		code += "print(" + names.get(0) + ".shape)" + System.lineSeparator();
		code += OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + ".eval(";
		for (int i = 0; i < inRais.size(); i ++)
			code += names.get(i) + ", channels=" + createChannelsArgCode(inRais.get(i)) +", ";
		code += "diameter=" + createDiamCode() + ")" + System.lineSeparator();
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
	
	protected String createDiamCode() {
		if (this.diameter  == null)
			return "None";
		else
			return "" + diameter;
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param descriptor
	 * 	the bioimage.io model descriptor
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 */
	public static Cellpose init(String weightsPath) throws IOException {
		if (!(new File(weightsPath).isFile()))
			throw new IllegalArgumentException("The path provided does not correspond to an existing file: " + weightsPath);		        
        Cellpose cellpose = new Cellpose(null, null, weightsPath, null, null);
		StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(new File(RDF_URL.toURI())))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append(System.lineSeparator());
            }
            cellpose.rdfString = content.toString();
        } catch (IOException | URISyntaxException e) {
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
	 * @param forceDownload
	 * 	whether to force the download or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ExecutionException 
	 */
	public static Cellpose fromPretained(String pretrainedModel, boolean forceDownload) throws IOException, InterruptedException, ExecutionException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), forceDownload);
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
	public static Cellpose fromPretained(String pretrainedModel, String modelsDir, boolean forceInstall) throws IOException, 
																					InterruptedException, ExecutionException {
		if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel) && !forceInstall) {
			String weightsPath = fileIsCellpose(pretrainedModel, modelsDir);
			if (weightsPath != null) return init(weightsPath);
			String fname = MultiFileDownloader.addTimeStampToFileName(pretrainedModel, true);
			fname = modelsDir + File.separator + fname;
			String path = donwloadPretrainedOfficial(pretrainedModel, fname, null);
			return init(path);
		} else if (PRETRAINED_CELLPOSE_MODELS.contains(pretrainedModel)) {
			String fname = MultiFileDownloader.addTimeStampToFileName(pretrainedModel, true);
			fname = modelsDir + File.separator + fname;
			String path = donwloadPretrainedOfficial(pretrainedModel, fname, null);
			return init(path);
		}
		if (!forceInstall) {
			List<ModelDescriptor> localModels = ModelDescriptorFactory.getModelsAtLocalRepo();
			ModelDescriptor model = localModels.stream()
					.filter(md -> md.getModelID().equals(pretrainedModel) 
							|| md.getName().toLowerCase().equals(pretrainedModel.toLowerCase()))
					.findFirst().orElse(null);
			if (model != null)
				return Cellpose.init(model);
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
	
	private static String fileIsCellpose(String pretrainedModel, String modelsDir) {
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
	
	private static List<String> findDirectoriesWithPattern(String folderPath, String keyword) {
        // Regex pattern to match: keyword_ddMMyyyy_HHmmss
        String regex = "^" + Pattern.quote(keyword) + "_\\d{8}_\\d{6}$";
        Pattern pattern = Pattern.compile(regex);
        return Arrays.stream(new File(folderPath).listFiles())
        		.filter(File::isDirectory)
        		.filter(ff -> pattern.matcher(ff.getName()).matches())
        		.map(ff -> ff.getAbsolutePath())
        		.collect(Collectors.toList());
    }
	
	
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
