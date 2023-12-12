/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.runmode.ops;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Code for the JDLL OP that allows running the whole stardist model (pre-processing + model executio
 * + post-processing and tiling) in Python using Appose
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class StardistInferJdllOp implements OpInterface {
	
	private String modelName;
	
	private Tensor<?> inputTensor;
	
	private String opFilePath;
	
	private String envPath;
	
	private LinkedHashMap<String, Object> inputsMap;
	
	private final static String MODEL_KEY = "model";
	
	private final static String INPUT_TENSOR_KEY = "input_tensor";
	
	private static final String OP_METHOD_NAME = "stardist_prediction_2d_mine";
	
	protected static final String STARDIST_FIELD_KEY = "stardist";
	
	private static final int N_STARDIST_OUTPUTS = 2;
	
	private static final String STARDIST_OP_FNAME = "stardist_inference.py";
	
	
	public static void main(String[] args) throws Exception {
		final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
		final Img< FloatType > img1 = imgFactory.create( 1, 512, 512, 3 );
		Tensor<FloatType> inpTensor = Tensor.build("input0", "byxc", img1);
		String modelName = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\model-runner-java\\models\\StarDist H&E Nuclei Segmentation_06092023_020924\\rdf.yaml";
		StardistInferJdllOp op = create(modelName, inpTensor);
		RunMode rm = RunMode.createRunMode(op);
		Map<String, Object> aa = rm.runOP();
		System.out.print(false);
	}
	
	/**
	 * Create a StarDist inference OP to execute the whole stardist model in Python
	 * with its pre- and post-processing
	 * @param <T>
	 * 	ImgLib2 data types accepted by the OP
	 * @param modelName
	 * 	name of the model
	 * @param tensor
	 * 	input tensor for stardist
	 * @return the OP can be run to use stardist
	 */
	public static < T extends RealType< T > & NativeType< T > > 
			StardistInferJdllOp create(String modelName, Tensor<T> tensor) {
		StardistInferJdllOp op = new StardistInferJdllOp();
		op.setModel(modelName);
		op.setInputTensor(tensor);
		op.installOp();
		return op;
	}
	
	public void setModel(String modelName) throws IllegalArgumentException {
		Objects.requireNonNull(modelName, "The modelName input argument cannot be null.");
		if (new File(modelName).isFile() && !isModelFileStardist(modelName))
			throw new IllegalArgumentException("The file selected does not correspond to "
					+ "the rdf.yaml file of a Bioiamge.io Stardist model.");
		else if (!(new File(modelName).isFile()) && !isModelNameStardist(modelName))
			throw new IllegalArgumentException("The model name provided does not correspond to a valid"
					+ " Stardist model present in the Bioimage.io online reposritory.");
		this.modelName = modelName;
	}
	
	public < T extends RealType< T > & NativeType< T > > void setInputTensor(Tensor<T> tensor) {
		inputTensor = tensor;
	}

	@Override
	public String getOpPythonFilename() {
		return STARDIST_OP_FNAME;
	}

	@Override
	public int getNumberOfOutputs() {
		return N_STARDIST_OUTPUTS;
	}

	@Override
	public void installOp() {
		// TODO this method checks if the OP file is at its correponding folder.
		// TODO if not unpack the python file and located (where??)
		opFilePath = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\model-runner-java\\python\\ops\\stardist_inference";
		// TODO check if the env has also been created
		// TODO if not create it (where??)
		envPath  = "C:\\Users\\angel\\git\\jep\\miniconda\\envs\\stardist";
	}

	@Override
	public LinkedHashMap<String, Object> getOpInputs() {
		Objects.requireNonNull(modelName, "The model of interest needs to be defined first.");
		Objects.requireNonNull(inputTensor, "The input tensor has not been defined. Please, define"
				+ " it with the method 'setInputTensor(Tensor<T> tensor)'");
		inputsMap = new LinkedHashMap<String, Object>();
		inputsMap.put(MODEL_KEY, modelName);
		inputsMap.put(INPUT_TENSOR_KEY, inputTensor);
		return this.inputsMap;
	}

	@Override
	public String getCondaEnv() {
		return envPath;
	}

	@Override
	public String getMethodName() {
		return OP_METHOD_NAME;
	}

	@Override
	public String getOpDir() {
		return opFilePath;
	}

	@Override
	public boolean isOpInstalled() {
		// TODO maybe remove this method? Make the check at installOp?
		return false;
	}
	
	/**
	 * Check whether a model file or a model name (model name, model ID or model nickname)
	 * correspond to a stardist model.
	 * The model name has to correspond to a stardist model present in the bioimage.io, for 
	 * example: 'chatty-frog'
	 * @param modelName
	 * 	file path or name of a stardist model
	 * @return true if it corresponds to a stardsit model or false otherwise
	 */
	public static boolean isModelCompatible(String modelName) {
		if (modelName == null)
			return false;
		if (new File(modelName).isFile())
			return isModelFileStardist(modelName);
		else {
			return isModelNameStardist(modelName);
		}
	}
	
	/**
	 * Whether the rdf.yaml file corresponding to a model contains
	 * the information to load a stardist model or not.
	 * @param modelFile
	 * 	rdf.yaml file that should correspond to a stardist model
	 * @return true if the rdf.yaml represents a stardist model and false otherwise
	 */
	public static boolean isModelFileStardist(String modelFile) {
		if (new File(modelFile).getName().equals(Constants.RDF_FNAME) == false)
			return false;
		try {
			ModelDescriptor descriptor = ModelDescriptor.readFromLocalFile(modelFile, false);
			 return descriptor.getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY);
		} catch (Exception e) {
			return false;
		}
	}
	
	/**
	 * Whether the model name (in the rdf.yaml any field among: 'name', 'nickname' or 'id')
	 * corresponds to a stardist model of the Bioiamge.io onlie repository or not
	 * @param modelName
	 * 	a String corresponding to any of the following fields of the rdf.yaml file of a stardist
	 * 	model: 'name', 'nickname' or 'id'
	 * @return true if it actually corresponds to astardist model or false otherwise
	 */
	public static boolean isModelNameStardist(String modelName) {
		BioimageioRepo br = BioimageioRepo.connect();
		if (br.selectByName(modelName) != null) {
			return br.selectByName(modelName).getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY);
		} else if (br.selectByID(modelName) != null) {
			return br.selectByID(modelName).getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY);
		} else if (br.selectByNickname(modelName) != null) {
			return br.selectByNickname(modelName).getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY);
		}
		return false;
	}
	
	/**
	 * Returns a list containing all the model names that corresponds to 
	 * StarDist models existing in the Bioimage.io online repository.
	 * @return list of StarDist model names from the Bioimage.io repository
	 */
	public static List<String> fetchStarDistModelNamesFromBioImage() {
		BioimageioRepo br = BioimageioRepo.connect();
		List<String> stardistModels = br.listAllModels(false).values().stream()
				.filter(md -> md.getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY))
				.map(md -> md.getName()).collect(Collectors.toList());
		return stardistModels;
	}
	
	/**
	 * Returns a list containing all the model nicknames that corresponds to 
	 * StarDist models existing in the Bioimage.io online repository.
	 * @return list of StarDist model nicknames from the Bioimage.io repository
	 */
	public static List<String> fetchStarDistModelNicknamesFromBioImage() {
		BioimageioRepo br = BioimageioRepo.connect();
		List<String> stardistModels = br.listAllModels(false).values().stream()
				.filter(md -> md.getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY))
				.map(md -> md.getNickname()).collect(Collectors.toList());
		return stardistModels;
	}
	
	/**
	 * Returns a list containing all the model IDs that corresponds to 
	 * StarDist models existing in the Bioimage.io online repository.
	 * @return list of StarDist model IDs from the Bioimage.io repository
	 */
	public static List<String> fetchStarDistModelIdsFromBioImage() {
		BioimageioRepo br = BioimageioRepo.connect();
		List<String> stardistModels = br.listAllModels(false).values().stream()
				.filter(md -> md.getConfig().getSpecMap().keySet().contains(STARDIST_FIELD_KEY))
				.map(md -> md.getModelID()).collect(Collectors.toList());
		return stardistModels;
	}

}
