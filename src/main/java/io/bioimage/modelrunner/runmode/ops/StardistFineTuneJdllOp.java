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
import java.util.Objects;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

public class StardistFineTuneJdllOp implements OpInterface {
	
	private String model;
	
	private String nModelPath;
	
	private float lr = (float) 1e-5;
	
	private int batchSize = 16;
	
	private Tensor<FloatType> trainingSamples;
	
	private Tensor<UnsignedShortType> groundTruth;
	
	private String opFilePath;
	
	private String envPath;
	
	private LinkedHashMap<String, Object> inputsMap;
	
	private final static String MODEL_KEY = "model";
	
	private final static String NEW_MODEL_DIR_KEY = "n_model_dir";
	
	private final static String TRAIN_SAMPLES_KEY = "train_samples";
	
	private final static String GROUND_TRUTH_KEY = "ground_truth";
	
	private final static String BATCH_SIZE_KEY = "batch_size";
	
	private final static String LR_KEY = "learning_rate";
	
	private static final String OP_METHOD_NAME = "stardist_prediction_2d_mine";
	
	private static final int N_STARDIST_OUTPUTS = 1;
	
	private static final String STARDIST_OP_FNAME = "stardist_inference.py";
	
	/**
	 * Create a JDLL OP to fine tune a stardist model with the wanted data.
	 * In order to set the data we want to fine tune the model on, use {@link #setFineTuningData(List, List)}
	 * or {@link #setFineTuningData(Tensor, Tensor)}. The batch size and learning rates
	 * can also be modified by with {@link #setBatchSize(int)} and {@link #setLearingRate(float)}.
	 * By default the batch size is 16 and the learning rate 1e-5.
	 * @param modelToFineTune
	 * 	Pre-trained model that is going to be fine tuned on the user's data, it
	 *  can be either a model existing in the users machine or a model existing in the model
	 *  zoo. If it is a model existing in th emodel zoo, it will have to be downloaded first.
	 * @param newModelDir
	 * 	directory where the new model will be saved
	 * @return a JDLL OP that can be used together with {@link RunMode} to fine tune a StarDist
	 * 	model on the user's data
	 */
	public StardistFineTuneJdllOp create(String modelToFineTune, String newModelDir) {
		StardistFineTuneJdllOp op = new StardistFineTuneJdllOp();
		op.setModel(modelToFineTune);
		op.nModelPath = newModelDir;
		return op;
	}
	
	public < T extends RealType< T > & NativeType< T > > 
		void setFineTuningData(List<Tensor<T>> trainingSamples, List<Tensor<T>> groundTruth) {
		
	}
	
	public < T extends RealType< T > & NativeType< T > > 
		void setFineTuningData(Tensor<T> trainingSamples, Tensor<T> groundTruth) {
		checkTrainAndGroundTruthDimensions(trainingSamples, groundTruth);
		setTrainingSamples(trainingSamples);
		setGroundTruth(groundTruth);
	}
	
	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}
	
	public void setLearingRate(float learningRate) {
		this.lr = learningRate;
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
	public boolean isOpInstalled() {
		// TODO Auto-generated method stub
		return true;
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
		inputsMap = new LinkedHashMap<String, Object>();
		inputsMap.put(MODEL_KEY, this.model);
		inputsMap.put(NEW_MODEL_DIR_KEY, this.nModelPath);
		inputsMap.put(TRAIN_SAMPLES_KEY, this.trainingSamples);
		inputsMap.put(GROUND_TRUTH_KEY, this.groundTruth);
		inputsMap.put(BATCH_SIZE_KEY, this.batchSize);
		inputsMap.put(LR_KEY, this.lr);
		return inputsMap;
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
	
	public void setModel(String modelName) throws IllegalArgumentException {
		Objects.requireNonNull(modelName, "The modelName input argument cannot be null.");
		if (new File(modelName).isFile() && !StardistInferJdllOp.isModelFileStardist(modelName))
			throw new IllegalArgumentException("The file selected does not correspond to "
					+ "the rdf.yaml file of a Bioiamge.io Stardist model.");
		else if (!(new File(modelName).isFile()) && !StardistInferJdllOp.isModelNameStardist(modelName))
			throw new IllegalArgumentException("The model name provided does not correspond to a valid"
					+ " Stardist model present in the Bioimage.io online reposritory.");
		this.model = modelName;
	}
	
	private < T extends RealType< T > & NativeType< T > > 
	 void checkTrainAndGroundTruthDimensions(Tensor<T> trainingSamples, Tensor<T> groundTruth) {
		
	}
	
	private < T extends RealType< T > & NativeType< T > > 
	 void setTrainingSamples(Tensor<T> trainingSamples) {
    	if (!(Util.getTypeFromInterval(trainingSamples.getData()) instanceof FloatType)) {
    		this.trainingSamples = Tensor.createCopyOfTensorInWantedDataType(trainingSamples, new FloatType());
    	} else {
    		this.trainingSamples = (Tensor<FloatType>) trainingSamples;
    	}
	}
	
	private < T extends RealType< T > & NativeType< T > > 
	 void setGroundTruth(Tensor<T> groundTruth) {
    	if (!(Util.getTypeFromInterval(groundTruth.getData()) instanceof UnsignedShortType)) {
    		this.groundTruth = Tensor.createCopyOfTensorInWantedDataType(groundTruth, new UnsignedShortType());
    	} else {
    		this.groundTruth = (Tensor<UnsignedShortType>) groundTruth;
    	}
	}

}
