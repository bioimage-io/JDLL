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
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.engine.installation.FileDownloader;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.FileUtils;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * TODO
 * TODO
 * TODO
 * TODO
 * TODO add support for stardist 3D
 * 
 * 
 * 
 * Class that defines the methods needed to fine tune a StarDist pre-trained model
 * using JDLL and Python with Appose.
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class StardistFineTuneJdllOp implements OpInterface {
	
	private String model;
	
	private String nModelParentPath;
	
	private String weightsToFineTune;
	
	private int nChannelsModel;
	
	private float lr = (float) 1e-5;
	
	private int batchSize = 16;
	
	private int epochs = 1;
	
	private boolean downloadStardistPretrained = false;
	
	private Tensor<FloatType> trainingSamples;
	
	private Tensor<UnsignedShortType> groundTruth;
	
	private String opFilePath;
	
	private String envPath;
	
	private LinkedHashMap<String, Object> inputsMap;
	
	private static final Map<String, String> PRETRAINED_3C_STARDIST_MODELS;
	static {
		PRETRAINED_3C_STARDIST_MODELS = new HashMap<String, String>();
		PRETRAINED_3C_STARDIST_MODELS.put("2D_versatile_fluo", "fearless-crab");
		PRETRAINED_3C_STARDIST_MODELS.put("2D_paper_dsb2018", null);
	}
	
	private static final Map<String, String> PRETRAINED_1C_STARDIST_MODELS;
	static {
		PRETRAINED_1C_STARDIST_MODELS = new HashMap<String, String>();
		PRETRAINED_1C_STARDIST_MODELS.put("2D_versatile_he", "chatty-frog");
	}
	
	private final static String STARDIST_CONFIG_KEY = "config";
	
	private final static String CONFIG_JSON = "config.json";
	
	private final static String STARDIST_THRES_KEY = "thresholds";
	
	private final static String THRES_JSON = "thresholds.json";
	
	private final static String MODEL_KEY = "model";
	
	private final static String NEW_MODEL_DIR_KEY = "n_model_dir";
	
	private final static String TRAIN_SAMPLES_KEY = "train_samples";
	
	private final static String GROUND_TRUTH_KEY = "ground_truth";
	
	private final static String WEIGHTS_TO_FINE_TUNE_KEY = "weights_file";
	
	private final static String PATCH_SIZE_KEY = "train_patch_size";
	
	private final static String BATCH_SIZE_KEY = "train_batch_size";
	
	private final static String LR_KEY = "train_learning_rate";
	
	private final static String EPOCHS_KEY = "train_epochs";
	
	private static final String STARDIST_WEIGHTS_FILE = "stardist_weights.h5";
	
	private static final String KERAS_SUFFIX_FILE = ".h5";
	
	private final static String DOWNLOAD_STARDIST_KEY = "download_pretrained_stardist";
	
	private static final String OP_METHOD_NAME = "finetune_stardist";
	
	private static final int N_STARDIST_OUTPUTS = 1;
	
	private static final String STARDIST_OP_FNAME = "stardist_fine_tune.py";
	
	private static final String STARDIST_2D_AXES = "byxc";
	
	private static final String STARDIST_3D_AXES = "bzyxc";
	
	private static final String GROUNDTRUTH_AXES = "byx";
	

	public static void main(String[] args) throws IOException, InterruptedException, Exception {
		final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
		final Img< FloatType > img1 = imgFactory.create( 2, 64, 64, 3 );
		Tensor<FloatType> inpTensor = Tensor.build("input0", "byxc", img1);
		final ImgFactory< FloatType > gtFactory = new ArrayImgFactory<>( new FloatType() );
		final Img< FloatType > gt = gtFactory.create( 2, 64, 64 );
		Tensor<FloatType> gtTensor = Tensor.build("gt", "byx", gt);
		String modelName = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\model-runner-java\\models";
		String p = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\model-runner-java\\models\\finetuned_StarDist H&E Nuclei Segmentation_04102023_123644";
		//p = "chatty-frog";
		//StardistFineTuneJdllOp op = finetuneAndCreateNew(p, modelName);
		StardistFineTuneJdllOp op = finetuneInPlace(p);
		op.installOp();
		op.setBatchSize(2);
		op.setEpochs(1);
		op.setFineTuningData(inpTensor, gtTensor);
		op.setLearingRate((float) 1e-5);
		RunMode rm = RunMode.createRunMode(op);
		Map<String, Object> aa = rm.runOP();
		System.out.print(false);
	}
	
	/**
	 * Create a JDLL OP to fine tune a stardist model with the wanted data.
	 * In order to set the data we want to fine tune the model on, use {@link #setFineTuningData(List, List)}
	 * or {@link #setFineTuningData(Tensor, Tensor)}. The batch size and learning rates
	 * can also be modified by with {@link #setBatchSize(int)} and {@link #setLearingRate(float)}.
	 * By default the batch size is 16 and the learning rate 1e-5.
	 * To set the number of epochs: {@link #setEpochs(int)}, default is 1.
	 * 
	 * Finally in some cases, if we want to fine tune a local model, the model might have 
	 * several weight files:
	 * - stardist
	 * 	- config.json
	 * 	- thresholds.json
	 * 	- stardist_weights.h5
	 * 	- weights_best.h5
	 * 	- weights_last.h5
	 * 
	 * If we are interested in fine tuning a stardist model starting from a certain weights file among
	 * the available ones, use the {@link #setWeightsToFineTune(String)} to provide the name of the
	 * file of interest. 
	 * If no specific weight file is designated through the {@link #setWeightsToFineTune(String)} method, 
	 * the system will automatically select and load a weights file containing the substring 'best', if available.
	 * 
	 * If there is no weights file containing the substring 'best', the system will automatically select and 
	 * load the file that comes first in alphabetical order from the available options. 
	 * 
	 * If there is only one weights file available, it is not necessary to 
	 * use {@link #setWeightsToFineTune(String)}. 
	 * 
	 * NOTE THAT weights_best.h5 AND weights_last.h5 WILL ALWAYS BE REWRITTEN AFTER EACH FINE TUNE ITERATION
	 * 
	 * @param modelToFineTune
	 * 	Pre-trained model that is going to be fine tuned on the user's data, it
	 *  can be either a model existing in the users machine or a model existing in the model
	 *  zoo. If it is a model existing in the model zoo, it will have to be downloaded first.
	 * @param newModelDir
	 * 	directory where the new finetuned model folder will be saved.
	 * @return a JDLL OP that can be used together with {@link RunMode} to fine tune a StarDist
	 * 	model on the user's data
	 * @throws InterruptedException 
	 * @throws IOException 
	 * @throws Exception 
	 */
	public static StardistFineTuneJdllOp finetuneInPlace(String modelToFineTune, String newModelDir) throws IOException, InterruptedException, Exception {
		Objects.requireNonNull(modelToFineTune, "modelToFineTune' cannot be null. It should correspond to either a Bioimage.io "
				+ "folder containing a StarDist model, the nickname of a StarDist model in the Bioimage.io (example: chatty-frog) "
				+ "or to one if the StarDist pre-trained available weigths (example: 2D_versatile_fluo)");
		Objects.requireNonNull(newModelDir,  "newModelDir' cannot be null. It should be a path to the directory where"
				+ "	the we want the fine tuned model to be saved.");
		if (new File(newModelDir).isDirectory() == false)
			throw new IllegalArgumentException("Argument 'newModelDir' should be an existing directory. In that "
					+ "directory the fine tuned StarDist model is going to be created.");
		StardistFineTuneJdllOp op = new StardistFineTuneJdllOp();
		op.nModelParentPath = newModelDir;
		op.model = modelToFineTune;
		op.setModel();
		try {
			op.findNChannels();
		} catch (Exception e) {
			throw new IllegalArgumentException("Unable to correctly read the rdf.yaml file "
					+ "of Bioimage.io StarDist model at :" + new File(op.model).getParent(), e);
		}
		return op;
	}
	
	/**
	 * Create a JDLL OP to fine tune a stardist model with the wanted data inplace.
	 * The model created overwrites the weights of the pre-trained model used for finetuning.
	 * In order to set the data we want to fine tune the model on, use {@link #setFineTuningData(List, List)}
	 * or {@link #setFineTuningData(Tensor, Tensor)}. The batch size and learning rates
	 * can also be modified by with {@link #setBatchSize(int)} and {@link #setLearingRate(float)}.
	 * By default the batch size is 16 and the learning rate 1e-5.
	 * To set the number of epochs: {@link #setEpochs(int)}, default is 1.
	 * 
	 * Finally in some cases, if we want to fine tune a local model, the model might have 
	 * several weight files:
	 * - stardist
	 * 	- config.json
	 * 	- thresholds.json
	 * 	- stardist_weights.h5
	 * 	- weights_best.h5
	 * 	- weights_last.h5
	 * 
	 * If we are interested in fine tuning a stardist model starting from a certain weights file among
	 * the available ones, use the {@link #setWeightsToFineTune(String)} to provide the name of the
	 * file of interest. 
	 * If no specific weight file is designated through the {@link #setWeightsToFineTune(String)} method, 
	 * the system will automatically select and load a weights file containing the substring 'best', if available.
	 * 
	 * If there is no weights file containing the substring 'best', the system will automatically select and 
	 * load the file that comes first in alphabetical order from the available options. 
	 * 
	 * If there is only one weights file available, it is not necessary to 
	 * use {@link #setWeightsToFineTune(String)}. 
	 * 
	 * NOTE THAT weights_best.h5 AND weights_last.h5 WILL ALWAYS BE REWRITTEN AFTER EACH FINE TUNE ITERATION
	 * 
	 * @param modelToFineTune
	 * 	Pre-trained model that is going to be fine tuned on the user's data, it
	 *  can be either a model existing in the users machine or a model existing in the model
	 *  zoo. If it is a model existing in the model zoo, it will have to be downloaded first.
	 * @return a JDLL OP that can be used together with {@link RunMode} to fine tune a StarDist
	 * 	model on the user's data
	 * @throws InterruptedException 
	 * @throws IOException 
	 * @throws Exception 
	 */
	public static StardistFineTuneJdllOp finetuneInPlace(String modelToFineTune) throws IOException, InterruptedException, Exception {
		Objects.requireNonNull(modelToFineTune, "modelToFineTune' cannot be null. It should correspond to either a Bioimage.io "
				+ "folder containing a StarDist model, the nickname of a StarDist model in the Bioimage.io (example: chatty-frog) "
				+ "or to one if the StarDist pre-trained available weigths (example: 2D_versatile_fluo)");
		if (new File(modelToFineTune).isDirectory() == false)
			throw new IllegalArgumentException("Argument 'modelToFineTune' should be an existing directory. "
					+ "That directory should contain the model that wants to be fine-tuned and overwritten.");
		StardistFineTuneJdllOp op = new StardistFineTuneJdllOp();
		op.model = modelToFineTune;
		op.setModel();
		try {
			op.findNChannels();
		} catch (Exception e) {
			throw new IllegalArgumentException("Unable to correctly read the rdf.yaml file "
					+ "of Bioimage.io StarDist model at :" + new File(op.model).getParent(), e);
		}
		return op;
	}
	
	/**
	 * This method should only be used when we try to fine tune models from local directories.
	 * 
	 * In some cases, if we want to fine tune a local model, the model might have 
	 * several weight files:
	 * - stardist
	 * 	- config.json
	 * 	- thresholds.json
	 * 	- stardist_weights.h5
	 * 	- weights_best.h5
	 * 	- weights_last.h5
	 * 
	 * If we are interested in fine tuning a stardist model starting from a certain weights file among
	 * the available ones, use the {@link #setWeightsToFineTune(String)} to provide the name of the
	 * file of interest. 
	 * If no specific weight file is designated through the {@link #setWeightsToFineTune(String)} method, 
	 * the system will automatically select and load a weights file containing the substring 'best', if available.
	 * 
	 * If there is no weights file containing the substring 'best', the system will automatically select and 
	 * load the file that comes first in alphabetical order from the available options. 
	 * 
	 * NOTE THAT weights_best.h5 AND weights_last.h5 WILL ALWAYS BE REWRITTEN AFTER EACH FINE TUNE ITERATION
	 * 
	 * If there is only one weights file available, it is not necessary to 
	 * use {@link #setWeightsToFineTune(String)}. 
	 * @param weigthsToFineTune
	 * 	name of the weights file we want to load. It must end in .h5 and be available in the stardist model folder
	 */
	public void setWeightsToFineTune(String weigthsToFineTune) {
		this.weightsToFineTune = weigthsToFineTune;
	}
	
	public < T extends RealType< T > & NativeType< T > > 
		void setFineTuningData(List<Tensor<T>> trainingSamples, List<Tensor<T>> groundTruth) {
		
	}
	
	public < T extends RealType< T > & NativeType< T > > 
		void setFineTuningData(Tensor<T> trainingSamples, Tensor<T> groundTruth) throws IOException, Exception {
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
	
	public void setEpochs(int epochs) {
		this.epochs = epochs;
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
		opFilePath = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\model-runner-java\\python\\ops\\\\stardist_fine_tune";
		// TODO check if the env has also been created
		// TODO if not create it (where??)
		envPath  = "C:\\Users\\angel\\git\\jep\\miniconda\\envs\\stardist";
	}

	@Override
	public LinkedHashMap<String, Object> getOpInputs() throws Exception {
		inputsMap = new LinkedHashMap<String, Object>();
		Objects.requireNonNull(trainingSamples, "Please make sure that the training samples have "
				+ "been provided and that they are not null.Use the method: "
				+ "setFineTuningData(Tensor<T> trainingSamples, Tensor<T> groundTruth)");
		inputsMap.put(MODEL_KEY, this.model);
		Objects.requireNonNull(groundTruth, "Please make sure that the ground truth has "
				+ "been provided and that it is not null.Use the method: "
				+ "setFineTuningData(Tensor<T> trainingSamples, Tensor<T> groundTruth)");
		inputsMap.put(TRAIN_SAMPLES_KEY, this.trainingSamples);
		inputsMap.put(GROUND_TRUTH_KEY, this.groundTruth);
		// TODO remove inputsMap.put(DOWNLOAD_STARDIST_KEY, this.downloadStardistPretrained);
		setUpConfigs();
		if (this.weightsToFineTune != null)
			setWeigthsFile();
		return inputsMap;
	}
	
	private void setWeigthsFile() {
		if (!weightsToFineTune.endsWith(KERAS_SUFFIX_FILE))
			throw new IllegalArgumentException("StarDist weigths files must always end with '" 
						+ KERAS_SUFFIX_FILE + "' and the provided file does not: " + this.weightsToFineTune);
		if (new File(weightsToFineTune).isFile() && !(new File(weightsToFineTune).getParent().equals(model)))
			throw new IllegalArgumentException("StarDist weigths files that can be fine tuned with this model"
					+ "should be in the folder: " + this.model);
		if (!(new File(weightsToFineTune).isFile()) && new File(model, weightsToFineTune).isFile())
			throw new IllegalArgumentException("The StarDist weigths file provided (" + this.weightsToFineTune
					+ ") cannot be found in the StarDist model folder : " + this.model);
		inputsMap.put(WEIGHTS_TO_FINE_TUNE_KEY, this.weightsToFineTune);
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
	
	public void setModel() throws IOException, InterruptedException, Exception {
		Objects.requireNonNull(model, "The modelName input argument cannot be null.");
		if (PRETRAINED_1C_STARDIST_MODELS.keySet().contains(model) 
				|| PRETRAINED_3C_STARDIST_MODELS.keySet().contains(model)) {
			this.downloadStardistPretrained = true;
			setUpStardistModelFromStardistRepo();
			return;
		}
		if (new File(model).isDirectory() && !(new File(model, Constants.RDF_FNAME).isFile()))
			throw new IllegalArgumentException("The directory selected does not correspond to "
					+ "a valid Bioimage.io model, it does not contain the required specs file: " + Constants.RDF_FNAME);
		else if (new File(model).isDirectory()&& !StardistInferJdllOp.isModelFileStardist(model + File.separator + Constants.RDF_FNAME))
			throw new IllegalArgumentException("The directory selected does not correspond to "
					+ "a Bioimage.io StarDist model, as per its specs file: " + Constants.RDF_FNAME);
		else if (new File(model).isDirectory())
			setUpStardistModelFromLocal();
		else if (!(new File(model).isDirectory()) && !StardistInferJdllOp.isModelNameStardist(model))
			throw new IllegalArgumentException("The model name provided does not correspond to a valid"
					+ " Stardist model present in the Bioimage.io online reposritory.");
		else if (!(new File(model).isDirectory()))
			setUpStardistModelFromBioimageio();
		else
			throw new IllegalArgumentException("Cannot recognise the model provided as a StarDist model. "
					+ "You can provide either the name of a StarDist model in the Bioimage.io, the path"
					+ " to a Bioimage.io StarDist model (parent dir of the rdf.yaml file) or the name of"
					+ " a pre-trained StarDist model.");
	}
	
	private void setUpStardistModelFromStardistRepo() throws IOException, InterruptedException, Exception {
		if (PRETRAINED_1C_STARDIST_MODELS.get(model) != null) {
			model = PRETRAINED_1C_STARDIST_MODELS.get(model);
			setUpStardistModelFromBioimageio();
		} else if (PRETRAINED_3C_STARDIST_MODELS.get(model) != null) {
			model = PRETRAINED_3C_STARDIST_MODELS.get(model);
			setUpStardistModelFromBioimageio();
		} else {
			// TODO what to do with 2D_paper_dsb2018 and DEMO model (they dont have model card)
			// TODO what to do with 2D_paper_dsb2018 and DEMO model (they dont have model card)
			// TODO what to do with 2D_paper_dsb2018 and DEMO model (they dont have model card)
			// TODO what to do with 2D_paper_dsb2018 and DEMO model (they dont have model card)
			// TODO what to do with 2D_paper_dsb2018 and DEMO model (they dont have model card)
			// TODO what to do with 2D_paper_dsb2018 and DEMO model (they dont have model card)
		}
	}
	
	private void setUpStardistModelFromBioimageio() throws IOException, InterruptedException, Exception {
		BioimageioRepo br = BioimageioRepo.connect();
		if (br.selectByName(model) != null) {
			model = br.downloadByName(model, nModelParentPath);
		} else if (br.selectByID(model) != null) {
			model = br.downloadModelByID(model, nModelParentPath);
		} else if (br.selectByNickname(model) != null) {
			model = br.downloadByNickame(model, nModelParentPath);
		}
		File folder = new File(model);
		String fineTuned = folder.getParent() + File.separator + "finetuned_" + folder.getName();
        File renamedFolder = new File(fineTuned);
        if (folder.renameTo(renamedFolder))
        	model = fineTuned;
        downloadBioimageioStardistWeights();
	}
	
	private void downloadBioimageioStardistWeights() throws IllegalArgumentException,
															IOException, Exception {
		
		File stardistSubfolder = new File(this.model, StardistInferJdllOp.STARDIST_FIELD_KEY);
        if (!stardistSubfolder.exists()) {
            if (!stardistSubfolder.mkdirs()) {
            	throw new IOException("Unable to create folder named 'stardist' at: " + this.model);
            }
        }
		setUpKerasWeights();
	}
	
	private void setUpConfigs() throws IOException, Exception {
		String rdfDir = new File(model).getParent();
		if (new File(rdfDir + File.separator + Constants.RDF_FNAME).exists()) {
			setUpConfigsBioimageio();
		} else if (!(new File(model + File.separator + CONFIG_JSON).exists())) {
			throw new IOException("Missing necessary file for StarDist: " + CONFIG_JSON);
		} else if (!(new File(model + File.separator + THRES_JSON).exists())) {
			throw new IOException("Missing necessary file for StarDist: " + THRES_JSON);
		} else {
			Map<String, Object> config = JSONUtils.load(model + File.separator + CONFIG_JSON);
			int w = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("x")];
			int h = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("y")];
			config.put(PATCH_SIZE_KEY, new int[] {w, h});
			config.put(BATCH_SIZE_KEY, this.batchSize);
			config.put(LR_KEY, this.lr);
			config.put(EPOCHS_KEY, this.epochs);
			JSONUtils.writeJSONFile(model + File.separator + CONFIG_JSON, (Map<String, Object>) config);
		}
	}
	
	@SuppressWarnings("unchecked")
	private void setUpConfigsBioimageio() throws IOException, Exception {
		String rdfDir = new File(model).getParent();
		ModelDescriptor descriptor = ModelDescriptor.readFromLocalFile(rdfDir + File.separator + Constants.RDF_FNAME, false);
		Object stardistInfo = descriptor.getConfig().getSpecMap().get(StardistInferJdllOp.STARDIST_FIELD_KEY);
		
		if (stardistInfo == null || !(stardistInfo instanceof Map)) {
			throw new IllegalArgumentException("The rdf.yaml file of the Bioimage.io StarDist "
					+ "model at: " + this.model + " is invalid. The field config>stardist is missing."
					+ " Look for StarDist models in the Bioimage.io repo to see how the rdf.yaml should look like.");
		}
		Object config = ((Map<String, Object>) stardistInfo).get(STARDIST_CONFIG_KEY);
		if (config == null || !(config instanceof Map)) {
			throw new IllegalArgumentException("The rdf.yaml file of the Bioimage.io StarDist "
					+ "model at: " + this.model + " is invalid. The field config>stardist>" + STARDIST_CONFIG_KEY + " is missing."
					+ " Look for StarDist models in the Bioimage.io repo to see how the rdf.yaml should look like.");
		}
		Object thres = ((Map<String, Object>) stardistInfo).get(STARDIST_THRES_KEY);
		if (thres == null || !(thres instanceof Map)) {
			throw new IllegalArgumentException("The rdf.yaml file of the Bioimage.io StarDist "
					+ "model at: " + this.model + " is invalid. The field config>stardist>" + STARDIST_THRES_KEY + " is missing."
					+ " Look for StarDist models in the Bioimage.io repo to see how the rdf.yaml should look like.");
		}
		int w = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("x")];
		int h = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("y")];
		((Map<String, Object>) config).put(PATCH_SIZE_KEY, new int[] {w, h});
		((Map<String, Object>) config).put(BATCH_SIZE_KEY, this.batchSize);
		((Map<String, Object>) config).put(LR_KEY, this.lr);
		((Map<String, Object>) config).put(EPOCHS_KEY, this.epochs);
		JSONUtils.writeJSONFile(model + File.separator + CONFIG_JSON, (Map<String, Object>) config);
		JSONUtils.writeJSONFile(model + File.separator + THRES_JSON, (Map<String, Object>) thres);
	}
	
	private void setUpKerasWeights() throws IOException, Exception {
		String rdfYamlFN = this.model + File.separator + Constants.RDF_FNAME;
		ModelDescriptor descriptor = ModelDescriptor.readFromLocalFile(rdfYamlFN, false);
		String stardistWeights = this.model + File.separator +  StardistInferJdllOp.STARDIST_FIELD_KEY;
		stardistWeights += File.separator + STARDIST_WEIGHTS_FILE;
		String stardistWeightsParent = this.model + File.separator + STARDIST_WEIGHTS_FILE;
		model = model + File.separator +  StardistInferJdllOp.STARDIST_FIELD_KEY;
		if (new File(stardistWeights).exists()) 
			return;
		if (new File(stardistWeights).exists()) {
			try {
	            Files.copy(Paths.get(stardistWeightsParent), Paths.get(stardistWeights), StandardCopyOption.REPLACE_EXISTING);
				return;
	        } catch (IOException e) {
	        }
		}
		downloadFileFromInternet(getKerasWeigthsLink(descriptor), new File(stardistWeights));
	}
	
	@SuppressWarnings("unchecked")
	private static String getKerasWeigthsLink(ModelDescriptor descriptor) throws IOException {
		Object yamlFiles = descriptor.getAttachments().get("files");
		if (yamlFiles == null || !(yamlFiles instanceof List))
			throw new IllegalArgumentException("");
		for (String url : (List<String>) yamlFiles) {
			try {
				if (DownloadModel.getFileNameFromURLString(url).equals(STARDIST_WEIGHTS_FILE))
					return url;
			} catch (MalformedURLException e) {
			}
		}
		throw new IOException("Stardist rdf.yaml file at : " + descriptor.getModelPath()
				+ " is invalid, as it does not contain the URL to StarDist Keras weights in "
				+ "the attachements field. Look for a StarDist model on the Bioimage.io "
				+ "repository for an example of a correct version.");
	}
	
	private void setUpStardistModelFromLocal() throws IllegalArgumentException, IOException, Exception {
		if (this.nModelParentPath == null) {
			File folder = new File(model);
			String fineTuned = folder.getParent() + File.separator + "finetuned_" + folder.getName();
			String fineTunedAux = "" + fineTuned;
			int c = 1;
			while (new File(fineTuned).exists())
				fineTuned = fineTunedAux + "-" + (c++);
	        if (folder.renameTo(new File(fineTuned)))
	        	model = fineTuned;
		} else {
			File folder = new File(model);
			String fineTuned = nModelParentPath + File.separator + "finetuned_" + folder.getName();
			String fineTunedAux = "" + fineTuned;
			int c = 1;
			while (new File(fineTuned).exists())
				fineTuned = fineTunedAux + "-" + (c++);
			if (!new File(fineTuned).mkdirs())
				throw new IOException("Unable to create directory for fine tuned model at: " + fineTuned); 
            Files.copy(Paths.get(model, Constants.RDF_FNAME), Paths.get(fineTuned, Constants.RDF_FNAME), StandardCopyOption.REPLACE_EXISTING);
            if (new File(model + File.separator + StardistInferJdllOp.STARDIST_FIELD_KEY).isDirectory()) {
				try {
					FileUtils.copyFolder(Paths.get(model, StardistInferJdllOp.STARDIST_FIELD_KEY), Paths.get(fineTuned, StardistInferJdllOp.STARDIST_FIELD_KEY));
		        } catch (IOException e) {
		        }
			}
            model = fineTuned;
		}
        downloadBioimageioStardistWeights();
	}
	
	private < T extends RealType< T > & NativeType< T > > 
	 void checkTrainAndGroundTruthDimensions(Tensor<T> trainingSamples, Tensor<T> groundTruth) {
		checkTrainingSamplesTensorDimsForStardist(trainingSamples);
		checkGroundTruthTensorDimsForStardist(groundTruth);

		int trW = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("x")];
		int trH = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("y")];
		int trB = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("b")];
		int gtW = groundTruth.getShape()[groundTruth.getAxesOrderString().indexOf("x")];
		int gtH = groundTruth.getShape()[groundTruth.getAxesOrderString().indexOf("y")];
		int gtB = groundTruth.getShape()[groundTruth.getAxesOrderString().indexOf("b")];

		if (gtW != trW)
			throw new IllegalArgumentException("Training samples (" + trW + ") and ground truth ("
					+ gtW + ") width (x-axis) must be the same.");
		if (trH != gtH)
			throw new IllegalArgumentException("Training samples (" + trH + ") and ground truth (" 
					+ gtH + ") height (y-axis) must be the same.");
		if (trB != gtB)
			throw new IllegalArgumentException("Training samples (" + trB + ") and ground truth (" 
					+ gtB + ") batch size (b-axis) must be the same.");
		 
		
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
	void checkTrainingSamplesTensorDimsForStardist(Tensor<T> trainingSamples) {
		String axes = trainingSamples.getAxesOrderString();
		String stardistAxes = STARDIST_2D_AXES;
		if (axes.length() == 5)
			stardistAxes = STARDIST_3D_AXES;
		else if (axes.length() != 5 && axes.length() != 4)
			throw new IllegalArgumentException("Training input tensors should have 4 dimensions ("
					+ STARDIST_2D_AXES + ") or 5 (" + STARDIST_3D_AXES + "), but it has " + axes.length() + " (" + axes + ").");
		
		checkDimOrderAndTranspose(trainingSamples, stardistAxes, "training input");
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
	void checkGroundTruthTensorDimsForStardist(Tensor<T> gt) {
		String axes = gt.getAxesOrderString();
		String stardistAxes = GROUNDTRUTH_AXES;
		if (axes.length() != GROUNDTRUTH_AXES.length())
			throw new IllegalArgumentException("Ground truth tensors should have 3 dimensions ("
					+ GROUNDTRUTH_AXES + "), but it has " + axes.length() + " (" + axes + ").");
		
		checkDimOrderAndTranspose(gt, stardistAxes, "ground truth");
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
	void checkDimOrderAndTranspose(Tensor<T> tensor, String stardistAxes, String errMsgObject) {
		for (int c = 0; c < stardistAxes.length(); c ++) {
			String axes = tensor.getAxesOrderString();
			int trueInd = axes.indexOf(stardistAxes.split("")[c]);
			if (trueInd == -1)
				throw new IllegalArgumentException("The " + errMsgObject + " tensors provided should have dimension '"
						+ stardistAxes.split("")[c] + "' in the axes order, but it does not (" + axes + ").");
			else if (trueInd == c)
				continue;
			IntervalView<T> wrapImg = Views.permute(tensor.getData(), trueInd, c);
			StringBuilder nAxes = new StringBuilder(axes);
			nAxes.setCharAt(c, stardistAxes.charAt(c));
			nAxes.setCharAt(trueInd, axes.charAt(c));
			tensor = Tensor.build(tensor.getName(), nAxes.toString(), wrapImg);
			c = 0;
		}
	}
	
	@SuppressWarnings("unchecked")
	private < T extends RealType< T > & NativeType< T > > 
	 void setTrainingSamples(Tensor<T> trainingSamples) {
		int tensorChannels = trainingSamples.getShape()[trainingSamples.getAxesOrderString().indexOf("c")];
		if (nChannelsModel != tensorChannels)
			throw new IllegalArgumentException("The pre-trained selected model only supports " 
					+ nChannelsModel + "-channel inputs whereas the provided training input tensor "
							+ "has " + tensorChannels + " channels.");
		if (!(Util.getTypeFromInterval(trainingSamples.getData()) instanceof FloatType)) {
    		this.trainingSamples = Tensor.createCopyOfTensorInWantedDataType(trainingSamples, new FloatType());
    	} else {
    		this.trainingSamples = Cast.unchecked(trainingSamples);
    	}
	}
	
	@SuppressWarnings("unchecked")
	private < T extends RealType< T > & NativeType< T > > 
	 void setGroundTruth(Tensor<T> groundTruth) {
    	if (!(Util.getTypeFromInterval(groundTruth.getData()) instanceof UnsignedShortType)) {
    		this.groundTruth = Tensor.createCopyOfTensorInWantedDataType(groundTruth, new UnsignedShortType());
    	} else {
    		this.groundTruth = Cast.unchecked(groundTruth);
    	}
	}
	
	private void findNChannels() throws Exception {
		if (this.downloadStardistPretrained && PRETRAINED_1C_STARDIST_MODELS.keySet().contains(this.model)) {
			this.nChannelsModel = 1;
		} else if (this.downloadStardistPretrained && PRETRAINED_3C_STARDIST_MODELS.keySet().contains(this.model)) {
			this.nChannelsModel = 3;
		}
		String rdfFileName = new File(model).getParentFile() + File.separator + Constants.RDF_FNAME;
		ModelDescriptor descriptor = ModelDescriptor.readFromLocalFile(rdfFileName, false);
		int cInd = descriptor.getInputTensors().get(0).getAxesOrder().indexOf("c");
		nChannelsModel = descriptor.getInputTensors().get(0).getShape().getTileMinimumSize()[cInd];
	}
	
	/**
	 * Method that downloads the model selected from the internet,
	 * copies it and unzips it into the models folder
	 * @param downloadURL
	 * 	url of the file to be downloaded
	 * @param targetFile
	 * 	file where the file from the url will be downloaded too
	 */
	public static void downloadFileFromInternet(String downloadURL, File targetFile) {
		FileOutputStream fos = null;
		ReadableByteChannel rbc = null;
		try {
			URL website = new URL(downloadURL);
			rbc = Channels.newChannel(website.openStream());
			// Create the new model file as a zip
			fos = new FileOutputStream(targetFile);
			// Send the correct parameters to the progress screen
			FileDownloader downloader = new FileDownloader(rbc, fos);
			downloader.call();
		} catch (IOException e) {
			String msg = "The link for the file: " + targetFile.getName() + " is broken.";
			new IOException(msg, e).printStackTrace();
		} finally {
			try {
				if (fos != null)
						fos.close();
				if (rbc != null)
					rbc.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
