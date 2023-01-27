package org.bioimageanalysis.icy.deeplearning.example;

import java.util.ArrayList;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadEngineException;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * This is an example of the library that runs a Deep Learning model on a supported engine locally
 * on your computer.
 * REgard that in order to this example to work, a Deep Learning model needs to be downloaded from the
 * Bioimage.io repo and a Java Deep Learning framework needs to be installed too.
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class ExampleLoadAndRunModel {
	
	public static < T extends RealType< T > & NativeType< T > > void main(String[] args) throws LoadEngineException, Exception {
		// Tag for the DL framework (engine) that wants to be used
		String engine = "onnx";
		// Version of the engine
		String engineVersion = "17";
		// Directory where all the engines are stored
		String enginesDir = "/Users/Cgarcia/git/deep-icy/engines";
		// Path to the model folder
		String modelFolder = "/Users/Cgarcia/git/deep-icy/models/HPA Bestfitting InceptionV3_30102022_133313";
		// Path to the model source. The model source locally is the path to the source file defined in the 
		// yaml inside the model folder
		String modelSource = "/Users/Cgarcia/git/deep-icy/models/HPA Bestfitting InceptionV3_30102022_133313/bestfitting-inceptionv3-single-cell.onnx";
		// Whether the engine is supported by CPu or not
		boolean cpu = true;
		// Whether the engine is supported by GPU or not
		boolean gpu = true;
		// Create the EngineInfo object. It is needed to load the wanted DL framework
		// among all the installed ones. The EngineInfo loads the corresponding engine by looking
		// at the enginesDir at searching for the folder that is named satisfying the characteristics specified.
		// REGARD THAT the engine folders need to follow a naming convention
		EngineInfo engineInfo = createEngineInfo(engine, engineVersion, enginesDir, cpu, gpu);
		// Load the corresponding model
		Model model = loadModel(modelFolder, modelSource, engineInfo);
		// Create an image that will be the backend of the Input Tensor
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		final Img< FloatType > img1 = imgFactory.create( 1, 4, 128, 128);
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("image", "bcyx", img1);
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		inputs.add(inpTensor);
		
		// Create the output tensors defined in the rdf.yaml file witht their corresponding 
		// name and axes and add them to the output list of tensors.
		/// Regard that output tensors can be built empty without allocating memory
		// or allocating memory by craeting the tensor with a sample empty image, or by
		// defining the dimensions and data type
		Tensor<T> outTensor = Tensor.buildEmptyTensor("classes", "bc");
		Tensor<T> outTensor2 = Tensor.buildEmptyTensor("features", "bc");
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
		outputs.add(outTensor);
		outputs.add(outTensor2);
		
		// Run the model on the input tensors. THe output tensors 
		// will be rewritten with the result of the execution
		model.runModel(inputs, outputs);
		System.out.print(false);
		
		// The result is stored in the list of tensors "outputs"
	}
	
	/**
	 * Method that creates the {@link EngineInfo} object.
	 * @param engine
	 * 	tag of the Deep Learning framework as definde in the bioimage.io
	 * @param engineVersion
	 * 	version of the Deep LEarning framework
	 * @param enginesDir
	 * 	directory where all the Deep Learning frameworks are installed
	 * @param cpu
	 * 	whether the engine is supported by CPU or not
	 * @param gpu
	 * 	whether the engine is supported by GPU or not
	 * @return an {@link EngineInfo} object to load a DL model
	 */
	public static EngineInfo createEngineInfo(String engine, String engineVersion, 
			String enginesDir, boolean cpu, boolean gpu) {
		return EngineInfo.defineDLEngine(engine, engineVersion, enginesDir, cpu, gpu);
	}
	
	/**
	 * Load the wanted model
	 * @param modelFolder
	 * 	path to the model folder downloaded
	 * @param modelSource
	 * 	local path to the source file of the model
	 * @param engineInfo
	 * 	Object containing the needed info about the Deep Learning 
	 * 	framework compatible with the wanted model
	 * @return a loaded DL model
	 * @throws LoadEngineException if there is any error loading the model
	 * @throws Exception 
	 */
	public static Model loadModel(String modelFolder, String modelSource, EngineInfo engineInfo) throws LoadEngineException, Exception {
		
		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		return model;
	}
}
