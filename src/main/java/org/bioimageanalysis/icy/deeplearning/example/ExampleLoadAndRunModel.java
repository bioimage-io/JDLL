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
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class ExampleLoadAndRunModel {
	
	public static < T extends RealType< T > & NativeType< T > > void main(String[] args) throws LoadEngineException, Exception {
		String engine = "onnx";
		String engineVersion = "17";
		String enginesDir = "/Users/Cgarcia/git/deep-icy/engines";
		String modelFolder = "/Users/Cgarcia/git/deep-icy/models/HPA Bestfitting InceptionV3_30102022_133313";
		String modelSource = "/Users/Cgarcia/git/deep-icy/models/HPA Bestfitting InceptionV3_30102022_133313/bestfitting-inceptionv3-single-cell.onnx";
		Model model = loadModel(engine, engineVersion, enginesDir, modelFolder, modelSource);
		
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		 
		// create an 3d-Img with dimensions 20x30x40 (here cellsize is 5x5x5)Ø
		final Img< FloatType > img1 = imgFactory.create( 1, 4, 128, 128);

		Tensor<FloatType> inpTensor = Tensor.build("image", "bcyx", img1);
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		inputs.add(inpTensor);
		
		// We need to specify the output tensors with its axes order and name, but empty
		/* TODO
		 * TODO
		 * TODO implement this for Tischer
		 * TODO
		 * TODO
		 * Tensor<T> outTensor = Tensor.buildEmptyTensor("output0", "bcyx", new long[] {1,2,3,4}, datatype);
		 */
		Tensor<T> outTensor = Tensor.buildEmptyTensor("classes", "bc");
		Tensor<T> outTensor2 = Tensor.buildEmptyTensor("features", "bc");
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
		outputs.add(outTensor);
		outputs.add(outTensor2);
		
		outputs = model.runModel(inputs, outputs);
		System.out.print(false);
		
		// The result is stored in the list of tensors "outputs"
	}
	
	public static Model loadModel(String engine, String engineVersion, String enginesDir, String modelFolder, String modelSource) throws LoadEngineException, Exception {
		
		boolean cpu = true;
		boolean gpu = false;
		EngineInfo engineInfo = EngineInfo.defineDLEngine(engine, engineVersion, enginesDir, cpu, gpu);
		
		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		return model;
	}
}
