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
import net.imglib2.type.numeric.real.FloatType;

/**
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class ExampleLoadAndRunModel {
	
	public static void main(String[] args) throws LoadEngineException, Exception {
		String engine = "torchscript";
		String engineVersion = "1.7.1";
		String enginesDir = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines";
		String modelFolder = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\arabidopsis-ovules-boundarymodel";
		String modelSource = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\arabidopsis-ovules-boundarymodel\\weights-torchscript.pt";
		Model model = loadModel(engine, engineVersion, enginesDir, modelFolder, modelSource);
		
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		 
		// create an 3d-Img with dimensions 20x30x40 (here cellsize is 5x5x5)Ø
		final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512);

		Tensor inpTensor = Tensor.build("input0", "bcyx", img1);
		List<Tensor> inputs = new ArrayList<Tensor>();
		inputs.add(inpTensor);
		
		// We need to specify the output tensors with its axes order and name, but empty
		Tensor outTensor = Tensor.buildEmptyTensor("output0", "bcyx");
		List<Tensor> outputs = new ArrayList<Tensor>();
		outputs.add(outTensor);
		
		outputs = model.runModel(inputs, outputs);
		
		// The result is stored in the list of tensors "outputs"
	}
	
	public static Model loadModel(String engine, String engineVersion, String enginesDir, String modelFolder, String modelSource) throws LoadEngineException, Exception {
		
		boolean cpu = true;
		boolean gpu = true;
		EngineInfo engineInfo = EngineInfo.defineDLEngine(engine, engineVersion, enginesDir, cpu, gpu);
		
		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		return model;
	}
}
