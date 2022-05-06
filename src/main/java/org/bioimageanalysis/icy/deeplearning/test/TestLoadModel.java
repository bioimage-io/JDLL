package org.bioimageanalysis.icy.deeplearning.test;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadEngineException;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;

public class TestLoadModel {

	
	public static void main(String[] args) throws LoadEngineException, Exception {
		String enginesDir = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines";
		boolean cpu = true;
		boolean gpu = false;
		EngineInfo engineInfo = EngineInfo.defineDLEngine("tensorflow_saved_model_bundle", "1.15.0", enginesDir, cpu, gpu);
		
		String modelFolder = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\sample";
		String modelSource = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\sample\\tf_weights.zip";

		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		System.out.println("done");
	}
}
