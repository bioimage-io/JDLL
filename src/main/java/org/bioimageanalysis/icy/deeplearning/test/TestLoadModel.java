package org.bioimageanalysis.icy.deeplearning.test;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadEngineException;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;

public class TestLoadModel {

	
	public static void main(String[] args) throws LoadEngineException, Exception {
		String enginesDir = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines";
		boolean cpu = true;
		boolean gpu = true;
		EngineInfo engineInfo = EngineInfo.defineDLEngine("torchscript", "1.7.1", enginesDir, cpu, gpu);
		
		String modelFolder = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\arabidopsis-ovules-boundarymodel";
		String modelSource = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\arabidopsis-ovules-boundarymodel\\weights-torchscript.pt";

		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo, null);
		model.loadModel();
		System.out.println("done");
	}
}
