package io.bioimage.modelrunner.example;

import java.io.File;
import java.io.IOException;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;

/**
 * Class that provides an example on how to download a model using JDLL.
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class ExampleDownloadModel {
	/**
	 * Field 'id' in the rdf.yaml file.
	 * Used in this example to identify the model we want to download.
	 * However, regard that the model can also be downloaded using the
	 * {@link ModelDescriptor} created from its rdf.yaml file, using the 
	 * field 'name' in the rdf.yaml as identifier or using the field
	 * 'rdf_source' as identifeir.
	 * For more details, explore the methods: {@link BioimageioRepo#downloadByName(String, String)},
	 * {@link BioimageioRepo#downloadByRdfSource(String, String)} 
	 * or {@link BioimageioRepo#downloadModel(io.bioimage.modelrunner.bioimageio.description.ModelDescriptor, String)}
	 */
	private static final String MODEL_ID = "10.5281/zenodo.5874741";
	/**
	 * Current directory
	 */
	private static final String CWD = System.getProperty("user.dir");
	/**
	 * Directory where the model will be downloaded, if you want to download it
	 * into another folder, please change it.
	 */
	private static final String MODELS_DIR = new File(CWD, "models").getAbsolutePath();
	
	/**
	 * Test method to check the download of models
	 * @param args
	 * 	there are no args in this method
	 * @throws IOException if there is any error related to finding the model or its files
	 * 	on the internet
	 * @throws InterruptedException if the thread is stopped while the model is being downloaded
	 */
	public static void main(String[] args) throws IOException, InterruptedException {
		// Create an instance of the BioimageRepo object
		BioimageioRepo br = BioimageioRepo.connect();
		// Retrieve a list of all the models that exist in the Bioimage.io repo.
		// Use verbose = false as we don' watn to print any info about the models in the terminal
		boolean verbose = false;
		br.listAllModels(verbose);
		// Download the model using the ID, regard that we can also download the model
		// using its gieven name, the model descriptor or the url to its rdf.yaml file.
		// The download of the model prints some information about the download in the terminal.
		// The download of the model stops the thread until the download is finished.
		br.downloadModelByID(MODEL_ID, MODELS_DIR);
	}
}
