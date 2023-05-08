package io.bioimage.modelrunner.example;

import java.io.File;
import java.io.IOException;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.download.DownloadTracker;
import io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer;

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
		// Create a consumer that gets live information about the download.
		// This consumer contains a LinkedHashMap that where the keys
		// correspond to the file being downloaded and the value corresponds
		// to the fraction of file that has already been downloaded.
		TwoParameterConsumer<String, Double> consumer = DownloadTracker.createConsumerProgress();
		// Download the model using the ID, regard that we can also download the model
		// using its gieven name, the model descriptor or the url to its rdf.yaml file.
		// The download of the model prints some information about the download in the terminal.
		// The download of the model stops the thread until the download is finished.
		br.downloadModelByID(MODEL_ID, MODELS_DIR, consumer);
		// Another option is to launch the download in a separate thread 
		// and wait for it to end while tracking the progress using the consumer
		Thread downloadThread = new Thread(() -> {
			try {
				br.downloadModelByID(MODEL_ID, MODELS_DIR, consumer);
			} catch (IOException | InterruptedException e) {
				// If one of the files to be downloaded is corrupted or the download thread 
				// is stopped abruptly
				e.printStackTrace();
			}
        });
		downloadThread.start();
		
		// Track the model download
		while (downloadThread.isAlive()) {
			Thread.sleep(1000);
			// GEt the total progress of the download
			Double totalProgress = consumer.get().get(DownloadTracker.TOTAL_PROGRESS_KEY);
			System.out.println("TOTAL PROGRESS OF THE DOWNLOAD: " + totalProgress);
		}
	}
}
