package io.bioimage.modelrunner.example;

import java.io.File;
import java.io.IOException;

import io.bioimage.modelrunner.bioimageio.download.DownloadTracker;
import io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer;
import io.bioimage.modelrunner.engine.installation.EngineManagement;
import io.bioimage.modelrunner.versionmanagement.AvailableEngines;
import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;

/**
 * Class that provides an example on how to download a Deep Learning framework (engine)
 * using JDLL
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class ExampleDownloadEngine {
	/**
	 * Field 'id' in the rdf.yaml file.
	 * Used in this example to download the engines requried to run the wanted model.
	 * However, regard that the engines can also be downloaded on their own, or
	 * again for a model, but providing either the name, the 'rdf_source' field
	 * of the rdf.yaml file or a {@link ModelDescriptor} object generated from an rdf.yaml file.
	 * 
	 * For more details, explore the methods: {@link EngineManagement#installEnginesForModelByNameinDir(String, String)},
	 * {@link EngineManagement#installEnginesForModelInDir(io.bioimage.modelrunner.bioimageio.description.ModelDescriptor, String)},
	 * {@link EngineManagement#installEnginesForModelByNameinDir(String, String, TwoParameterConsumer)},
	 * or to install the engine without associating it to a model:
	 * {@link EngineManagement#installEngineWithArgs(String, String, boolean, boolean)}, 
	 * {@link EngineManagement#installEngine(DeepLearningVersion)},
	 * {@link EngineManagement#installEngineByCompleteName(String)},
	 * {@link EngineManagement#installEngineForWeightsInDir(io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat, String)}
	 */
	private static final String MODEL_ID = "10.5281/zenodo.5874741";
	/**
	 * Current directory
	 */
	private static final String CWD = System.getProperty("user.dir");
	/**
	 * Directory where the engine will be downloaded, if you want to download it
	 * into another folder, please change it.
	 */
	private static final String ENGINES_DIR = new File(CWD, "engines").getAbsolutePath();
	
	/**
	 * Test method to check the download of an engine.
	 * First the download of the required engines for a mdoel will be tested and then
	 * an engine will be installed on its own, providing some parameters
	 * @param args
	 * 	there are no args in this method
	 * @throws IOException if there is any error related to finding the model or its files
	 * 	on the internet
	 * @throws InterruptedException if the thread is stopped while the model is being downloaded
	 */
	public static void main(String[] args) throws IOException, InterruptedException {
		// Create a consumer that gets live information about the download.
		// This consumer contains a LinkedHashMap that where the keys
		// correspond to the file being downloaded and the value corresponds
		// to the fraction of file that has already been downloaded.
		TwoParameterConsumer<String, Double> consumer = DownloadTracker.createConsumerProgress();
		// Download the engines in the wanted dir that are needed to run the model
		// defined by the model ID.
		// This method prints information about the total progress of the download and of the 
		// particular files being downloaded on the terminal.
		EngineManagement.installEnginesForModelByIDInDir(MODEL_ID, ENGINES_DIR, consumer);
		// Another option is to launch the download in a separate thread 
		// and wait for it to end while tracking the progress using the consumer
		Thread downloadThread = new Thread(() -> {
			try {
				// In this case, the engine downloaded is defined independently from any model
				String engine = "pytorch";
				String version = "1.11.0";
				boolean gpu = true;
				boolean cpu = true;
				DeepLearningVersion dlv = 
						AvailableEngines.getEngineForOsByParams(engine, version, cpu, gpu);
				EngineManagement.installEngineInDir(dlv, ENGINES_DIR, consumer);
			} catch (IOException | InterruptedException e) {
				// If one of the files to be downloaded is corrupted or the download thread 
				// is stopped abruptly
				e.printStackTrace();
			}
        });
		downloadThread.start();
		
		// Track the engine download
		while (downloadThread.isAlive()) {
			Thread.sleep(1000);
			// GEt the total progress of the download
			Double totalProgress = consumer.get().get(DownloadTracker.TOTAL_PROGRESS_KEY);
			System.out.println("TOTAL PROGRESS OF THE DOWNLOAD: " + totalProgress);
		}
	}
}
