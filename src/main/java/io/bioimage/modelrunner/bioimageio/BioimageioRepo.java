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
package io.bioimage.modelrunner.bioimageio;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Map.Entry;
import java.util.function.Consumer;

import javax.net.ssl.HttpsURLConnection;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import io.bioimage.modelrunner.bioimageio.bioengine.BioEngineAvailableModels;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.bioimageio.download.DownloadTracker;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.Log;

/**
 * Class to interact with the Bioimage.io API. Used to get information
 * about models and to download them
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class BioimageioRepo {
	/**
	 * Message displayed when there are no models found
	 */
	private static final String MODELS_NOT_FOUND_MSG = "BioImage.io: Unable to find models.";
	/**
	 * Message displayed when there is an API error
	 */
	private static final String API_ERR_MSG = "BioImage.io: There has been an error accessing the API. No model retrieved.";
	/**
	 * URL to the file containing all the model zoo models
	 */
	public static String location = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/all_versions.json";
	/**
	 * JSon containing all the info about the Bioimage.io models
	 */
	private JsonArray collections;
	/**
	 * List of all the IDs of the models existing in the BioImage.io
	 */
	private static List<String> modelIDs;
	
	private static LinkedHashMap<Path, ModelDescriptor> models;
	
	/**
	 * Structure of the map:
	 * 
	 * happy-fish :{
	 * 				0.1.0:{
	 * 					   source: http/:.....
	 * 					   latest: false
	 * 					  }
	 * 				0.2.0:{
	 * 					   source: http/:.....
	 * 					   latest: false
	 * 					  }
	 * 				0.3.0:{
	 * 					   source: http/:.....
	 * 					   latest: true
	 * 					  }
	 * 				}
	 * 
	 * 
	 */
	private static Map<String, Map<String, Map<String, String>>> modelsInfo;
		
	private Consumer<String> consumer;
	
	/**
	 * Constructor for the object that retrieves information about the 
	 * models in teh Bioimage.io repo. It also handles the download of said models.
	 */
	private BioimageioRepo() {
		setCollectionsRepo();
	}
	
	/**
	 * Constructor for the object that retrieves information about the 
	 * models in teh Bioimage.io repo. It also handles the download of said models.
	 * The provided consumer passes the info about the models retrieved to the main program
	 * @param consumer
	 * 	String consumer that will contain the info about the models accessed (time accessed, 
	 * 	name, ...)
	 */
	private BioimageioRepo(Consumer<String> consumer) {
		this.consumer = consumer;
		setCollectionsRepo();
	}
	
	/**
	 * Create an instance of {@link BioimageioRepo}. This instance can be used to retrieve 
	 * information about the models i the Bioimage.io repository and handle their download.
	 * @return an instance of the {@link BioimageioRepo}
	 */
	public static BioimageioRepo connect() {
		return new BioimageioRepo();
	}
	
	/**
	 * Create an instance of {@link BioimageioRepo}. This instance can be used to retrieve 
	 * information about the models i the Bioimage.io repository and handle their download.
	 * @param consumer
	 * 	a String consumer that will record all the info about the access to the models in the
	 * 	bioimage.io
	 * @return an instance of the {@link BioimageioRepo}
	 */
	public static BioimageioRepo connect(Consumer<String> consumer) {
		return new BioimageioRepo(consumer);
	}
	
	/**
	 * Refresh the list of models fetched from the Bioimage.io.
	 * Connects to the Bioimage.io website and retrieves all the models available
	 */
	public void refresh() {
		models = null;
		listAllModels(false);
	}
	
	/**
	 * Method that connects to the BioImage.io API and retrieves the models available
	 * at the Bioimage.io model repository.
	 * The models are specified at: {@link #location}.
	 * Once the method has been called, the list of models is not refreshed (that means
	 * the method does not check the list of Bioimage.io models and returns what 
	 * was obtained with the first call) unless the method {@link #refresh()}
	 * is used, {@link #refresh()} actually calls again this method to retrieve the list from zero,
	 * if not the same list as the one retrieved for the first time calling the method is used..
	 * @param verbose
	 * 	whether to print in the terminal and send that printed information in the consumer (if it 
	 * 	exists) or not
	 * @return an object containing the URL location of the model as key and the {@link ModelDescriptor}
	 * 	with the yaml file information in the value
	 */
	public Map<Path, ModelDescriptor> listAllModels(boolean verbose) {
		if (models != null && models.entrySet().size() > 0)
			return models;
		if (verbose)
			Log.addProgressAndShowInTerminal(consumer, "BioImage.io: Accessing the BioImage.io API to retrieve available models", true);
		models = new LinkedHashMap<Path, ModelDescriptor>();
		if (collections == null) {
			if (verbose)
				Log.addProgressAndShowInTerminal(consumer, MODELS_NOT_FOUND_MSG, true);
			return models;
		}
		for (Object resource : collections) {
			if (Thread.interrupted())
				break;
			JsonObject jsonResource = (JsonObject) resource;
			try {
				if (jsonResource.get("type") == null || !jsonResource.get("type").getAsString().equals("model"))
					continue;
				String url = jsonResource.get("versions").getAsJsonArray().asList().stream()
					    .max(Comparator.comparingLong(elem -> strToTimestamp(elem.getAsJsonObject().get("created").getAsString())))
					    .map(elem -> elem.getAsJsonObject().get("source").getAsString())
					    .orElseThrow(null);
				if (url == null)
					continue;
				String stringRDF = getJSONFromUrl(url);
				ModelDescriptor descriptor = ModelDescriptorFactory.readFromYamlTextString(stringRDF);
				models.put(Paths.get(url), descriptor);
			} catch (Exception ex) {
				// TODO Maybe add some error message? This should be responsibility of the BioImage.io user
				// Only display error message if there was an error creating
				// the descriptor from the yaml file
				if (verbose) {
					String errMSg = "Could not load descriptor for the Bioimage.io model " + jsonResource.get("concept") + ".";
					Log.addProgressAndShowInTerminal(consumer, errMSg, true);
				}
                ex.printStackTrace();
			}
		}
		return models;
	}
	
	/**
	 * MEthod that reads the yaml file that contains all the information about the bioimage.io.
	 * Including the models available.
	 * This method also stores the model IDs of the available models.
	 * The file is at: {@link #location}
	 */
	private void setCollectionsRepo() {
		modelIDs = new ArrayList<String>();
		modelsInfo = new HashMap<String, Map<String, Map<String, String>>>();
		String text = getJSONFromUrl(location);
		if (text == null) {
			Log.addProgressAndShowInTerminal(consumer, MODELS_NOT_FOUND_MSG, true);
			Log.addProgressAndShowInTerminal(consumer, "BioImage.io: Cannot access file: " + location, true);
			Log.addProgressAndShowInTerminal(consumer, "BioImage.io: Please review the certificates needed to access the website.", true);
			return;
		}
		JsonObject json = null;
		try {
			json = (JsonObject) JsonParser.parseString(text);
		} catch (Exception ex) {
			collections = null;
			Log.addProgressAndShowInTerminal(consumer, MODELS_NOT_FOUND_MSG, true);
			return;
		}
		// Iterate over the array corresponding to the key: "resources"
		// which contains all the resources of the Bioimage.io
		collections = (JsonArray) json.get("entries");
		if (collections == null) {
			Log.addProgressAndShowInTerminal(consumer, MODELS_NOT_FOUND_MSG, true);
			return;
		}
		for (Object resource : collections) {
			JsonObject jsonResource = (JsonObject) resource;
			if (jsonResource.get("type") == null || !jsonResource.get("type").getAsString().equals("model"))
				continue;
			String modelID = jsonResource.get("concept").getAsString();
			modelIDs.add(modelID);
			
			HashMap<String, Map<String, String>> vMap = new HashMap<String, Map<String, String>>();
			for (JsonElement vv : jsonResource.get("versions").getAsJsonArray()) {
				JsonObject vvObject = vv.getAsJsonObject();
				Map<String, String> map = new HashMap<String, String>();
				map.put("latest", "false");
				map.put("source", vvObject.get("source").getAsString());
				vMap.put(vvObject.get("v").getAsString(), map);
			}
			String lastV = jsonResource.get("versions").getAsJsonArray().asList().stream()
				    .max(Comparator.comparingLong(elem -> strToTimestamp(elem.getAsJsonObject().get("created").getAsString())))
				    .map(elem -> elem.getAsJsonObject().get("v").getAsString())
				    .orElseThrow(null);
			vMap.get(lastV).put("latest", "true");
			modelsInfo.put(modelID, vMap);
		}
	}
	
	/**
	 * Retrieve the {@link ModelDescriptor} for the rdf.yaml whose URL
	 * has been provided.
	 * If the URL does not exist, or does not point to a valid rdf.yaml file,
	 * the method will return null. 
	 * Regard that the only URLs that will be valid are the ones defined in the
	 * field 'rdf_source' of the rdf.yaml file and that the URLs have to point
	 * to the *raw* github file.
	 * @param rdfSource
	 * 	URL pointing to the rdf.yaml file of interest
	 * @return the {@link ModelDescriptor} from the rdf.yaml of interest or null
	 * if the URL does not point to a valid URL
	 */
	public static ModelDescriptor retreiveDescriptorFromURL(String rdfSource) {
		ModelDescriptor descriptor = null;
		String stringRDF = getJSONFromUrl(rdfSource);
		if (stringRDF == null)
			return descriptor;
		try {
			descriptor = ModelDescriptorFactory.readFromYamlTextString(stringRDF);
		} catch (Exception ex) {
			return descriptor;
		}
		return descriptor;
	}

	/**
	 * Method used to read a yaml or json file from a server as a raw string
	 * @param url
	 * 	String url of the file
	 * @return a String representation of the file. It is null if the file was not accessed
	 */
	public static String getJSONFromUrl(String url) {
		return getJSONFromUrl(url, null);
	}

	/**
	 * Method used to read a yaml or json file from a server as a raw string
	 * @param url
	 * 	String url of the file
	 * @param consumer
	 * 	object to communicate with the main interface
	 * @return a String representation of the file. It is null if the file was not accessed
	 */
	private static String getJSONFromUrl(String url, Consumer<String> consumer) {

		HttpsURLConnection con = null;
		try {
			URL u = new URL(url);
			con = (HttpsURLConnection) u.openConnection();
			con.connect();
			InputStream inputStream = con.getInputStream();
			
			 ByteArrayOutputStream result = new ByteArrayOutputStream();
			 byte[] buffer = new byte[1024];
			 for (int length; (length = inputStream.read(buffer)) != -1; ) {
			     result.write(buffer, 0, length);
			 }
			 // StandardCharsets.UTF_8.name() > JDK 7
			 String txt = result.toString("UTF-8");
			 inputStream.close();
			 result.close();
			 return txt;
		} 
		catch (IOException ex) {
			Log.addProgressAndShowInTerminal(consumer, API_ERR_MSG, true);
			ex.printStackTrace();
		} 
		finally {
			if (con != null) {
				try {
					con.disconnect();
				} catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		}
		return null;
	}
	
	/**
	 * Create {@link Path} from Url String. This method removes the http:// or https://
	 * at the begining because in windows machines it caused errors creating Paths
	 * @param downloadUrl
	 * 	String url of the model of interest
	 * @return the path to the String url
	 */
	public static Path createPathFromURLString(String downloadUrl) {
		Path path;
		try {
			if (downloadUrl.startsWith("https://")) {
				downloadUrl = downloadUrl.substring(("https://").length());
			} else if (downloadUrl.startsWith("http://")) {
				downloadUrl = downloadUrl.substring(("http://").length());
			}
			path = new File(downloadUrl).toPath();
		} catch (Exception ex) {
			int startName = downloadUrl.lastIndexOf("/");
			downloadUrl = downloadUrl.substring(startName + 1);
			path = new File(downloadUrl).toPath();
		}
		return path;
	}
	
	/**
	 * Return a list with all the model IDs for the models existing in the Bioimage.io repo
	 * @return list with the ids for each of the models in the repo
	 */
	public static List<String> getModelIDs(){
		if (modelIDs == null || modelIDs.size() == 0)
			BioimageioRepo.connect();
		if (modelIDs == null)
			return new ArrayList<String>();
		return modelIDs;
	}
	
	/**
	 * Check whether a model is available on the Bioengine or not by providing its id
	 * @param id
	 * 	id of the model of interest
	 * @return true if it is available or false otherwise
	 * @throws IOException if there is no connection to the internet or the
	 * 	JSOn with the information cannot be accessed: https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.json
	 */
	public static boolean isModelOnTheBioengineById(String id) throws IOException {
		return BioEngineAvailableModels.isModelSupportedInBioengine(id);
	}
	
	/**
	 * Return the {@link ModelDescriptor} for the model defined by the modelID
	 * (field 'id' in the rdf.yaml) introduced as a parameter.
	 * @param modelID
	 * 	unique ID for each Bioimage.io model
	 * @return the {@link ModelDescriptor} of the model
	 */
	public ModelDescriptor selectByID(String modelID) {
		Objects.requireNonNull(modelID, "Argument 'modelID' cannot be null.");
		Entry<Path, ModelDescriptor> modelEntry = this.listAllModels(false).entrySet().stream()
				.filter(ee -> {
					String id = ee.getValue().getModelID();
					if (id.length() - id.replace("/", "").length() == 2) {
						id = id.substring(0, id.lastIndexOf("/"));
					}
					if (modelID.length() - modelID.replace("/", "").length() == 2) {
						return modelID.substring(0, modelID.lastIndexOf("/")).equals(id);
					}
					return modelID.equals(id);
				}).findFirst().orElse(null);
		if (modelEntry != null)
			return modelEntry.getValue();
		return null;
	}
	
	/**
	 * Return the {@link ModelDescriptor} for the model defined by the name
	 * (field 'name' in the rdf.yaml) introduced as a parameter.
	 * @param name
	 * 	unique name for each Bioimage.io model
	 * @return the {@link ModelDescriptor} of the model
	 */
	public ModelDescriptor selectByName(String name) {
		Objects.requireNonNull(name, "Argument 'name' cannot be null.");
		Entry<Path, ModelDescriptor> modelEntry = this.listAllModels(false).entrySet().stream()
				.filter(ee -> ee.getValue().getName().equals(name)).findFirst().orElse(null);
		if (modelEntry != null)
			return modelEntry.getValue();
		return null;
		
	}
	
	/**
	 * Download the model in the Bioimage.io defined by the {@link ModelDescriptor}
	 * provided as a parameter.
	 * This method launches one thread for the download of the files of the model and 
	 * another thread to track the progress download. The thread where this method has
	 * been launched is just used to print the information about the progress using
	 * {@link DownloadTracker#printProgress(Thread, io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer)}
	 * 
	 * @param descriptor
	 * 	the {@link ModelDescriptor} of the model that wants to be downloaded
	 * @param modelsDirectory
	 * 	the folder where the model is going to be downloaded. Regard that the model
	 * 	is a folder too. So if the argument provided is "C:\\users\\carlos\\models",
	 * 	the model path will then be: "C:\\users\\carlos\\models\\model_name_date string""
	 * @return the path to the model that was just installed. 
	 * @throws IOException	if there is any error downloading the files from the URLs provided
	 * @throws InterruptedException	if the download or tracking threads are interrupted abruptly
	 */
	public static String downloadModel(ModelDescriptor descriptor, String modelsDirectory) 
			throws IOException, InterruptedException {
		return downloadModel(descriptor, modelsDirectory, null);
		}
	
	/**
	 * Download the model in the Bioimage.io defined by the {@link ModelDescriptor}
	 * provided as a parameter.
	 * This method launches one thread for the download of the files of the model and 
	 * another thread to track the progress download. The thread where this method has
	 * been launched is just used to print the information about the progress using
	 * {@link DownloadTracker#printProgress(Thread, DownloadTracker.TwoParameterConsumer)}
	 * 
	 * @param descriptor
	 * 	the {@link ModelDescriptor} of the model that wants to be downloaded
	 * @param modelsDirectory
	 * 	the folder where the model is going to be downloaded. Regard that the model
	 * 	is a folder too. So if the argument provided is "C:\\users\\carlos\\models",
	 * 	the model path will then be: "C:\\users\\carlos\\models\\model_name_date string""
	 * @param consumer
	 * 	a {@link DownloadTracker.TwoParameterConsumer} consumer that ccan be used to track the
	 * 	download of the individual files that compose the model.
	 * @return the path to the model that was just installed. 
	 * @throws IOException	if there is any error downloading the files from the URLs provided
	 * @throws InterruptedException	if the download or tracking threads are interrupted abruptly
	 */
	public static String downloadModel(ModelDescriptor descriptor, String modelsDirectory, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		DownloadModel dm = DownloadModel.build(descriptor, modelsDirectory);
		Thread downloadThread = new Thread(() -> {
			try {
				dm.downloadModel();
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}
        });
		if (consumer == null)
			consumer = DownloadTracker.createConsumerProgress();
		DownloadTracker mdt = DownloadTracker.getBMZModelDownloadTracker(consumer, dm, downloadThread);
		downloadThread.start();
		Thread trackerThread = new Thread(() -> {
            try {
				mdt.track();
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}
        });
		trackerThread.start();
		try { DownloadTracker.printProgress(downloadThread, consumer); } 
		catch (InterruptedException ex) { throw new InterruptedException("Model download interrupted."); }
		
		List<String> badDownloads = mdt.findMissingDownloads();
		
		if (badDownloads.size() > 0 && Thread.currentThread().isAlive())
			throw new IOException("The following files of model '" + descriptor.getName()
			+ "' were downloaded incorrectly: " + badDownloads.toString());
		return dm.getModelFolder();
	}
	
	/**
	 * Download the model in the Bioimage.io whose id (field 'id' in the
	 * rdf.yaml file) corresponds to the first parameter given
	 * 
	 * This method launches one thread for the download of the files of the model and 
	 * another thread to track the progress download. The thread where this method has
	 * been launched is just used to print the information about the progress using
	 * {@link DownloadTracker#printProgress(Thread, io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer)}
	 * 
	 * @param id
	 * 	the id of the model of interest. This is the field 'id' of the model descriptor
	 * @param modelsDirectory
	 * 	the folder where the model is going to be downloaded. Regard that the model
	 * 	is a folder too. So if the argument provided is "C:\\users\\carlos\\models",
	 * 	the model path will then be: "C:\\users\\carlos\\models\\model_name_date string""
	 * @return the path to the model that was just installed. 
	 * @throws IOException	if there is any error downloading the files from the URLs provided
	 * @throws InterruptedException	if the download or tracking threads are interrupted abruptly
	 */
	public String  downloadModelByID(String id, String modelsDirectory) throws IOException, InterruptedException {
		ModelDescriptor model = selectByID(id);
		if (model == null)
			throw new IllegalArgumentException("The provided id does not correspond "
					+ "to an existing model in the Bioimage.io online repo.");
		return downloadModel(model, modelsDirectory, null);
	}
	
	/**
	 * Download the model in the Bioimage.io whose id (field 'id' in the
	 * rdf.yaml file) corresponds to the first parameter given
	 * 
	 * This method launches one thread for the download of the files of the model and 
	 * another thread to track the progress download. The thread where this method has
	 * been launched is just used to print the information about the progress using
	 * {@link DownloadTracker#printProgress(Thread, io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer)}
	 * 
	 * @param id
	 * 	the id of the model of interest. This is the field 'id' of the model descriptor
	 * @param modelsDirectory
	 * 	the folder where the model is going to be downloaded. Regard that the model
	 * 	is a folder too. So if the argument provided is "C:\\users\\carlos\\models",
	 * 	the model path will then be: "C:\\users\\carlos\\models\\model_name_date string""
	 * @param consumer
	 * 	a {@link DownloadTracker.TwoParameterConsumer} consumer that ccan be used to track the
	 * 	download of the individual files that compose the model.
	 * @return the path to the model that was just installed. 
	 * @throws IOException	if there is any error downloading the files from the URLs provided
	 * @throws InterruptedException	if the download or tracking threads are interrupted abruptly
	 */
	public String downloadModelByID(String id, String modelsDirectory, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		ModelDescriptor model = selectByID(id);
		if (model == null)
			throw new IllegalArgumentException("The provided id does not correspond "
					+ "to an existing model in the Bioimage.io online repo.");
		return downloadModel(model, modelsDirectory, consumer);
	}
	
	/**
	 * Download the model in the Bioimage.io whose name (field 'name' in the
	 * rdf.yaml file) corresponds to the first parameter given
	 * 
	 * This method launches one thread for the download of the files of the model and 
	 * another thread to track the progress download. The thread where this method has
	 * been launched is just used to print the information about the progress using
	 * {@link DownloadTracker#printProgress(Thread, io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer)}
	 * 
	 * @param name
	 * 	the name of the model of interest. This is the field 'name' of the model descriptor
	 * @param modelsDirectory
	 * 	the folder where the model is going to be downloaded. Regard that the model
	 * 	is a folder too. So if the argument provided is "C:\\users\\carlos\\models",
	 * 	the model path will then be: "C:\\users\\carlos\\models\\model_name_date string""
	 * @return the path to the model that was just installed. 
	 * @throws IOException	if there is any error downloading the files from the URLs provided
	 * @throws InterruptedException	if the download or tracking threads are interrupted abruptly
	 */
	public String downloadByName(String name, String modelsDirectory) throws IOException, InterruptedException {
		ModelDescriptor model = selectByName(name);
		if (model == null)
			throw new IllegalArgumentException("The provided name does not correspond "
					+ "to an existing model in the Bioimage.io online repo.");
		return downloadModel(model, modelsDirectory, null);
	}
	
	/**
	 * Download the model in the Bioimage.io whose name (field 'name' in the
	 * rdf.yaml file) corresponds to the first parameter given
	 * 
	 * This method launches one thread for the download of the files of the model and 
	 * another thread to track the progress download. The thread where this method has
	 * been launched is just used to print the information about the progress using
	 * {@link DownloadTracker#printProgress(Thread, io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer)}
	 * 
	 * @param name
	 * 	the name of the model of interest. This is the field 'name' of the model descriptor
	 * @param modelsDirectory
	 * 	the folder where the model is going to be downloaded. Regard that the model
	 * 	is a folder too. So if the argument provided is "C:\\users\\carlos\\models",
	 * 	the model path will then be: "C:\\users\\carlos\\models\\model_name_date string""
	 * @param consumer
	 * 	a {@link DownloadTracker.TwoParameterConsumer} consumer that ccan be used to track the
	 * 	download of the individual files that compose the model.
	 * @return the path to the model that was just installed. 
	 * @throws IOException	if there is any error downloading the files from the URLs provided
	 * @throws InterruptedException	if the download or tracking threads are interrupted abruptly
	 */
	public String downloadByName(String name, String modelsDirectory, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		ModelDescriptor model = selectByName(name);
		if (model == null)
			throw new IllegalArgumentException("The provided name does not correspond "
					+ "to an existing model in the Bioimage.io online repo.");
		return downloadModel(model, modelsDirectory, consumer);
	}
	
	private static long strToTimestamp(String str) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
        LocalDateTime localDateTime = LocalDateTime.parse(str, formatter);
        Timestamp timestamp = Timestamp.valueOf(localDateTime);
        return timestamp.getTime();
	}
	
	
	public String getModelURL(String id) {
		return getModelRdfUrl(id, null);
	}
	
	
	public String getModelRdfUrl(String id, String version) {
		if (modelsInfo == null || modelsInfo.get("id") == null) {
			BioimageioRepo.connect();
		}
		
		if (modelsInfo.get(id) == null)
			return null;
		String rdfURL;
		if (version == null) {
			rdfURL = modelsInfo.get(id).values().stream()
					.filter(val -> val.get("latest").equals("true")).findFirst().get().get("source");
		} else if (version != null && modelsInfo.get(id).get(version) != null) {
			rdfURL = modelsInfo.get(id).get(version).get("source");
		} else {
			return null;
		}
		if (rdfURL == null) return null;
		
		return rdfURL.substring(0, rdfURL.length() - Constants.RDF_FNAME.length());
	}
	
	
	public static void main(String[] args) {
		BioimageioRepo br = new BioimageioRepo();
		br.listAllModels(false);
	}
}
