/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.bioimageio;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import javax.net.ssl.HttpsURLConnection;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.utils.Log;

/**
 * Class to interact with the Bioimage.io API. Used to get information
 * about models and to download them
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class ModelRepo {
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
	public static String location = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/collection.json";

	public HashMap<String, ModelDescriptor> models = new HashMap<String, ModelDescriptor>();
	/**
	 * JSon containing all the info about the Bioimage.io models
	 */
	private JsonArray collections;
	/**
	 * List of all the model IDs of the models existing in the BioImage.io
	 */
	private static List<String> modelIDs;
	
	private Consumer<String> consumer;
	
	public static void main(String[] args) {
		connect().listAllModels();
	}
	
	public ModelRepo() {
		setCollectionsRepo();
	}
	
	public ModelRepo(Consumer<String> consumer) {
		this.consumer = consumer;
		setCollectionsRepo();
	}
	
	/**
	 * Create an instance of the models stored in the Bioimage.io repository reading the 
	 * collections rdf.yaml.
	 * @return an instance of the {@link ModelRepo}
	 */
	public static ModelRepo connect() {
		return new ModelRepo();
	}
	
	/**
	 * Create an instance of the models stored in the Bioimage.io repository reading the 
	 * collections rdf.yaml.
	 * @return an instance of the {@link ModelRepo}
	 */
	public static ModelRepo connect(Consumer<String> consumer) {
		return new ModelRepo(consumer);
	}
	
	/**
	 * Method that connects to the BioImage.io API and retrieves the models available
	 * at the Bioimage.io model repository
	 * @return an object containing the zip location of the model as key and the {@link ModelDescriptor}
	 * 	with the yaml file information in the value
	 */
	public Map<Path, ModelDescriptor> listAllModels() {
		Log.addProgressAndShowInTerminal(consumer, "BioImage.io: Accessing the BioImage.io API to retrieve available models", true);
		Map<Path, ModelDescriptor> models = new HashMap<Path, ModelDescriptor>();
		if (collections == null) {
			Log.addProgressAndShowInTerminal(consumer, MODELS_NOT_FOUND_MSG, true);
			return models;
		}
		for (Object resource : collections) {
			if (Thread.interrupted())
				break;
			Path modelPath = null;
			JsonObject jsonResource = (JsonObject) resource;
			try {
				if (jsonResource.get("type") == null || !jsonResource.get("type").getAsString().equals("model"))
					continue;
				String stringRDF = getJSONFromUrl(jsonResource.get("rdf_source").getAsString());
				modelPath = createPathFromURLString(jsonResource.get("rdf_source").getAsString());
				ModelDescriptor descriptor = ModelDescriptor.loadFromYamlTextString(stringRDF);
				models.put(modelPath, descriptor);
			} catch (Exception ex) {
				// TODO Maybe add some error message? This should be responsibility of the BioImage.io user
				// Only display error message if there was an error creating
				// the descriptor from the yaml file
				if (modelPath != null) {
					String errMSg = "Could not load descriptor for the Bioimage.io model " + modelPath.getFileName() + ": " + ex.toString();
					Log.addProgressAndShowInTerminal(consumer, errMSg, true);
				}
                ex.printStackTrace();
			}
		}
		return models;
	}
	
	/**
	 * Method that stores all the model IDs for the models available in the BIoImage.io repo
	 */
	private void setCollectionsRepo() {
		modelIDs = new ArrayList<String>();
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
		collections = (JsonArray) json.get("collection");
		if (collections == null) {
			Log.addProgressAndShowInTerminal(consumer, MODELS_NOT_FOUND_MSG, true);
			return;
		}
		for (Object resource : collections) {
			JsonObject jsonResource = (JsonObject) resource;
			if (jsonResource.get("type") == null || !jsonResource.get("type").getAsString().equals("model"))
				continue;
			String modelID = jsonResource.get("id").getAsString();
			modelIDs.add(modelID);
		}
	}

	/**
	 * Method used to read a yaml or json file from a server as a raw string
	 * @param url
	 * 	String url of the file
	 * @return a String representation of the file. It is null if the file was not accessed
	 */
	private static String getJSONFromUrl(String url) {
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
		return modelIDs;
	}
	
	public ModelDescriptor selecByID(String modelID) {
		
	}
	
	public ModelDescriptor selectByName(String name) {
		
	}
	
	public ModelDescriptor selectByRdfSource(String rdfURL) {
		
	}
}
