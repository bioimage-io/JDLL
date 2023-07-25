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
package io.bioimage.modelrunner.bioimageio.bioengine;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import javax.net.ssl.HttpsURLConnection;

import io.bioimage.modelrunner.utils.Constants;

import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;

/**
 * Class to handle the JSON file that contains the information about the 
 * models supported by the Bioengine. The model is located at: 
 * https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.json
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class BioEngineAvailableModels {

	/**
	 * Collection of models available at the bioengine
	 */
	private ArrayList<LinkedTreeMap<String, String>> collection;
	/**
	 * Logs for the CI of the Bioengine
	 */
	private ArrayList<String> conversion_logs;
	/**
	 * Description of the file
	 */
	private String description;
	/**
	 * Format version
	 */
	private String format_version;
	/**
	 * Name of the file
	 */
	private String name;
	/**
	 * Tags fo the file
	 */
	private ArrayList<String> tags;
	/***
	 * Type of the file
	 */
	private String type;
	/**
	 * Version of the file
	 */
	private String version;

	/**
	 * URL of the Json file containing the bioengine compatible models json
	 */
	final private static String BIOENGINE_COMPATIBLE_JSON = 
			"https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.json";
	/**
	 * Key for the ID of the models supported by the bioengine in the collections object
	 */
	final private static String ID_KEY = "id";
	
	private static BioEngineAvailableModels BAM;
	/**
	 * Address of the first public server that hosts a bioengine instance
	 */
	public static final String PUBLIC_BIOENGINE_SERVER = "https://ai.imjoy.io";
	
	/**
	 * Method that parses the json file that contains all the supported models by the BioEngine
	 * @return obejct containing the spported models by the bioengine
	 * @throws IOException if the bioengine website cannot 
	 * be accessed or if there is any error parsing the json
	 */
	public static BioEngineAvailableModels load() throws IOException
    {
		HttpsURLConnection con;
		try {
			URL url = new URL(BIOENGINE_COMPATIBLE_JSON);
			con = (HttpsURLConnection) url.openConnection();
		} catch (IOException ex) {
			throw new IOException("Unable to access the following link: " + BIOENGINE_COMPATIBLE_JSON
					+ System.lineSeparator() + "Please, check your Internet connection. If the site does "
					+ "not exist, please report it at : " + Constants.ISSUES_LINK);
		}
		try (InputStream inputStream = con.getInputStream();
			InputStreamReader inpStreamReader = new InputStreamReader(inputStream);
		    BufferedReader br = new BufferedReader(inpStreamReader);){
	        Gson g = new Gson();
	        return g.fromJson(br, BioEngineAvailableModels.class); 
		} catch (IOException ex) {
			throw new IOException("Unable to parse JSON that contains BioEngine compatible models at: "
					+ System.lineSeparator() + BIOENGINE_COMPATIBLE_JSON + System.lineSeparator()
					+ ex.getCause().toString());
		}
    }
	
	/**
	 * Creates empty BioEngineAvailableModels object that works with all its methods.
	 * Created to avoid errors caused by {@link NullPointerException}
	 * @return
	 */
	private static BioEngineAvailableModels createEmptyObject() {
		BioEngineAvailableModels availableModels = new BioEngineAvailableModels();
		availableModels.collection = new ArrayList<LinkedTreeMap<String, String>>();
		availableModels.conversion_logs = new ArrayList<String>();
		return availableModels;
	}
	
	/**
	 * Method that returns a list of the IDs corresponding to the models that are supported by 
	 * the bioengine
	 * @return
	 */
	public List<String> getListOfSupportedIDs() {
		return collection.stream().map(x -> x.get(ID_KEY)).collect(Collectors.toList());
	}
	
	/**
	 * Method that finds whether a model is supported by the Bioengine by looking if its
	 * model ID is spedified in the JSON file that contains all valid models
	 * @param modelID
	 * 	ID of the model of interest
	 * @return true if supported, false otherwise
	 */
	public boolean isModelSupportedByBioengine(String modelID) {
		LinkedTreeMap<String, String> isFound = collection.stream()
				.filter(x -> modelID.startsWith(x.get(ID_KEY)+ "/")).findFirst().orElse(null);
		if (isFound == null)
			return false;
		return true;
	}
	
	/**
	 * Static method that finds whether a model is supported by the Bioengine by looking if its
	 * model ID is spedified in the JSON file that contains all valid models.
	 * This method only recovers the file that contains the information of the models
	 * supported by the Bioengine the first time. In order to retrieve the file each
	 * time, do the following:
	 * 	BioEngineAvailableModels bam =  BioEngineAvailableModels.load();
	 * 	String id = "model/ID":
	 * 	boolean supported = bam.isModelSupportedByBioengine(id);
	 * @param modelID
	 * 	ID of the model of interest
	 * @return true if supported, false otherwise
	 * @throws IOException if the json with the Bioengine information is not found, or 
	 * 	there is no internet
	 */
	public static boolean isModelSupportedInBioengine(String modelID) throws IOException {
		if (BAM == null) {
			try {
				BAM =  BioEngineAvailableModels.load();
			} catch (IOException ex) {
				BAM = createEmptyObject();
			}
		}
		return BAM.isModelSupportedByBioengine(modelID);
	}
	
	public ArrayList<LinkedTreeMap<String, String>> getCollection() {
		return this.collection;
	}
	
	public String getType() {
		return this.type;
	}
	
	public ArrayList<String> getConversionLogs() {
		return this.conversion_logs;
	}
	
	public ArrayList<String> getTags() {
		return this.tags;
	}
	
	public String getName() {
		return this.name;
	}
	
	public String getFormatVersion() {
		return this.format_version;
	}
	
	public String getVersion() {
		return this.version;
	}
	
	public String getDescription() {
		return this.description;
	}
	
	public static String getBioengineJson() {
		return BIOENGINE_COMPATIBLE_JSON;
	}
}
