/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;

/**
 * Class to interact with the Bioimage.io API. Used to get information
 * about models and to download them
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class BioimageioDirectConnection {
	/**
	 * URL to the file containing all the model zoo models
	 */
	public static final String LOCATE = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/all_versions.json";
	/**
	 * JSon containing all the info about the Bioimage.io models
	 */
	private static JsonArray COLLECTIONS;
	
	private static HashMap<String, ModelDescriptor> MODELS = new HashMap<String, ModelDescriptor>();
	
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
	private static List<Map<String, Object>> MODELS_INFO;
	
	private static final String SEPARATOR = UUID.randomUUID().toString();
	
	/**
	 * Refresh the list of models fetched from the Bioimage.io.
	 * Connects to the Bioimage.io website and retrieves all the models available
	 */
	public static void refresh() {
		MODELS = new HashMap<String, ModelDescriptor>();
		setCollectionsRepo();
	}
	
	/**
	 * MEthod that reads the yaml file that contains all the information about the bioimage.io.
	 * Including the models available.
	 * This method also stores the model IDs of the available models.
	 * The file is at: {@link #LOCATE}
	 */
	private  static void setCollectionsRepo() {
		MODELS_INFO = new ArrayList<Map<String, Object>>();
		String text = BioimageioRepo.getJSONFromUrl(LOCATE, null);
		if (text == null) {
			return;
		}
		JsonObject json = null;
		try {
			json = (JsonObject) JsonParser.parseString(text);
		} catch (Exception ex) {
			COLLECTIONS = null;
			return;
		}
		// Iterate over the array corresponding to the key: "resources"
		// which contains all the resources of the Bioimage.io
		COLLECTIONS = (JsonArray) json.get("entries");
		if (COLLECTIONS == null) {
			return;
		}
		for (Object resource : COLLECTIONS) {
			JsonObject jsonResource = (JsonObject) resource;
			if (jsonResource.get("type") == null || !jsonResource.get("type").getAsString().equals("model"))
				continue;
			Map<String, Object> map = new HashMap<String, Object>();
			map.put("concept", jsonResource.get("concept").isJsonNull() ? null : jsonResource.get("concept").getAsString());
			map.put("concept_doi", jsonResource.get("concept_doi").isJsonNull() ? null : jsonResource.get("concept_doi").getAsString());

			List<Map<String, Object>> vArray = new ArrayList<Map<String, Object>>();
			String maxCreated = "";
			for (JsonElement vv : jsonResource.get("versions").getAsJsonArray()) {
				Map<String, JsonElement> jsonMap = vv.getAsJsonObject().asMap();
				Map<String, Object> vMap = new HashMap<String, Object>();
				for (String kk : jsonMap.keySet())
					vMap.put(kk, jsonMap.get(kk).isJsonNull() ? null : jsonMap.get(kk).getAsString());
				if (vMap.get("created") != null && ((String) vMap.get("created")).compareTo(maxCreated) > 0)
					maxCreated = (String) vMap.get("created");
				vArray.add(vMap);
			}
			if (maxCreated.equals(""))
				continue;
			for (Map<String, Object> entry : vArray) {
				if (entry.get("created") != null && ((String) entry.get("created")).equals(maxCreated))
					entry.put("latest", true);
				else
					entry.put("latest", false);
			}
			map.put("versions", vArray);
			String slug = ((String) vArray.get(0).get("source")).split("/")[5];
			map.put("slug", slug);
			MODELS_INFO.add(map);
		}
	}
	
	/**
	 * Return the {@link ModelDescriptor} for the model defined by the modelID
	 * (field 'id' in the rdf.yaml) introduced as a parameter.
	 * @param modelID
	 * 	unique ID for each Bioimage.io model
	 * @return the {@link ModelDescriptor} of the model
	 */
	public static ModelDescriptor selectByID(String modelID) {
		return selectByID(modelID, "latest");
	}
	
	/**
	 * Return the {@link ModelDescriptor} for the model defined by the modelID
	 * (field 'id' in the rdf.yaml) introduced as a parameter.
	 * @param modelID
	 * 	unique ID for each Bioimage.io model
	 * @return the {@link ModelDescriptor} of the model
	 */
	public static ModelDescriptor selectByID(String modelID, String version) {
		Objects.requireNonNull(modelID, "ID should be non null");
		if (MODELS.get(modelID + SEPARATOR + version) != null)
			return MODELS.get(modelID + SEPARATOR + version);
		if (MODELS_INFO == null)
			setCollectionsRepo();
		
		Map<String, Object> select = MODELS_INFO.stream().filter(mm -> {
			if (mm.get("concept") != null && modelID.equals(mm.get("concept")))
				return true;
			if (mm.get("slug") != null && modelID.equals(mm.get("slug")))
				return true;
			if (mm.get("concept_doi") != null && modelID.equals(mm.get("concept_doi")))
				return true;
			String beginingID;
			if (!modelID.endsWith("/"))
				beginingID = modelID + "/";
			else
				beginingID = modelID;
			if (mm.get("concept_doi") != null && ((String) mm.get("concept_doi")).startsWith(beginingID))
				return true;
			return false;
		}).findFirst().orElse(null);
		if (select == null)
			return null;
		String url;
		if (version.equals("latest"))
			url = (String) ((List<Map<String, Object>>) select.get("versions")).stream()
										.filter(vv -> (boolean) vv.get("latest"))
										.findFirst().get().get("source");
		else {
			Map<String, Object> map = ((List<Map<String, Object>>) select.get("versions")).stream()
			.filter(vv -> ((String) vv.get("v")).equals(version))
			.findFirst().orElse(null);
			if (map == null)
				return null;
			url = (String) map.get("source");
		}
		if (url == null)
			return null;
		String stringRDF = BioimageioRepo.getJSONFromUrl(url);
		ModelDescriptor descriptor = ModelDescriptorFactory.readFromYamlTextString(stringRDF);
		MODELS.put(modelID + SEPARATOR + version, descriptor);
		return descriptor;
	}
	
	
	public static void main(String[] args) {
		BioimageioDirectConnection.selectByID("affable-shark");
		BioimageioDirectConnection.selectByID("affable-shark");
	}
}
