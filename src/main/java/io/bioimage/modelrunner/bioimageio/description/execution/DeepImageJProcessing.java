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
package io.bioimage.modelrunner.bioimageio.description.execution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** TODO be able to run JAva specific code
 * Class to define the DeepImageJ specific transformations. DeepIcy cannot yet execute
 * the model specific Java transformations of some DeepImageJ models due to the DIJ
 * specific interface that is not implemented in Icy
 * 
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class DeepImageJProcessing {
	
	/**
	 * List of macro files containing any kind of pre- or post-processing
	 */
	private List<String> macros;
	/**
	 * Whether the processing executes an external executable.
	 */
	private boolean containsJava = false;
	/**
	 * Key used in the rdf.yaml to specify that the processing uses Macro
	 */
	private static String macroKey = "ij.IJ::runMacroFile";
	/**
	 * Key for the specs in the processing map
	 */
	private static String specsKey = "spec";
	/**
	 * Key for the kwargs in the transformations map
	 */
	private static String kwargsKey = "kwargs";
	
	/**
	 * Build the DIJ processing object
	 * @param prediction
	 * 	map from the rdf.yaml containing the info about DIJ processing
	 * @return the object with the needed processing
	 */
	public static DeepImageJProcessing build(Object prediction) {
		DeepImageJProcessing dijProcessing = new DeepImageJProcessing();
		dijProcessing.macros = new ArrayList<String>();
		if (prediction == null) {
			return dijProcessing;
		} else if (prediction instanceof List<?>) {
			dijProcessing.buildFromList((List<Object>) prediction);
		} else if (prediction instanceof HashMap<?, ?>) {
			dijProcessing.buildFromHashMap((HashMap<String, Object>) prediction);
		} else if (prediction instanceof HashMap<?, ?>) {
			dijProcessing.buildFromMap((Map<String, Object>) prediction);
		}
		return dijProcessing;
		
		
	}
	
	/**
	 * build the {@link #DeepImageJProcessing()} object with information from a list
	 * @param prediction
	 * 	list containing information about DIJ processing
	 */
	private void buildFromList(List<Object> prediction) {
		for (Object pp : prediction) {
			if (pp instanceof Map<?, ?>)
				buildFromMap((Map<String, Object>) pp);
			else if (pp instanceof HashMap<?, ?>)
				buildFromHashMap((HashMap<String, Object>) pp);
			
		}
	}
	
	/**
	 * build the {@link #DeepImageJProcessing()} object with information from a map
	 * @param prediction
	 * 	map containing information about DIJ processing
	 */
	private void buildFromMap(Map<String, Object> prediction) {
		if (prediction.get(specsKey) != null && prediction.get(specsKey).equals(macroKey))
			macros.add((String) prediction.get(kwargsKey));
		else if (prediction.get(specsKey) != null)
			containsJava = true;
	}
	
	/**
	 * build the {@link #DeepImageJProcessing()} object with information from a HashMap
	 * @param prediction
	 * 	map containing information about DIJ processing
	 */
	private void buildFromHashMap(HashMap<String, Object> prediction) {
		if (prediction.get(specsKey) != null && prediction.get(specsKey).equals(macroKey))
			macros.add((String) prediction.get(kwargsKey));
		else if (prediction.get(specsKey) != null)
			containsJava = true;
	}
	
	/**
	 * Whether the model requires execution of external Java code
	 * @return if the model requires execution of external Java code
	 */
	public boolean containsJava() {
		return containsJava;
	}
	
	/**
	 * return the list of macros used for processing
	 * @return the list of macros used for processing
	 */
	public List<String> getMacros() {
		return macros;
	}

}
