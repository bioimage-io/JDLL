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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import io.bioimage.modelrunner.bioimageio.bioengine.tensor.BioengineTensor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;

public class BioengineInterface implements DeepLearningEngineInterface {
	
	private String server;
	
	private ModelDescriptor rdf;
	
	Map<String, Object> kwargs = new HashMap<String, Object>();
	
	private String modelID;
	
	private String modelWeights;
	
	private static final String MODEL_NAME_KEY = "";
	
	private static final String INPUTS_KEY = "";
	
	private static final String ID_KEY = "";
	
	private static final String RDF_KEY = "";
	
	private static final String WW_KEY = "";
	/**
	 * Name of the default model used to run a model coming from the BioImage.io repo
	 */
	private static final String DEFAULT_BMZ_MODEL_NAME = "bioengine-model-runner";
	/**
	 * String key corresponding to the decode Json parameter in the 
	 * {@link #kwargs} map
	 */
	private static final String DECODE_JSON_KEY = "decode_json";
	/**
	 * Value corresponding to the decode Json parameter in the 
	 * {@link #kwargs} map. It is fixed.
	 */
	private static final boolean DECODE_JSON_VAL = true;
	/**
	 * Key for the input of the BioEngine corresponding to the type of serialization
	 */
	private static final String SERIALIZATION_KEY = "serialization";
	/**
	 * Value for the BioEngine serialization
	 */
	private static final String SERIALIZATION_VAL = "imjoy";

	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		
		List<Object> inputs = new ArrayList<Object>();
		for (Tensor<?> tt : inputTensors) {
			inputs.add(BioengineTensor.build(tt).getAsMap());
		}
		if (rdf.getName().equals("cellpose-python")) {
    		Map<String, Object> pp = new HashMap<String, Object>();
    		pp.put("diameter", 30);
    		inputs.add(pp);
		}
		if (modelID != null) {
			Map<String, Object> auxMap = new HashMap<String, Object>();
			auxMap.put(INPUTS_KEY, inputs);
			auxMap.put(ID_KEY, modelID);
			auxMap.put(RDF_KEY, true);
			if (modelWeights != null)
				auxMap.put(WW_KEY, modelWeights);
			ArrayList<Object> auxList = new ArrayList<Object>();
			auxList.add(auxMap);
			kwargs.put(INPUTS_KEY, auxList);
		} else {
			kwargs.put(INPUTS_KEY, inputs);
		}
		
	}

	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		try {
			rdf = ModelDescriptor.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME, false);

			if (rdf.getName().equals("cellpose-python")) {
				kwargs.put(MODEL_NAME_KEY, "cellpose-python");
				kwargs.put(DECODE_JSON_KEY, DECODE_JSON_VAL);
			} else {

				workaroundModelID();
				kwargs.put(MODEL_NAME_KEY, rdf.getName());
				kwargs.put(SERIALIZATION_KEY, SERIALIZATION_VAL);
			}
			if (true) {
				
			} else {
				
			}
		} catch (Exception e) {
			throw new LoadModelException("The rdf.yaml file for "
					+ "model at '" + modelFolder + "' cannot be read.");
		}
	}

	@Override
	public void closeModel() {
		// TODO Auto-generated method stub
		
	}
	
	public void addServer(String server) {
		this.server = server;
	}
	
	/** TODO 
	 * Workaround for BioImage.io model runner. It does not work with the full version, it
	 * only works with: major_version_/second_version
	 */
	private void workaroundModelID() {
		if (modelID == null)
			return;
		int nSubversions = modelID.length() - modelID.replace("/", "").length();
		if (nSubversions == 2) {
			modelID = modelID.substring(0, modelID.lastIndexOf("/"));
		}
	}
	
	/**
	 * Whether the name of the model corresponds to the key used to run BioImage.io models
	 * @param name
	 * 	name of a model
	 * @return true if the model name correpsonds to the Bioimage.io runner and false otherwise
	 */
	private static boolean isBioImageIoKey(String name) {
		if (name != null && name.equals(DEFAULT_BMZ_MODEL_NAME))
			return true;
		else
			return false;
	}

}
