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
package io.bioimage.modelrunner.bioimageio.description.weights;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.versionmanagement.SupportedVersions;
import io.bioimage.modelrunner.versionmanagement.VersionStringUtils;

/**
 * The model weights information for the current model.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando 
 */
public class ModelWeight
{
	/**
	 * String representing the selected weight by the user in the DeepIcy GUI
	 */
    private String selectedEngine;
	/**
	 * String representing the selected weight version by the user in the DeepIcy GUI
	 */
    private String selectedVersion;
	/**
	 * Object containing the information about the weights selected
	 */
    private WeightFormat selectedWeights;
	/**
	 * Object containing the information about the weights loaded
	 */
    private static Map<String, WeightFormat> loadedWeights = new HashMap<String, WeightFormat>();
	/**
	 * Map with all the engines defined in the rdf.yaml
	 */
    private HashMap<String, WeightFormat> weightsDic; 
    private static String kerasIdentifier = "keras_hdf5";
    private static String onnxIdentifier = "onnx";
    private static String torchIdentifier = "pytorch_state_dict";
    private static String tfIdentifier = "tensorflow_saved_model_bundle";
    private static String tfJsIdentifier = "tensorflow_js";
    private static String torchscriptIdentifier = "torchscript";
    private static String bioengineIdentifier = "bioengine";
    /**
     * Builds a weight information element from the element map.
     * 
     * @param yamlFieldElements
     *        The element map.
     * @return The model weight information instance.
     */
    public static ModelWeight build(Map<String, Object> yamlFieldElements)
    {
        ModelWeight model = new ModelWeight();
        Set<String> weightsFormats = yamlFieldElements.keySet();
        // Reset the list with the inlcuded frameworks
        model.weightsDic = new HashMap<String, WeightFormat>();
        for (String ww : weightsFormats) {
        	Map<String, Object> weights = (Map<String, Object>) yamlFieldElements.get(ww);
        	if (weights == null)
        		continue;
	        if (ww.contentEquals(kerasIdentifier)) {
	        	KerasWeights weightsObject = new KerasWeights(weights);
	        	model.weightsDic.put(model.kerasEngineName(weightsObject), weightsObject);
	    	} else if (ww.contentEquals(onnxIdentifier)) {
	    		OnnxWeights weightsObject = new OnnxWeights(weights);
	    		model.weightsDic.put(model.onnxEngineName(weightsObject), weightsObject);
	    	} else if (ww.contentEquals(torchIdentifier)) {
	    		PytorchWeights weightsObject = new PytorchWeights(weights);
	    		model.weightsDic.put(model.torchEngineName(weightsObject), weightsObject);
	    	} else if (ww.contentEquals(tfIdentifier)) {
	    		TfWeights weightsObject = new TfWeights(weights);
	    		model.weightsDic.put(model.tfEngineName(weightsObject), weightsObject);
	    	} else if (ww.contentEquals(tfJsIdentifier)) {
	    		TfJsWeights weightsObject = new TfJsWeights(weights);
	    		model.weightsDic.put(model.tfJsEngineName(weightsObject), weightsObject);
	    	// TODO remove || ww.contentEquals("pytorch_script") as it is for very old models
	    	} else if (ww.contentEquals(torchscriptIdentifier)|| ww.contentEquals("pytorch_script")) {
	    		TorchscriptWeights weightsObject = new TorchscriptWeights(weights);
	    		model.weightsDic.put(model.torchscriptEngineName(weightsObject), weightsObject);
	    	}
        }
        return model;
    }
    
    /**
     * Get the weight object for the weight identifier provided. If it does not exist or the model
     * does not have weights in the wanted format, the method will return null
     * @param weightID
     * 	the identifier for the model weights. 
     * @return the weights in the wanted format if they are available
     */
    public WeightFormat getModelWeights(String weightID) {
    	if (this.weightsDic.get(weightID) != null)
    		return weightsDic.get(weightID);
    	String sel = this.weightsDic.keySet().stream().filter(kk -> kk.startsWith(weightID)).findFirst().orElse(null);
    	if (sel == null)
    		return null;
    	return this.weightsDic.get(sel);
    }

	/**
     * Return the corresponding weight format
     * @param weightsFormat
     * 	the tag corresponding to a particular engine
     * @param version
     * 	the version of interest
     * @return a {@link WeightFormat} object that contains the info of some weights
     * @throws IllegalArgumentException if the set of wanted weights is not supported by JDLL
     */
    public WeightFormat getSupportedWeightObject(String weightsFormat, String version) throws IllegalArgumentException
    {
    	if (weightsFormat.equals(getBioengineID()))
    		return null;
    	WeightFormat ww = weightsDic.values().stream()
    			.filter(w -> {
    				if (!w.getFramework().equals(weightsFormat)) return false;
    				if (w.getTrainingVersion().equals(
    						SupportedVersions.getClosestSupportedPythonVersion(weightsFormat, version))) return true;
					if (VersionStringUtils.areTheyTheSameVersionUntilPoint(w.getTrainingVersion(), version, 2)) return true;
					return false;
    			}).findFirst().orElse(null);
    	
    	if (ww == null) {
    		throw new IllegalArgumentException("JDLL does not support the provided weight format: "
    				+ weightsFormat + " . Supported weight formats are: " + kerasIdentifier
    				+ ", " + onnxIdentifier + ", " + torchIdentifier + ", " + tfIdentifier
    				+ ", " + tfJsIdentifier + ", " + torchscriptIdentifier + ", " + bioengineIdentifier);
    	}
    	return ww;
    }
    
    /**
     * REturn a list of the  weights supported by the model as {@link WeightFormat}
     * @return list of supported weights as {@link WeightFormat}
     */
    public List<WeightFormat> gettAllSupportedWeightObjects(){
    	return weightsDic.values().stream().collect(Collectors.toList());
    }
    
    /**
     * Return a list containing all the frameworks (engines) where the model has weights
     * @return list of supported Deep Learning frameworks with the corresponding version
     */
    public List<String> getSupportedWeightNamesAndVersion(){
    	return this.weightsDic.keySet().stream().collect(Collectors.toList());
    }
    
    /**
     * Get list with the supported Deep Learning frameworks. Does not the same framework
     * several times if it is repeated.
     * @return the list of supported engines (DL frameworks) among the ones where the model
     * 	has weights for
     */
    public List<String> getAllSuportedWeightNames() {
    	return weightsDic.entrySet().stream().
    			map(i -> i.getValue().getFramework()).
    			filter(i -> !i.equals(kerasIdentifier) && !i.equals(tfJsIdentifier)).
    			distinct().collect(Collectors.toList());
    }

	/**
	 * Get the weights format selected to make inference.
	 * For models that contain several sets of weights
	 * from different frameworks in the
	 * same model folder
	 * 
	 * @return the selected weights engine
	 */
	public String getSelectedWeightsIdentifier() {
		return selectedEngine;
	}
	
	/**
	 * GEt the training version of the selected weights
	 * @return the training version of the selected weights
	 * @throws IOException if the weights do not exist
	 */
	public String getWeightsSelectedVersion() throws IOException {
		return selectedVersion;
	}
	
	/**
	 * Return the object containing the information about the selected weights
	 * @return the yaml information about the selected weights
	 */
	public WeightFormat getSelectedWeights() {
		return this.selectedWeights;
	}

	/**
	 * Sets the Deep Learning framework of the weights of the
	 * model selected. 
	 * For models that contain several sets of weights
	 * from different frameworks in the
	 * 
	 * @param weightFormat
	 *  the format (framework) of the weights 
	 *  @param version
	 *  the version of the weight format in Python
	 */
	public void setSelectedWeightsFormat(String weightFormat, String version) {
		if (weightFormat.startsWith(kerasIdentifier)) {
			this.selectedEngine = kerasIdentifier;
		} else if (weightFormat.startsWith(onnxIdentifier)) {
			this.selectedEngine = onnxIdentifier;
		} else if (weightFormat.startsWith(torchIdentifier)) {
			this.selectedEngine = torchIdentifier;
		} else if (weightFormat.startsWith(tfIdentifier)) {
			this.selectedEngine = tfIdentifier;
		} else if (weightFormat.startsWith(tfJsIdentifier)) {
			this.selectedEngine = tfJsIdentifier;
		} else if (weightFormat.startsWith(torchscriptIdentifier)) {
			this.selectedEngine = torchscriptIdentifier;
		} else if (weightFormat.startsWith(bioengineIdentifier)) {
			this.selectedEngine = bioengineIdentifier;
		} else {
			throw new IllegalArgumentException("Unsupported Deep Learning framework for JDLL.");
		}
		this.selectedVersion = version;
		setSelectedWeights(weightFormat, version);
	}
	
	/**
	 * Set the pair of weights selected by the user by saving the object that contains the info
	 * about them
	 * @param weightFormat
	 * 	the string representing the weights selected. It should be the tag used in the Bioimage.io to
	 * 	refer to the weights: https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md
	 * @param version
	 * 	the python version of the weight format
	 */
	private void setSelectedWeights(String weightFormat, String version) {
		this.selectedWeights = getSupportedWeightObject(weightFormat, version);
	}
	
	/**
	 * Set the weights as loaded. Once there are loaded weights, no other weights of
	 * that same engine can be loaded
	 */
	public void setWeightsAsLoaded() {
		if (selectedWeights != null)
			loadedWeights.put(selectedWeights.getFramework(), selectedWeights);
	}

    /**
     * Create the name of a pair of torchscript names. The name contains the name of the weights and
     * version number. If no version is provided, "Unknown" is used as version identifier
     * @param ww
     * 	weights object for torchscript
     * @return the complete weights name
     */
    private String torchscriptEngineName(TorchscriptWeights ww) {
    	String name = torchscriptIdentifier + "_v";
    	String suffix = ww.getTrainingVersion();
    	if (suffix == null) {
    		boolean exist = true;
    		suffix = "Unknown";
    		int c = 0;
    		while (exist) {
    			if (!this.weightsDic.keySet().contains(name + suffix + c)) {
    				suffix = suffix + c;
    				exist = false;
    			}
    			c ++;
    		}
    	}
		return name + suffix;
	}

    /**
     * Create the name of a pair of torchscript names. The name contains the name of the weights and
     * version number. If no version is provided, "Unknown" is used as version identifier
     * @param ww
     * 	weights object for tensorflow javascript
     * @return the complete weights name
     */
	private String tfJsEngineName(TfJsWeights ww) {
    	String name = tfJsIdentifier + "_v";
    	String suffix = ww.getTrainingVersion();
    	if (suffix == null) {
    		boolean exist = true;
    		suffix = "Unknown";
    		int c = 0;
    		while (exist) {
    			if (!this.weightsDic.keySet().contains(name + suffix + c)) {
    				suffix = suffix + c;
    				exist = false;
    			}
    			c ++;
    		}
    	}
		return name + suffix;
	}

    /**
     * Create the name of a pair of torchscript names. The name contains the name of the weights and
     * version number. If no version is provided, "Unknown" is used as version identifier
     * @param ww
     * 	weights object for onnx
     * @return the complete weights name
     */
	private String onnxEngineName(OnnxWeights ww) {
    	String name = onnxIdentifier + "_v";
    	String suffix = ww.getTrainingVersion();
    	if (suffix == null) {
    		boolean exist = true;
    		suffix = "Unknown";
    		int c = 0;
    		while (exist) {
    			if (!this.weightsDic.keySet().contains(name + suffix + c)) {
    				suffix = suffix + c;
    				exist = false;
    			}
    			c ++;
    		}
    	}
		return name + suffix;
	}

    /**
     * Create the name of a pair of torchscript names. The name contains the name of the weights and
     * version number. If no version is provided, "Unknown" is used as version identifier
     * @param ww
     * 	weights object for tensorflow
     * @return the complete weights name
     */
	private String tfEngineName(TfWeights ww) {
    	String name = tfIdentifier + "_v";
    	String suffix = ww.getTrainingVersion();
    	if (suffix == null) {
    		boolean exist = true;
    		suffix = "Unknown";
    		int c = 0;
    		while (exist) {
    			if (!weightsDic.keySet().contains(name + suffix + c)) {
    				suffix = suffix + c;
    				exist = false;
    			}
    			c ++;
    		}
    	}
		return name + suffix;
	}

    /**
     * Create the name of a pair of torchscript names. The name contains the name of the weights and
     * version number. If no version is provided, "Unknown" is used as version identifier
     * @param ww
     * 	weights object for pytorch
     * @return the complete weights name
     */
	private String torchEngineName(PytorchWeights ww) {
    	String name = torchIdentifier + "_v";
    	String suffix = ww.getTrainingVersion();
    	if (suffix == null) {
    		boolean exist = true;
    		suffix = "Unknown";
    		int c = 0;
    		while (exist) {
    			if (!weightsDic.keySet().contains(name + suffix + c)) {
    				suffix = suffix + c;
    				exist = false;
    			}
    			c ++;
    		}
    	}
		return name + suffix;
	}

    /**
     * Create the name of a pair of torchscript names. The name contains the name of the weights and
     * version number. If no version is provided, "Unknown" is used as version identifier
     * @param ww
     * 	weights object for keras
     * @return the complete weights name
     */
	private String kerasEngineName(KerasWeights ww) {
    	String name = kerasIdentifier + "_v";
    	String suffix = ww.getTrainingVersion();
    	if (suffix == null) {
    		boolean exist = true;
    		suffix = "Unknown";
    		int c = 0;
    		while (exist) {
    			if (!weightsDic.keySet().contains(name + suffix + c)) {
    				suffix = suffix + c;
    				exist = false;
    			}
    			c ++;
    		}
    	}
		return name + suffix;
	}

	/**
	 * 
	 * @return the identifier key used for the Keras Deep Learning framework
	 */
	public static String getKerasID() {
		return kerasIdentifier;
	}

	/**
	 * 
	 * @return the identifier key used for the Onnx Deep Learning framework
	 */
	public static String getOnnxID() {
		return onnxIdentifier;
	}

	/**
	 * 
	 * @return the identifier key used for the Pytorch Deep Learning framework
	 */
	public static String getPytorchID() {
		return torchIdentifier;
	}

	/**
	 * 
	 * @return the identifier key used for the Tensorflow JS Deep Learning framework
	 */
	public static String getTensorflowJsID() {
		return tfJsIdentifier;
	}

	/**
	 * 
	 * @return the identifier key used for the Tensorflow Deep Learning framework
	 */
	public static String getTensorflowID() {
		return tfIdentifier;
	}

	/**
	 * 
	 * @return the identifier key used for the torchscript Deep Learning framework
	 */
	public static String getTorchscriptID() {
		return torchscriptIdentifier;
	}

	/**
	 * 
	 * @return the identifier key used for the Bioengine Deep Learning framework
	 */
	public static String getBioengineID() {
		return bioengineIdentifier;
	}
}
