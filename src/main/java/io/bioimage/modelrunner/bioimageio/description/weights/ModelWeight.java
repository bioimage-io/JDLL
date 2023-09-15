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
package io.bioimage.modelrunner.bioimageio.description.weights;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
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
    private static String gpuSuffix = " (supports gpu)";
    /**
     * List of all the not supported Deep Learning frameworks by DeepIcy
     */
    private static ArrayList<String> supported = 
    		new ArrayList<String>(Arrays.asList(torchscriptIdentifier, tfIdentifier, onnxIdentifier));
    /**
     * Suffix added to the engine version when the engine version is not installed
     */
    private static String missingVersion = " (please install)";
    /**
     * Suffix added to the engine version when the engine version is not installed
     */
    private static String notSupported = " (not supported)";
    /**
     * Suffix added to the engine version when another version of 
     * the same engine has already been loaded
     */
    private static String alreadyLoaded = " (restart Icy)";
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
     * Return the corresponding weight format
     * @param weightsFormat
     * 	the tag corresponding to a particular engine
     * @return a {@link WeightFormat} object that contains the info of some weights
     * @throws IOException if the set of wanted weights is not present
     */
    public WeightFormat getWeightsByIdentifier(String weightsFormat) throws IOException
    {
    	if (weightsFormat.equals(getBioengineID()))
    		return null;
    	WeightFormat ww = weightsDic.get(weightsFormat);
    	
    	if (ww == null) {
    		throw new IOException("The selected model does not contain "
    				+ "a set of " + weightsFormat + " weights.");
    	}
    	return ww;
    }
    
    /**
     * REturn a list of the supported weigths by the model as {@link WeightFormat}
     * @return list of supported weigths as {@link WeightFormat}
     */
    public List<WeightFormat> getSupportedWeights(){
    	return weightsDic.values().stream().collect(Collectors.toList());
    }
    
    /**
     * Return a list containing all the frameworks (engines) where the model has weights
     * @return list of supported Deep Learning frameworks with the corresponding version
     */
    public List<String> getEnginesListWithVersions(){
    	return this.weightsDic.keySet().stream().collect(Collectors.toList());
    }
    
    /**
     * Get list with the supported Deep Learning frameworks. Does not the same framework
     * several times if it is repeated.
     * @return the list of supported engines (DL frameworks) among the ones where the model
     * 	has weights for
     */
    public List<String> getSupportedDLFrameworks() {
    	return weightsDic.entrySet().stream().
    			map(i -> i.getValue().getFramework()).
    			distinct().collect(Collectors.toList());
    }
    
    /**
     * Return a map where the keys are engine version names plus a tag that indicates
     * whether that engine is found in the local repo and the value is the actual
     * engine version name
     * @return the map with the engines and information about whether it is downloaded or not
     */
    public Map<String, String> getDownloadedEnginesMap(){
		// Find the suffixes for each of the needed engines
    	Map<String, String> suffixes = weightsDic.keySet().stream()
    			.collect(Collectors.toMap(i -> i + findLocalEngine(i, weightsDic.get(i).getTrainingVersion()), 
    					i -> i));
    	return addCpuOrGpu(suffixes);
    }
    
    /**
     * Method to include the possibility of running in either GPU or CPU
     * if any of them is available
     * @param suffixes
     * 	map without the cpu or gpu possibility
     * @return map with the cpu and gpu possibility
     */
    private Map<String, String> addCpuOrGpu(Map<String, String> suffixes){
    	Map<String, String> nSuffixes = new HashMap<String, String>();
    	for (Entry<String, String> entry : suffixes.entrySet()) {
    		if (entry.getKey().toLowerCase().endsWith(missingVersion) 
    				|| entry.getKey().endsWith(notSupported) 
    				|| entry.getKey().endsWith(alreadyLoaded)
    				|| entry.getKey().startsWith(bioengineIdentifier)) {
    			nSuffixes.put(entry.getKey(), entry.getValue());
    			continue;
    		}
    		String engine = weightsDic.get(entry.getKey()).getFramework();
    		String trainingVersion = weightsDic.get(entry.getKey()).getTrainingVersion();
    		List<DeepLearningVersion> copiesOfVersion = new ArrayList<DeepLearningVersion>();
			try {
				InstalledEngines installed = InstalledEngines.buildEnginesFinder();
				List<String> downloadedVersions = installed.getDownloadedPythonVersionsForFramework(engine);
	    		String executionVersion = VersionStringUtils.getMostCompatibleEngineVersion(trainingVersion, downloadedVersions, engine);
	    		copiesOfVersion = installed.getDownloadedForFramework(engine)
	    		.stream().filter(v -> v.getPythonVersion().equals(executionVersion)).collect(Collectors.toList());
    		} catch (IOException e) {
				e.printStackTrace();
			}
    		for (DeepLearningVersion vv : copiesOfVersion) {
    			if (vv.getGPU() && !nSuffixes.keySet().contains(entry.getKey() + gpuSuffix))
    				nSuffixes.put(entry.getKey() + gpuSuffix, entry.getValue());
    			else if (vv.getCPU() && !nSuffixes.keySet().contains(entry.getKey()))
        				nSuffixes.put(entry.getKey(), entry.getValue());
    		}
    	}
    	return nSuffixes;
    }
    
    /**
     * REturn a String depending on if the engine is found in the local repo or not.
     * Returns an empty String "" if the engine is found, " (another version available)"
     * if the engine is found but using another version and " (missing)" if no version of
     * the engine is found
     * @param version
     * 	the name of the version of the DL engine
     * @param trainingVersion
     * 	the training version number of the corresponding engine
     * @return the corresponding String depending on the presence of the engine or not
     */
    private String findLocalEngine(String version, String trainingVersion) {
    	String engine = weightsDic.get(version).getFramework();
    	if (engine.startsWith(bioengineIdentifier)) {
    		return "";
    	} else if (!supported.contains(engine)) {
    		return notSupported;
    	}
    	List<String> downloadedVersions = new ArrayList<String>();
		try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder();
	    	downloadedVersions = installed.getDownloadedPythonVersionsForFramework(engine);
		} catch (IOException e) {
			e.printStackTrace();
		}
		String executionVersion = VersionStringUtils.getMostCompatibleEngineVersion(trainingVersion, downloadedVersions, engine);
		if (executionVersion == null)
			return missingVersion;
		// TODO consider if the version needs to be exactly the same or is it enough to share version and subversion (1.15.1 ~= 1.15.3)
		// TODO else if (!VersionManagement.areTheyTheSameVersion(executionVersion, trainingVersion)) 
		else if (!VersionStringUtils.areTheyTheSameVersionUntilPoint(executionVersion, trainingVersion, 2)) 
			return missingVersion;
		else if (loadedWeights.get(engine) != null && 
			!VersionStringUtils.areTheyTheSameVersionUntilPoint(loadedWeights.get(engine).getTrainingVersion(), trainingVersion, 2))
			return alreadyLoaded;
		else
			return "";
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
	 * @param selectedWeights
	 *  the format (framework) of the weights 
	 * @throws IOException if the weights are not found in the avaiable ones
	 */
	public void setSelectedWeightsFormat(String selectedWeights) throws IOException {
		if (selectedWeights.startsWith(kerasIdentifier)) {
			this.selectedEngine = kerasIdentifier;
		} else if (selectedWeights.startsWith(onnxIdentifier)) {
			this.selectedEngine = onnxIdentifier;
		} else if (selectedWeights.startsWith(torchIdentifier)) {
			this.selectedEngine = torchIdentifier;
		} else if (selectedWeights.startsWith(tfIdentifier)) {
			this.selectedEngine = tfIdentifier;
		} else if (selectedWeights.startsWith(tfJsIdentifier)) {
			this.selectedEngine = tfJsIdentifier;
		} else if (selectedWeights.startsWith(torchscriptIdentifier)) {
			this.selectedEngine = torchscriptIdentifier;
		} else if (selectedWeights.startsWith(bioengineIdentifier)) {
			this.selectedEngine = bioengineIdentifier;
		} else {
			throw new IllegalArgumentException("Unsupported Deep Learning framework for JDLL.");
		}
		setSelectedVersion(selectedWeights);
		setSelectedWeights(selectedWeights);
	}
	
	/**
	 * Sets the Deep Learning engine version selected by the user
	 * @param selectedWeights
	 * 	the selected weights format and version by the user in the GUI
	 */
	private void setSelectedVersion(String selectedWeights) {
		if (selectedWeights.equals(bioengineIdentifier)) {
			this.selectedVersion =  "";
			return;
		}
		String preffix = this.selectedEngine + "_v";
		this.selectedVersion = selectedWeights.substring(preffix.length());		
	}
	
	/**
	 * Set the pair of weights selected by the user by saving the object that contains the info
	 * about them
	 * @param selectedWeights
	 * 	the string selected by the user as weights
	 * @throws IOException if the weights are not found in the avaiable ones
	 */
	private void setSelectedWeights(String selectedWeights) throws IOException {
		this.selectedWeights = getWeightsByIdentifier(selectedWeights);
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
	 * REturn the tag used to identify Deep Learning engines that are not present
	 * in the local engines repo
	 * @return missing tag used to identify non installed engines
	 */
	public static String getMissingEngineTag() {
		return missingVersion;
	}
	
	/**
	 * REturn the tag used to identify Deep Learning engines that are not supported by DeepIcy
	 * @return tag used to identify which are the engines not supported by the library
	 */
	public static String getNotSupportedEngineTag() {
		return notSupported;
	}
	
	/**
	 * REturn the tag used to identify Deep Learning engines where another
	 * version oof the engine has been loaded
	 * @return whether the DL framework has already been loaded or not
	 */
	public static String getAlreadyLoadedEngineTag() {
		return alreadyLoaded;
	}
	
	/**
	 * REturn the tag used to identify Deep Learning engines that support GPU
	 * @return te suffix used to identify if a engine uses GPU or not
	 */
	public static String getGPUSuffix() {
		return gpuSuffix;
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
