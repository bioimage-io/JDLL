/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package io.bioimage.modelrunner.versionmanagement;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.system.PlatformDetection;

import com.google.gson.Gson;

/**
 * TODO remove unused methods
 * Holds the list of available TensorFlow versions read from the JSON file.
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos Garcia Lopez de Haro
 */
public class AvailableDeepLearningVersions
{
	/**
	 * HashMap that translates the keys used to name Deep Learning engines
	 * in the rdf.yaml to the ones used by the Deep Learning manager
	 */
	private static HashMap<String, String> engineKeys;
	static {
		engineKeys = new HashMap<String, String>();
		engineKeys.put(EngineInfo.getBioimageioPytorchKey(), EngineInfo.getPytorchKey());
		engineKeys.put(EngineInfo.getBioimageioTfKey(), EngineInfo.getTensorflowKey());
		engineKeys.put(EngineInfo.getBioimageioOnnxKey(), EngineInfo.getOnnxKey());
		engineKeys.put(EngineInfo.getBioimageioKerasKey(), EngineInfo.getKerasKey());
	}
	
	/**
	 * MEthod that returns all the possible names for each the DL engines existing at the moment
	 * @return the map that associates the bioimage.io key for each weight with the weights of 
	 * each of the frameworks as defined by the engine tag at:
	 * https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 */
	public static HashMap<String, String> getEngineKeys(){
		return engineKeys;
	}
	
    /**
     * Creates an instance containing only Deep Learning versions compatible with the current system.
     * 
     * @return The available versions instance.
     */
    public static AvailableDeepLearningVersions loadCompatibleOnly()
    {
        AvailableDeepLearningVersions availableVersions = load();
        String currentPlatform = new PlatformDetection().toString();
        availableVersions.setVersions(availableVersions.getVersions().stream()
                .filter(v -> v.getOs().equals(currentPlatform)).collect(Collectors.toList()));
        availableVersions.getVersions().stream().forEach(x -> x.setEnginesDir());
        return availableVersions;
    }
    
    /**
     * Remove the python repeated versions from the list of versions. There
     * can be several JAva versions that reproduce the same Python version
     * @param versions
     * 	original list of versions that might contain repeated Python versions
     * @return list of versions without repeated Python versions
     */
    public static List<DeepLearningVersion> removeRepeatedPythonVersions(List<DeepLearningVersion> versions) {
    	List<DeepLearningVersion> nVersions = new ArrayList<DeepLearningVersion>();
    	for (DeepLearningVersion vv : versions) {
    		List<DeepLearningVersion> coinc = nVersions.stream()
    				.filter(v -> vv.getPythonVersion().equals(v.getPythonVersion()) 
    						&& vv.getCPU() == v.getCPU() && vv.getGPU() == v.getGPU())
    				.collect(Collectors.toList());
    		if (coinc.size() != 0 && coinc.get(0).isJavaVersionBigger(vv))
    			continue;
    		else if (coinc.size() != 0)
    			nVersions.remove(coinc.get(0));
    		nVersions.add(vv);
    	}
    	return nVersions;
    }
    
    /**
     * Return a list of all the Python versions compatible to the host system
     * 
     * @return the list of deep learning versions for the given engine
     */
    public static List<String> getAvailableCompatiblePythonVersions() {
        AvailableDeepLearningVersions availableVersions = load();
        String currentPlatform = new PlatformDetection().toString();
        List<String> availablePythonVersions = availableVersions.getVersions().stream()
				                .filter(v -> v.getOs().equals(currentPlatform))
				                .map(DeepLearningVersion::getPythonVersion)
				                .collect(Collectors.toList());
        return availablePythonVersions;
    }
    
    /**
     * Creates an instance containing only Deep Learning versions compatible with
     * the current system and corresponding to the version of interest
     * 
     * @param engine
     * 	engine name as specified by the bioimage.io, defined at
     * 	https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md
     * @return The available versions instance.
     */
    public static AvailableDeepLearningVersions getAvailableVersionsForEngine(String engine) {
    	boolean engineExists = engineKeys.keySet().stream().anyMatch(i -> i.equals(engine));
    	AvailableDeepLearningVersions availableVersions = new AvailableDeepLearningVersions();
    	if (!engineExists) {
    		availableVersions.setVersions(new ArrayList<DeepLearningVersion>());
    		return availableVersions;
    	}
        availableVersions = load();
        String currentPlatform = new PlatformDetection().toString();
        availableVersions.setVersions(availableVersions.getVersions().stream()
                .filter(v -> v.getOs().equals(currentPlatform) 
                		&& engineKeys.get(engine).toLowerCase().contains(v.getEngine().toLowerCase())
                		)
                .collect(Collectors.toList()));
        return availableVersions;
    }
    
    /**
     * Return a list of all the Python versions of the corresponding engine
     * are installed in the local machine
     * 
     * @param engine
     * 	the engine of interest
     * @return the list of deep learning versions for the given engine
     */
    public static List<String> getAvailableCompatiblePythonVersionsForEngine(String engine) {
    	boolean engineExists = engineKeys.keySet().stream().anyMatch(i -> i.equals(engine));
    	if (!engineExists) {
    		return new ArrayList<String>();
    	}
    	AvailableDeepLearningVersions availableVersions = load();
        String currentPlatform = new PlatformDetection().toString();
        List<String> availablePythonVersions = availableVersions.getVersions().stream()
                .filter(v -> v.getOs().equals(currentPlatform) && engineKeys.get(engine).toLowerCase().contains(v.getEngine().toLowerCase()))
                .map(DeepLearningVersion::getPythonVersion)
                .collect(Collectors.toList());
        return availablePythonVersions;
    }

    /**
     * Loads all available versions from {@code availableTFVersion.json} file.
     * 
     * @return The instance of all available versions.
     */
    public static AvailableDeepLearningVersions load()
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                AvailableDeepLearningVersions.class.getClassLoader().getResourceAsStream("availableDLVersions.json")));
        Gson g = new Gson();
        AvailableDeepLearningVersions availableVersions = g.fromJson(br, AvailableDeepLearningVersions.class);
        return availableVersions;
    }

    private List<DeepLearningVersion> versions;

    /**
     * Retrieves the list of available TF versions.
     * 
     * @return The list of TF versions in this instance.
     */
    public List<DeepLearningVersion> getVersions()
    {
        return versions;
    }

    /**
     * Sets the list of versions available in this instance.
     * 
     * @param versions
     *        The versions to be available in this instance.
     */
    public void setVersions(List<DeepLearningVersion> versions)
    {
        this.versions = versions;
    }
    
    /**
     * Check if an engine is supported by the dl-modelrunner or not
     * @param framework
	 * 	DL framework as specified by the Bioimage.io model zoo ()https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md)
	 * @param version
	 * 	the version of the framework
	 * @param cpu
	 * 	whether the engine supports cpu or not
	 * @param gpu
	 * 	whether the engine supports gpu or not
     * @return true if the engine exists and false otherwise
     */
    public static boolean isEngineSupported(String framework, String version, boolean cpu, boolean gpu) {
    	Map<String, String> enginesMap = getEngineKeys().entrySet().stream()
    			.collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
    	if (enginesMap.get(framework) != null)
			framework = AvailableDeepLearningVersions.getEngineKeys().get(framework);
    	DeepLearningVersion engine = AvailableDeepLearningVersions.getAvailableVersionsForEngine(framework).getVersions()
				.stream().filter(v -> v.getPythonVersion().equals(version) 
						&& v.getOs().equals(new PlatformDetection().toString())
						&& v.getCPU() == cpu
						&& v.getGPU() == gpu).findFirst().orElse(null);
		if (engine == null) 
			return false;
		return true;
    }

}
