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
package io.bioimage.modelrunner.versionmanagement;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.system.PlatformDetection;

/**
 * Class that finds the locally installed Deep Learning frameworks
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class InstalledEngines {
	/**
	 * Path to the engines folder
	 */
	private Path path;
	/**
	 * Name of the engines folder
	 */
	private static String ENGINES_FOLDER_NAME = "engines";
	
	/**
	 * Object used to find installed DL frameworks in the software
	 * @param path
	 * 	path to the folder where all the framewoks are installed
	 * @throws IOException if the path is not a directory or if it is impossible to create 
	 * 	 a dir in that path
	 */
	private InstalledEngines(Path path) throws IOException {
		if (!path.toFile().isDirectory() && !path.toFile().mkdirs())
			throw new IOException("Unable to find or create Deep Learning engines "
					+ "directory: " + path.toString());
		this.path = path;
	}
	
	/**
	 * Constructor that will try to find the DL engines in the folder called {@link #ENGINES_FOLDER_NAME} 
	 * ("engines") inside the software directory
	 * @return an object to find the installed DL frameworks in the software
	 * @throws IOException if the path is not a directory or if it is impossible to create 
	 * 	 a dir in that path
	 */
	public static InstalledEngines buildEnginesFinder() throws IOException {
		return new InstalledEngines(Paths.get(ENGINES_FOLDER_NAME).toAbsolutePath());
	}
	
	/**
	 * Constructor that will try to find the DL engines in the folder defines by the input parameter
	 * @param enginesDirectory
	 * 	path to the folder where the installed engines should be
	 * @return an object to find the installed DL frameworks in the software
	 * @throws IOException if the path is not a directory or if it is impossible to create 
	 * 	 a dir in that path
	 */
	public static InstalledEngines buildEnginesFinder(String enginesDirectory) throws IOException {
		return new InstalledEngines(Paths.get(enginesDirectory));
	}
    
    /**
     * Get String array of engine folders in the engines folder
     * @return string array with folder names inside the engines folder
     */
    public String[] getEnginePathsAsStrings() {
    	if (!path.toFile().exists())
    		return new String[0];
    	return path.toFile().list();
    }
    
    /**
     * Get String array of engine folders in the engines folder
     * @return string array with folder names inside the engines folder
     */
    public File[] getEnginePathsAsFiles() {
    	if (!path.toFile().exists())
    		return new File[0];
    	return path.toFile().listFiles();
    }

    /**
     * Returns a list of all the downloaded {@link DeepLearningVersion}s
     * 
     * @return list with the downloaded DeepLearningVersion
     */
    public List<DeepLearningVersion> loadAllDownloaded()
    {
    	if (this.getEnginePathsAsStrings().length == 0)
    		return new ArrayList<DeepLearningVersion>();
    	List<DeepLearningVersion> versions = Arrays.stream(this.getEnginePathsAsFiles())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromFile(t);
					} catch (Exception e) {
						return null;
					}
				})
				.filter(v -> v != null && v.checkMissingJars().size() == 0).collect(Collectors.toList());
        return versions;
    }

    /**
     * Returns a list of all the downloaded {@link DeepLearningVersion}s
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @return list with the downloaded DeepLearningVersion
     */
    public static List<DeepLearningVersion> loadDownloaded(String enginesPath)
    {
    	try{
    		return buildEnginesFinder(enginesPath).loadAllDownloaded();
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system and corresponding to the engine of interest
     * 
     * @param engine
     * 	name of the engine as defined with the engine tag at:
     * 	https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
     * 	for example tensorflow, pytorch, onnx
     * @return The available versions instance.
     */
    public List<DeepLearningVersion> getDownloadedForEngine(String engine) {
    	String searchEngine = AvailableEngines.getSupportedVersionsEngineTag(engine);
    	if (searchEngine == null)
    		return new ArrayList<DeepLearningVersion>();
        return loadDownloadedForOS().stream()
	        .filter(v -> searchEngine.contains(v.getEngine().toLowerCase()))
			.collect(Collectors.toList());
    }	
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system, corresponding to the engine of interest and corresponding version
     * 
     * @param engine
     * 	name of the engine as defined with the engine tag at:
     * 	https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
     * 	for example tensorflow, pytorch, onnx
     * @param version
     * 	version of interest of the engine
     * @return The available versions instance.
     */
    public List<DeepLearningVersion> getDownloadedForVersionedEngine(String engine, String version) {
    	String searchEngine = AvailableEngines.getSupportedVersionsEngineTag(engine);
    	if (searchEngine == null)
    		return new ArrayList<DeepLearningVersion>();
        return loadDownloadedForOS().stream()
	        .filter(v -> searchEngine.contains(v.getEngine().toLowerCase())
	        		&& v.getPythonVersion().equals(version))
			.collect(Collectors.toList());
    }	
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system and corresponding to the engine of interest
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @param engine
     * 	name of the engine as defined with the engine tag at:
     * 	https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
     * 	for example tensorflow, pytorch, onnx
     * @return The available versions instance.
     */
    public static List<DeepLearningVersion> getDownloadedForEngine(String enginesPath, String engine) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedForEngine(engine);
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }	

    /**
     * Returns a list of all the downloaded {@link DeepLearningVersion}s
     * that are compatible with the operating system
     * @return list with the downloaded DeepLearningVersion
     */
    public List<DeepLearningVersion> loadDownloadedForOS()
    {
        String currentPlatform = new PlatformDetection().toString();
    	List<DeepLearningVersion> versions = loadAllDownloaded();
    	versions.stream().filter(v -> v.getOs().equals(currentPlatform)
				&& (!(new PlatformDetection().isUsingRosseta()) || v.getRosetta()))
    	.collect(Collectors.toList());
        return versions;
    }
    
    /**
     * Returns all the available installed engines compatible with the OS.
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @return List of available engines engines compatible with the OS.
     */
    public static List<DeepLearningVersion> loadDownloadedForOS(String enginesPath) {
    	try{
    		return buildEnginesFinder(enginesPath).loadDownloadedForOS();
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }
    
    /**
     * Return a list of all the downloaded Python versions of the corresponding engine
     * are installed in the local machine
     * 
     * @param engine
     * 	the engine of interest
     * @return the list of deep learning versions for the given engine
     */
    public List<String> getDownloadedPythonVersionsForEngine(String engine) {
    	return getDownloadedForEngine(engine).stream()
    			.map(DeepLearningVersion::getPythonVersion).collect(Collectors.toList());
    }
    
    /**
     * Return a list of all the downloaded Python versions of the corresponding engine
     * are installed in the local machine
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @param engine
     * 	the engine of interest
     * @return the list of deep learning versions for the given engine
     */
    public static List<String> getDownloadedPythonVersionsForEngine(String enginesPath, String engine) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedPythonVersionsForEngine(engine);
    	} catch (IOException ex) {
    		return new ArrayList<String>();
    	}
    }
    
    /**
     * 
     * @return the string path to the folder where the engines are installed
     */
    public static String getEnginesDir() {
    	return Paths.get(ENGINES_FOLDER_NAME).toAbsolutePath().toString();
    }
    
    /**
     * Statically set the string path to the folder where the engines are installed
     * @param dir
     * 	the string path to the folder where the engines are installed
     * @throws IOException if the dir does not exist
     */
    public static void setEnginesDirectory(String dir) throws IOException {
    	if (!(new File(dir).isDirectory()))
    		throw new IOException("The engines directory must correspond to an already existing folder. "
    				+ "The provided path is not  valid: " + dir);
    	ENGINES_FOLDER_NAME = dir;
    }

	/**
	 * For a specific Deep Learning framework, specified by the parameter
	 * engine, and a specific version of interest, return the closest existing
	 * version among the installed ones for the DL framework
     * @param engine
     * 	the engine of interest
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @param version
	 * 	the version of interest
	 * @return the closest version to the version provided for the engine provided
	 */
    public String getMostCompatibleVersionForEngine(String engine, String version) {
		List<String> downloadedVersions = getDownloadedPythonVersionsForEngine(engine);
		return  VersionStringUtils.getMostCompatibleEngineVersion(version, downloadedVersions, engine);
    }

	/**
	 * For a specific Deep Learning framework, specified by the parameter
	 * engine, and a specific version of interest, return the closest existing
	 * version among the installed ones for the DL framework
     * @param engine
     * 	the engine of interest
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @param version
	 * 	the version of interest
     * @param enginesDir
     * 	path to where the engines are stored
	 * @return the closest version to the version provided for the engine provided
	 */
    public static String getMostCompatibleVersionForEngine(String engine, String version, String enginesDir) {
		try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			List<String> downloadedVersions = installed.getDownloadedPythonVersionsForEngine(engine);
			return  VersionStringUtils.getMostCompatibleEngineVersion(version, downloadedVersions, engine);
		} catch (IOException e) {
			return null;
		}
    }
    
    /**
     * Returns a list of all the installed engine versions that are compatible
     * with the versioned engine provided in the input parameters.
     * 
     * @param engine
     * 	name of the DL framework of interest
     * @param version
     * 	original version we are looking for compatibles
     * @param enginesDir
     * 	directory where all the engines are installed
     * @return a list of all the string versions compatible
     *  with the provided versioned engine
     */
    public static List<String> getOrderedListOfCompatibleVesionsForEngine(String engine, 
    		String version, String enginesDir) {
    	
    }
    
    /**
     * Check whether the engine version of interest is installed or not
     * @param engine
     * 	DL framework of interest
     * @param version
     * 	version of the DL framework
     * @return true if it is installed and false otherwise
     */
    public boolean checkEngineVersionInstalled(String engine, String version) {
		List<String> downloadedVersions = getDownloadedPythonVersionsForEngine(engine);
		String v = downloadedVersions.stream()
				.filter(vv -> vv.equals(version)).findFirst().orElse(null);
		return v != null;
    }
    
    /**
     * Check whether the engine version of interest is installed or not
     * @param engine
     * 	DL framework of interest
     * @param version
     * 	version of the DL framework
     * @param enginesDir
     * 	directory where all the engines are located
     * @return true if it is installed and false otherwise
     */
    public static boolean checkEngineVersionInstalled(String engine, String version, String enginesDir) {
    	try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			return installed.checkEngineVersionInstalled(engine, version);
		} catch (IOException e) {
			return false;
		}
    }
    
    /**
     * Returns a list of the installed Deep Learning versions (engines) that satisfy the filters
     * specified by the arguments of the method.
     * 
     * If one of the parameter is not relevant for the search, setting it to null will deactivate
     * it for the search.
     * If we do not care whether the engine supports GPu or not, we can set 'gpu = null', and
     * the resulting list of engines will contain both engines that support and do not support GPU.
     * 
     * 
     * @param engine
     * 	the name of the DL framework. Can be null.
     * @param version
     * 	the version of the DL framework in Python. Can be null.
     * @param cpu
     * 	whether it supports running on CPU or not. Can be null.
     * @param gpu
     * 	whether it supports running on GPU or not. Can be null.
     * @param rosetta
     * 	only relevant for MAC M1 and M2. Whether the framework can run as x86_64 in 
     * 	arm64 based MACOS. Can be null.
     * @return a list containing a list of installed engiens satisfying the constraints
     */
    public List<DeepLearningVersion> checkEngineWithArgsInstalled(String engine, 
    		String version, Boolean cpu, Boolean gpu, Boolean rosetta) {
    	String searchEngine;
    	if (engine != null)
    		searchEngine = AvailableEngines.getSupportedVersionsEngineTag(engine);
    	else
    		searchEngine = null;
    	if (searchEngine == null && engine != null)
    		return new ArrayList<DeepLearningVersion>();
		List<DeepLearningVersion> filtered = loadDownloadedForOS().stream().filter(vv ->{
			if (searchEngine != null && !vv.getEngine().toLowerCase().equals(searchEngine))
				return false;
			else if (version != null && !vv.getPythonVersion().toLowerCase().equals(version.toLowerCase()))
				return false;
			else if (cpu != null && vv.getCPU() != cpu)
				return false;
			else if (gpu != null && vv.getGPU() != gpu)
				return false;
			else if (rosetta != null && vv.getRosetta() != rosetta)
				return false;
			return true;
		}).collect(Collectors.toList());
		return filtered;
    }
    
    /**
     * Returns a list of the installed Deep Learning versions (engines) that satisfy the filters
     * specified by the arguments of the method.
     * 
     * If one of the parameter is not relevant for the search, setting it to null will deactivate
     * it for the search.
     * If we do not care whether the engine supports GPu or not, we can set 'gpu = null', and
     * the resulting list of engines will contain both engines that support and do not support GPU.
     * 
     * The ONLY PARAMETER THAT CANNOT BE NULL IS: enginesDir
     * 
     * @param engine
     * 	the name of the DL framework. Can be null.
     * @param version
     * 	the version of the DL framework in Python. Can be null.
     * @param cpu
     * 	whether it supports running on CPU or not. Can be null.
     * @param gpu
     * 	whether it supports running on GPU or not. Can be null.
     * @param rosetta
     * 	only relevant for MAC M1 and M2. Whether the framework can run as x86_64 in 
     * 	arm64 based MACOS. Can be null.
     * @param enginesDir
     * 	the directory where all the engines are stored. CANNOT BE NULL.
     * @return a list containing a list of installed engiens satisfying the constraints
     */
    public static List<DeepLearningVersion> checkEngineWithArgsInstalled(String engine, 
    		String version, Boolean cpu, Boolean gpu, Boolean rosetta, String enginesDir) {
    	Objects.requireNonNull(enginesDir);
    	try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			return installed.checkEngineWithArgsInstalled(enginesDir, version, cpu, gpu, rosetta);
		} catch (IOException e) {
			return new ArrayList<DeepLearningVersion>();
		}
    }
    
}
