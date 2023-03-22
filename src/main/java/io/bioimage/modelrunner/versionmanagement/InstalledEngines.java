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

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
    public List<DeepLearningVersion> loadDownloaded()
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
    		return buildEnginesFinder(enginesPath).loadDownloaded();
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
    public List<DeepLearningVersion> getDownloadedCompatibleEnginesForEngine(String engine) {
    	String searchEngine = AvailableEngines.getSupportedVersionsEngineTag(engine);
    	if (searchEngine == null)
    		return new ArrayList<DeepLearningVersion>();
        return loadDownloadedCompatible().stream()
	        .filter(v -> searchEngine.contains(v.getEngine().toLowerCase()))
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
    public static List<DeepLearningVersion> getDownloadedCompatibleEnginesForEngine(String enginesPath, String engine) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedCompatibleEnginesForEngine(engine);
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }	

    /**
     * Returns a list of all the downloaded {@link DeepLearningVersion}s
     * that are compatible with the operating system
     * @return list with the downloaded DeepLearningVersion
     */
    public List<DeepLearningVersion> loadDownloadedCompatible()
    {
        String currentPlatform = new PlatformDetection().toString();
    	List<DeepLearningVersion> versions = loadDownloaded();
    	versions.stream().filter(v -> v.getOs().equals(currentPlatform)).collect(Collectors.toList());
        return versions;
    }
    
    /**
     * Returns all the available installed engines compatible with the OS.
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @return List of available engines engines compatible with the OS.
     */
    public static List<DeepLearningVersion> loadDownloadedCompatible(String enginesPath) {
    	try{
    		return buildEnginesFinder(enginesPath).loadDownloadedCompatible();
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
    public List<String> getDownloadedCompatiblePythonVersionsForEngine(String engine) {
    	return getDownloadedCompatibleEnginesForEngine(engine).stream()
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
    public static List<String> getDownloadedCompatiblePythonVersionsForEngine(String enginesPath, String engine) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedCompatiblePythonVersionsForEngine(engine);
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
		List<String> downloadedVersions = getDownloadedCompatiblePythonVersionsForEngine(engine);
		return  VersionStringUtils.getMostCompatibleEngineVersion(version, downloadedVersions, engine);
    }

	/**
	 * For a specific Deep Learning framework, specified by the parameter
	 * engine, and a specific version of interest, return the closest existing
	 * version among the installed ones for the DL framework
     * @param enginesDir
     * 	path to where the engines are stored
     * @param engine
     * 	the engine of interest
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @param version
	 * 	the version of interest
	 * @return the closest version to the version provided for the engine provided
	 */
    public static String getMostCompatibleVersionForEngine(String enginesDir, String engine, String version) {
		try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			List<String> downloadedVersions = installed.getDownloadedCompatiblePythonVersionsForEngine(engine);
			return  VersionStringUtils.getMostCompatibleEngineVersion(version, downloadedVersions, engine);
		} catch (IOException e) {
			return null;
		}
    }
}
