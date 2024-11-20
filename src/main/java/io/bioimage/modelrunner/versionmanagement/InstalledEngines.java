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
    public List<DeepLearningVersion> getAll()
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
    public static List<DeepLearningVersion> getAll(String enginesPath)
    {
    	try{
    		return buildEnginesFinder(enginesPath).getAll();
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system and corresponding to the engine of interest
     * 
     * @param framework
     * 	name of the engine as defined with the engine tag at:
     * 	https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
     * 	for example tensorflow, pytorch, onnx
     * @return The available versions instance.
     */
    public List<DeepLearningVersion> getDownloadedForFramework(String framework) {
    	String searchEngine = AvailableEngines.getSupportedFrameworkTag(framework);
    	if (searchEngine == null)
    		return new ArrayList<DeepLearningVersion>();
        return getDownloadedForOS().stream()
	        .filter(v -> searchEngine.contains(v.getFramework().toLowerCase()))
			.collect(Collectors.toList());
    }	
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system, corresponding to the engine of interest and corresponding version
     * 
     * 
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
     * 	name of the engine as defined with the engine tag at:
     * 	https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
     * 	for example tensorflow, pytorch, onnx
     * @param version
     * 	version of interest of the engine
     * @return The available versions instance.
     */
    public List<DeepLearningVersion> getDownloadedForVersionedFramework(String framework, String version) {
    	String searchEngine = AvailableEngines.getSupportedFrameworkTag(framework);
    	if (searchEngine == null)
    		return new ArrayList<DeepLearningVersion>();
        return checkEngineWithArgsInstalledForOS(framework, 
        		version, null, null);
    }	
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system and corresponding to the engine of interest
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @param framework
     * 	name of the engine as defined with the engine tag at:
     * 	https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
     * 	for example tensorflow, pytorch, onnx
     * @return The available versions instance.
     */
    public static List<DeepLearningVersion> getDownloadedForFramework(String enginesPath, String framework) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedForFramework(framework);
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }	

    /**
     * Returns a list of all the downloaded {@link DeepLearningVersion}s
     * that are compatible with the operating system
     * @return list with the downloaded DeepLearningVersion
     */
    public List<DeepLearningVersion> getDownloadedForOS()
    {
        String currentPlatform = new PlatformDetection().toString();
        boolean rosetta = PlatformDetection.isUsingRosseta();
        int javaVersion = PlatformDetection.getJavaVersion();
    	List<DeepLearningVersion> versions = getAll();
    	versions = versions.stream().filter(v -> v.getOs().equals(currentPlatform)
    			&& javaVersion >= v.getMinJavaVersion()
				&& (!rosetta || (rosetta && v.getRosetta())))
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
    public static List<DeepLearningVersion> getDownloadedForOS(String enginesPath) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedForOS();
    	} catch (IOException ex) {
    		return new ArrayList<DeepLearningVersion>();
    	}
    }
    
    /**
     * Return a list of all the downloaded Python versions of the corresponding engine
     * are installed in the local machine
     * 
     * @param framework
     * 	the engine of interest
     * @return the list of deep learning versions for the given engine
     */
    public List<String> getDownloadedPythonVersionsForFramework(String framework) {
    	return getDownloadedForFramework(framework).stream()
    			.map(DeepLearningVersion::getPythonVersion).collect(Collectors.toList());
    }
    
    /**
     * Return a list of all the downloaded Python versions of the corresponding engine
     * are installed in the local machine
     * 
     * @param enginesPath
     * 	path to where the engines are stored
     * @param framework
     * 	the engine of interest
     * @return the list of deep learning versions for the given engine
     */
    public static List<String> getDownloadedPythonVersionsForFramework(String enginesPath, String framework) {
    	try{
    		return buildEnginesFinder(enginesPath).getDownloadedPythonVersionsForFramework(framework);
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
     * @param framework
     * 	the engine of interest
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @param version
	 * 	the version of interest
	 * @return the closest version to the version provided for the engine provided
	 */
    public String getMostCompatibleVersionForFramework(String framework, String version) {
		List<String> downloadedVersions = getDownloadedPythonVersionsForFramework(framework);
		return  VersionStringUtils.getMostCompatibleEngineVersion(version, downloadedVersions, framework);
    }

	/**
	 * For a specific Deep Learning framework, specified by the parameter
	 * engine, and a specific version of interest, return the closest existing
	 * version among the installed ones for the DL framework
     * @param framework
     * 	the engine of interest
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @param version
	 * 	the version of interest
     * @param enginesDir
     * 	path to where the engines are stored
	 * @return the closest version to the version provided for the engine provided
	 */
    public static String getMostCompatibleVersionForFramework(String framework, String version, String enginesDir) {
		try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			List<String> downloadedVersions = installed.getDownloadedPythonVersionsForFramework(framework);
			return  VersionStringUtils.getMostCompatibleEngineVersion(version, downloadedVersions, framework);
		} catch (IOException e) {
			return null;
		}
    }
    
    /**
     * Returns a list of all the installed engine versions that are compatible
     * with the versioned engine provided in the input parameters.
     * 
     * 
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
     * 	name of the DL framework of interest
     * @param version
     * 	original version we are looking for compatibles
     * @param enginesDir
     * 	directory where all the engines are installed
     * @return a list of all the string versions compatible
     *  with the provided versioned engine
     */
    public static List<String> getOrderedListOfCompatibleVesionsForFramework(String framework, 
    		String version, String enginesDir) {
    	try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			List<String> downloadedVersions = installed.getDownloadedPythonVersionsForFramework(framework);
			return  VersionStringUtils.getCompatibleEngineVersionsInOrder(version, downloadedVersions, framework);
		} catch (IOException e) {
			return null;
		}
    }
    
    /**
     * Check whether the engine version of interest is installed or not
     * 
     * 
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
     * 	DL framework of interest
     * @param version
     * 	version of the DL framework
     * @return true if it is installed and false otherwise
     */
    public boolean checkFrameworkVersionInstalled(String framework, String version) {
		List<String> downloadedVersions = getDownloadedPythonVersionsForFramework(framework);
		String v = downloadedVersions.stream()
				.filter(vv -> vv.equals(version)).findFirst().orElse(null);
		return v != null;
    }
    
    /**
     * Check whether the engine version of interest is installed or not
     * 
     * 
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
     * 	DL framework of interest
     * @param version
     * 	version of the DL framework
     * @param enginesDir
     * 	directory where all the engines are located
     * @return true if it is installed and false otherwise
     */
    public static boolean checkFrameworkVersionInstalled(String framework, String version, String enginesDir) {
    	try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			return installed.checkFrameworkVersionInstalled(framework, version);
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
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
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
     * @param minJavaVersion
     * 	minimum Java version that the engine needs to work. Can be null.
     * @return a list containing a list of installed engiens satisfying the constraints
     */
    public List<DeepLearningVersion> checkEngineWithArgsInstalled(String framework, 
    		String version, Boolean cpu, Boolean gpu, Boolean rosetta, Integer minJavaVersion) {
    	String searchEngine;
    	if (framework != null)
    		searchEngine = AvailableEngines.getSupportedFrameworkTag(framework);
    	else
    		searchEngine = null;
    	if (searchEngine == null && framework != null)
    		return new ArrayList<DeepLearningVersion>();
    	String javaVersion;
    	if (version != null)
    		javaVersion = SupportedVersions.getJavaVersionForPythonVersion(searchEngine, version);
    	else
    		javaVersion = null;
		List<DeepLearningVersion> filtered = getDownloadedForOS().stream().filter(vv ->{
			if (searchEngine != null && !vv.getFramework().toLowerCase().equals(searchEngine))
				return false;
			else if (javaVersion != null && !vv.getVersion().toLowerCase().equals(javaVersion.toLowerCase()))
				return false;
			else if (cpu != null && vv.getCPU() != cpu)
				return false;
			else if (gpu != null && vv.getGPU() != gpu)
				return false;
			else if (minJavaVersion != null && vv.getMinJavaVersion() > minJavaVersion)
				return false;
			else if (rosetta != null && rosetta == true && vv.getRosetta() != rosetta)
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
     * 
     * 
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
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
     * @param minJavaVersion
     * 	minimum Java version that the engine needs to work. Can be null
     * @param enginesDir
     * 	the directory where all the engines are stored. CANNOT BE NULL.
     * @return a list containing a list of installed engiens satisfying the constraints
     */
    public static List<DeepLearningVersion> checkEngineWithArgsInstalled(String framework, 
    		String version, Boolean cpu, Boolean gpu, Boolean rosetta, 
    		Integer minJavaVersion, String enginesDir) {
    	Objects.requireNonNull(enginesDir);
    	try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			return installed.checkEngineWithArgsInstalled(framework, version, 
					cpu, gpu, rosetta, minJavaVersion);
		} catch (IOException e) {
			return new ArrayList<DeepLearningVersion>();
		}
    }
    
    /**
     * Returns a list of the installed Deep Learning versions (engines) that 
     * satisfy the filters specified by the arguments of the method.
     * The method filter the engines not compatible with the current operating system
     * and Java version automatically.
     * 
     * If one of the parameter is not relevant for the search, setting it to null will deactivate
     * it for the search.
     * If we do not care whether the engine supports GPu or not, we can set 'gpu = null', and
     * the resulting list of engines will contain both engines that support and do not support GPU.
     * 
     * 
     * Note that this method looks at the framework versions specified at:
     * https://github.com/bioimage-io/JDLL/blob/main/src/main/resources/supportedVersions.json
     * 
     * This file contains all the versions for each framework supported by JDLL.
     * Note that several of the python versions point to a single Java API version. This
     * happens because not every Python version has an exact Java APi made for it. HOwever,
     * the Java API is made with enough flexibility so that is compatible with the previous
     * Python versions that do not have an API. 
     * BEcause of this, for some versions such as Tensorflow 2.8, the version that will be
     * retrieved by this method will be Tensorflow 2.10.1. The API created for Tensorflow 2.10.1
     * is completely capable of running Tensorflow 2.8.
     * 
     * @param framework
     * 	the name of the DL framework. Can be null.
     * @param version
     * 	the version of the DL framework in Python. Can be null.
     * @param cpu
     * 	whether it supports running on CPU or not. Can be null.
     * @param gpu
     * 	whether it supports running on GPU or not. Can be null.
     * @return a list containing a list of installed engiens satisfying the constraints
     */
    public List<DeepLearningVersion> checkEngineWithArgsInstalledForOS(String framework, 
    		String version, Boolean cpu, Boolean gpu) {
    	int javaVersion = PlatformDetection.getJavaVersion();
    	boolean rosetta = PlatformDetection.isUsingRosseta();
    	return checkEngineWithArgsInstalled(framework, version, cpu, gpu, 
    			rosetta, javaVersion);
    }
    
    /**
     * Returns a list of the installed Deep Learning versions (engines) that satisfy the filters
     * specified by the arguments of the method.
     * The method filter the engines not compatible with the current operating system
     * and Java version automatically.
     * 
     * If one of the parameter is not relevant for the search, setting it to null will deactivate
     * it for the search.
     * If we do not care whether the engine supports GPu or not, we can set 'gpu = null', and
     * the resulting list of engines will contain both engines that support and do not support GPU.
     * 
     * The ONLY PARAMETER THAT CANNOT BE NULL IS: enginesDir
     * 
     * @param framework
     * 	the name of the DL framework. Can be null.
     * @param version
     * 	the version of the DL framework in Python. Can be null.
     * @param cpu
     * 	whether it supports running on CPU or not. Can be null.
     * @param gpu
     * 	whether it supports running on GPU or not. Can be null.
     * @param enginesDir
     * 	the directory where all the engines are stored. CANNOT BE NULL.
     * @return a list containing a list of installed engiens satisfying the constraints
     */
    public static List<DeepLearningVersion> checkEngineWithArgsInstalledForOS(String framework, 
    		String version, Boolean cpu, Boolean gpu,  String enginesDir) {
    	Objects.requireNonNull(enginesDir);
    	try {
			InstalledEngines installed = InstalledEngines.buildEnginesFinder(enginesDir);
			return installed.checkEngineWithArgsInstalledForOS(framework, version, 
					cpu, gpu);
		} catch (IOException e) {
			return new ArrayList<DeepLearningVersion>();
		}
    }
    
}
