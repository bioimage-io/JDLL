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
public class InstalledDeepLearningVersions {
	
	private Path path;
	
	private static String ENGINES_FOLDER_NAME = "engines";
	
	/**
	 * Object used to find installed DL frameworks in the software
	 * @param path
	 * 	path to the folder where all the framewoks are installed
	 * @throws IOException 
	 */
	private InstalledDeepLearningVersions(Path path) throws IOException {
		if (!path.toFile().isDirectory() && !path.toFile().mkdirs())
			throw new IOException("Unable to find or create Deep Learning engines "
					+ "directory: " + path.toString());
		this.path = path;
	}
	
	/**
	 * Constructor that will try to find the DL engines in the folder called {@link #ENGINES_FOLDER_NAME} 
	 * ("engines") inside the software directory
	 * @return an object to find the installed DL frameworks in the software
	 * @throws IOException 
	 */
	public static InstalledDeepLearningVersions buildEnginesFinder() throws IOException {
		return new InstalledDeepLearningVersions(Paths.get(ENGINES_FOLDER_NAME).toAbsolutePath());
	}
	
	/**
	 * Constructor that will try to find the DL engines in the folder defines by the input parameter
	 * @param enginesDirectory
	 * 	path to the folder where the installed engines should be
	 * @return an object to find the installed DL frameworks in the software
	 * @throws IOException 
	 */
	public static InstalledDeepLearningVersions buildEnginesFinder(String enginesDirectory) throws IOException {
		return new InstalledDeepLearningVersions(Paths.get(enginesDirectory));
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
						// TODO print stack trace??
						e.printStackTrace();
						System.out.println("");
						System.out.println("Folder '" + t.getName() + "' does not contain a supported Deep Learning engine version");
						return null;
					}
				})
				.filter(v -> v != null && v.checkMissingJars().size() == 0).collect(Collectors.toList());
        return versions;
    }
    
    /**
     * Return a list of all the downloaded Python versions compatible to the host system
     * 
     * @return the list of deep learning versions for the given engine
     */
    public List<String> getDownloadedCompatiblePythonVersions() {
        String currentPlatform = new PlatformDetection().toString();
    	List<String> versions = Arrays.stream(this.getEnginePathsAsFiles())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromFile(t);
					} catch (Exception e) {
						// TODO print stack trace??
						e.printStackTrace();
						System.out.println("");
						System.out.println("Folder '" + t.getName() + "' does not contain a supported Deep Learning engine version");
						return null;
					}
				})
				.filter(v -> v != null 
					&& v.checkMissingJars().size() == 0 
					&& v.getOs().equals(currentPlatform)
					)
                .map(DeepLearningVersion::getPythonVersion)
				.collect(Collectors.toList());
        return versions;
    }
    
    /**
     * Creates a list containing only downloaded Deep Learning versions compatible with
     * the current system and corresponding to the engine of interest
     * 
     * @return The available versions instance.
     */
    public List<DeepLearningVersion> getDownloadedCompatibleEnginesForEngine(String engine) {
    	boolean engineExists = AvailableDeepLearningVersions.getEngineKeys().keySet().stream().anyMatch(i -> i.equals(engine));
    	if (!engineExists) {
    		return new ArrayList<DeepLearningVersion>();
    	}
        String currentPlatform = new PlatformDetection().toString();
        List<DeepLearningVersion> versions = Arrays.stream(this.getEnginePathsAsFiles())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromFile(t);
					} catch (Exception e) {
						System.out.println("");
						System.out.println("Folder '" + t.getName() + "' does not contain a supported Deep Learning engine version");
						return null;
					}
				})
				.filter(v -> v != null && v.checkMissingJars().size() == 0 
					&& v.getOs().equals(currentPlatform) 
					&& AvailableDeepLearningVersions.getEngineKeys().get(engine).toLowerCase().contains(v.getEngine().toLowerCase())
				)
                .collect(Collectors.toList());
        return versions;
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
     * Return a list of all the downloaded Python versions of the corresponding engine
     * are installed in the local machine
     * 
     * @param engine
     * 	the engine of interest
     * @return the list of deep learning versions for the given engine
     */
    public List<String> getDownloadedCompatiblePythonVersionsForEngine(String engine) {
    	boolean engineExists = AvailableDeepLearningVersions.getEngineKeys().keySet().stream().anyMatch(i -> i.equals(engine));
    	if (!engineExists) {
    		return new ArrayList<String>();
    	}
        String currentPlatform = new PlatformDetection().toString();
    	List<String> versions = Arrays.stream(this.getEnginePathsAsFiles())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromFile(t);
					} catch (Exception e) {
						System.out.println("");
						System.out.println("Folder '" + t.getName() + "' does not contain a supported Deep Learning engine version");
						return null;
					}
				})
				.filter(v -> v != null 
						&& v.checkMissingJars().size() == 0 
						&& v.getOs().equals(currentPlatform) 
						&& AvailableDeepLearningVersions.getEngineKeys().get(engine).toLowerCase().contains(v.getEngine().toLowerCase())
						)
                .map(DeepLearningVersion::getPythonVersion)
				.collect(Collectors.toList());
        return versions;
    }
    
    public static String getInstalledVersionsDir() {
    	return Paths.get(ENGINES_FOLDER_NAME).toAbsolutePath().toString();
    }
    
    public static void setEnginesDirectory(String dir) throws IOException {
    	if (!(new File(dir).isDirectory()))
    		throw new IOException("The engines directory must correspond to an already existing folder. "
    				+ "The provided path is not  valid: " + dir);
    	ENGINES_FOLDER_NAME = dir;
    }
}
