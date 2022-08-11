package org.bioimageanalysis.icy.deeplearning.versionmanagement;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.bioimageanalysis.icy.deeplearning.system.PlatformDetection;

/**
 * Class that finds the locally installed Deep Learning frameworks
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class InstalledDeepLearningVersions {
	
	private Path path;
	
	private InstalledDeepLearningVersions(Path path) {
		if (!path.toFile().isDirectory())
			throw new IllegalArgumentException("");
		this.path = path;
	}
	
	public static InstalledDeepLearningVersions buildEnginesFinder() {
		Path downloadsPath = Paths.get(".", "engines").toAbsolutePath();
	}
	
	public static InstalledDeepLearningVersions buildEnginesFinder(String enginesDirectory) {
		
	}
    
    /**
     * Get String array of engine folders in the engines folder
     * @return string array with folder names inside the engines folder
     */
    public String[] getEnginePaths() {
    	if (!path.toFile().exists())
    		return new String[0];
    	return path.toFile().list();
    }

    /**
     * REturns a list of all the downloaded {@link org.bioimageanalysis.icy.deeplearning.versionmanager.version.DeepLearningVersions}
     * 
     * @return list with the downloaded DeepLearningVersion
     */
    public List<DeepLearningVersion> loadDownloaded()
    {
    	if (this.getEnginePaths().length == 0)
    		return new ArrayList<DeepLearningVersion>();
    	List<DeepLearningVersion> versions = Arrays.stream(this.getEnginePaths())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromString(t);
					} catch (Exception e) {
						// TODO print stack trace??
						e.printStackTrace();
						System.out.println("");
						System.out.println("Folder '" + new File(t).getName() + "' does not contain a supported Deep Learning engine version");
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
    	List<String> versions = Arrays.stream(this.getEnginePaths())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromString(t);
					} catch (Exception e) {
						// TODO print stack trace??
						e.printStackTrace();
						System.out.println("");
						System.out.println("Folder '" + new File(t).getName() + "' does not contain a supported Deep Learning engine version");
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
        List<DeepLearningVersion> versions = Arrays.stream(this.getEnginePaths())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromString(t);
					} catch (Exception e) {
						System.out.println("");
						System.out.println("Folder '" + new File(t).getName() + "' does not contain a supported Deep Learning engine version");
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
     * REturns a list of all the downloaded {@link org.bioimageanalysis.icy.deeplearning.versionmanager.version.DeepLearningVersions}
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
    	List<String> versions = Arrays.stream(this.getEnginePaths())
    			.map(t -> {
					try {
						return DeepLearningVersion.fromString(t);
					} catch (Exception e) {
						System.out.println("");
						System.out.println("Folder '" + new File(t).getName() + "' does not contain a supported Deep Learning engine version");
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
}
