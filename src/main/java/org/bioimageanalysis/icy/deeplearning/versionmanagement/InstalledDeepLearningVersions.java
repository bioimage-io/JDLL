package org.bioimageanalysis.icy.deeplearning.versionmanagement;

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

    
    /**
     * REturns a list of all the downloaded {@link org.bioimageanalysis.icy.deeplearning.versionmanager.version.DeepLearningVersions}
     * 
     * @return list with the downloaded DeepLearningVersion
     */
    public static List<DeepLearningVersion> loadDownloaded()
    {
    	if (DeepLearningVersionDownloader.getEnginePaths().length == 0)
    		return new ArrayList<DeepLearningVersion>();
    	List<DeepLearningVersion> versions = Arrays.stream(DeepLearningVersionDownloader.getEnginePaths())
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
    public static List<String> getDownloadedCompatiblePythonVersions() {
        String currentPlatform = new PlatformDetection().toString();
    	List<String> versions = Arrays.stream(DeepLearningVersionDownloader.getEnginePaths())
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
    public static List<DeepLearningVersion> getDownloadedCompatibleEnginesForEngine(String engine) {
    	boolean engineExists = engineKeys.keySet().stream().anyMatch(i -> i.equals(engine));
    	if (!engineExists) {
    		return new ArrayList<DeepLearningVersion>();
    	}
        String currentPlatform = new PlatformDetection().toString();
        List<DeepLearningVersion> versions = Arrays.stream(DeepLearningVersionDownloader.getEnginePaths())
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
					&& engineKeys.get(engine).toLowerCase().contains(v.getEngine().toLowerCase())
				)
                .collect(Collectors.toList());
        return versions;
    }

    /**
     * REturns a list of all the downloaded {@link org.bioimageanalysis.icy.deeplearning.versionmanager.version.DeepLearningVersions}
     * that are compatible with the operating system
     * @return list with the downloaded DeepLearningVersion
     */
    public static List<DeepLearningVersion> loadDownloadedCompatible()
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
    public static List<String> getDownloadedCompatiblePythonVersionsForEngine(String engine) {
    	boolean engineExists = engineKeys.keySet().stream().anyMatch(i -> i.equals(engine));
    	if (!engineExists) {
    		return new ArrayList<String>();
    	}
        String currentPlatform = new PlatformDetection().toString();
    	List<String> versions = Arrays.stream(DeepLearningVersionDownloader.getEnginePaths())
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
						&& engineKeys.get(engine).toLowerCase().contains(v.getEngine().toLowerCase())
						)
                .map(DeepLearningVersion::getPythonVersion)
				.collect(Collectors.toList());
        return versions;
    }
}
