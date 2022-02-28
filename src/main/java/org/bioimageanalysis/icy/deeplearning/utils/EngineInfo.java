package org.bioimageanalysis.icy.deeplearning.utils;

import java.io.File;
import java.util.Objects;

import org.bioimageanalysis.icy.system.PlatformDetection;

/**
 * Class to create an object that contains all the information about a 
 * Deep Learning framework (engine) that is needed to launch the engine
 * in an independent ClassLoader
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class EngineInfo {
	/**
	 * Deep Learning framework (engine)., for example Pytorch or Tensorflow 
	 */
	private String engine;
	/**
	 * Version of the Deep Learning framework (engine) used to train the 
	 * model. This version usually corresponds to the Python API.
	 */
	private String version;
	/**
	 * Version of the Deep Learning framework (engine) of the Java API, which
	 * is going to be used to load the model.
	 */
	private String versionJava;
	/**
	 * Whether the version wanted is used for CPU, GPU or MKL
	 */
	private String engineMachine;
	/**
	 * Operating system of the machine where the plugin is running
	 */
	private String os;
	/**
	 * Tags to open the model, only needed for Tensorflow
	 */
	private String tfTag;
	private String tfSigDef;
	/**
	 * Directory where the all the jars needed to load a version
	 * are stored. It should be organized in the following way:
	 * - jarsDirectory
	 * 	- engineName1_engineJavaVersion1_engineOs1_engineMachine1
	 * 		- engine1Jar1.jar
	 * 		- engine1Jar2.jar
	 * 		- ...
	 * 		- engine1JarN.jar
	 * 	- engineName2_engineJavaVersion2_engineOs2_engineMachine2
	 * 		- engine2Jar1.jar
	 * 		- engine2Jar2.jar
	 * 		- ...
	 * 		- engine2JarN.jar
	 */
	private String jarsDirectory;
	/**
	 * If the JARs directory is not going to change during the exction of the program
	 */
	private static String STATIC_JARS_DIRECTORY;
	/**
	 * Object containing all the supported versions for the selected
	 * Deep Learning framework (engine)
	 */
	private SupportedVersions supportedVersions;
	/**
	 * TODO change by official bioimageio tags?
	 * Variable containing the name used to refer to Tensorflow
	 * in the program
	 */
	private static String tensorflowEngineName = "tensorflow";
	/**
	 * Variable containing the name used to refer to Pytorch
	 * in the program
	 */
	private static String pytorchEngineName = "pytorch";
	/**
	 * Variable containing the name used to refer to Tensorflow
	 * in the program
	 */
	private static String tensorflowJavaBioimageioTag = "tensorflow_saved_model_bundle";
	/**
	 * Variable containing the name used to refer to Pytorch
	 * in the program
	 */
	private static String pytorchJavaBioimageioTag = "torchscript";
	/**
	 * Variable that stores which version of Tensorflow 1
	 * has been already loaded to avoid errors for loading
	 * two different native libraries in the same namespace
	 */
	private static String loadedTf1Version;
	/**
	 * Variable that stores which version of Tensorflow 2
	 * has been already loaded to avoid errors for loading
	 * two different native libraries in the same namespace
	 */
	private static String loadedTf2Version;
	/**
	 * Variable that stores which version of Pytorch
	 * has been already loaded to avoid errors for loading
	 * two different native libraries in the same namespace
	 */
	private static String loadedPytorchVersion;

	
	/**
	 * Information needed to know how to launch the corresponding 
	 * Deep Learning framework
	 * 
	 * @param engine
	 * 	name of the Deep Learning framework (engine). For example: Pytorch, Tensorflow....
	 * @param version
	 * 	version of the training Deep Learning framework (engine)
	 * @param jarsDirectory
	 * 	directory where the folder containing the JARs needed to launch the 
	 * 	corresponding engine are located
	 * @return an object containing all the information needed to launch a 
	 * 	Deep learning framework
	 */
	private EngineInfo(String engine, String version, String jarsDirectory) {
		Objects.requireNonNull(engine, "The Deep Learning engine should not be null.");
		Objects.requireNonNull(version, "The Deep Learning engine version should not be null.");
		Objects.requireNonNull(jarsDirectory, "The Jars directory should not be null.");
		setEngine(engine);
		this.version = version;
		this.jarsDirectory = jarsDirectory;
		this.os = new PlatformDetection().toString();
		this.engineMachine = findCpuGpuOrMkl();
		setSupportedVersions();
		this.versionJava = findCorrespondingJavaVersion();
		
	}
	
	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * in the program
	 * 
	 * @param engine
	 * 	name of the Deep Learning framework (engine). For example: Pytorch, Tensorflow....
	 * @param version
	 * 	version of the training Deep Learning framework (engine)
	 * @param jarsDirectory
	 * 	directory where the folder containing the JARs needed to launch the 
	 * 	corresponding engine are located
	 * @return an object containing all the information needed to launch a 
	 * 	Deep learning framework
	 */
	public static EngineInfo defineDLEngine(String engine, String version, String jarsDirectory) {
		return new EngineInfo(engine, version, jarsDirectory);
	}
	
	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * in the program
	 * 
	 * @param engine
	 * 	name of the Deep Learning framework (engine). For example: Pytorch, Tensorflow....
	 * @param version
	 * 	version of the training Deep Learning framework (engine)
	 * @return an object containing all the information needed to launch a 
	 * 	Deep learning framework
	 */
	public static EngineInfo defineDLEngine(String engine, String version) {
		Objects.requireNonNull(STATIC_JARS_DIRECTORY, "The Jars directory should not be null.");
		return new EngineInfo(engine, version, STATIC_JARS_DIRECTORY);
	}
	
	/**
	 * Retrieve the complete name of the Deep Learning framework (engine)
	 * version. It includes the engine, the Java version, the os and the the machine.
	 * It should be the name of the directory where the needed JARs are stored.
	 * @return a String with all the characteristics of the Deep Learning engine
	 * @throws
	 * 	Exception throws exception if the Deep Learning framework is not
	 * 	fully defined
	 */
	public String getDeepLearningVersionJarsDirectory() throws Exception {
		if (engine != null || version != null) {
			String vv = this.engine + "_" + this.versionJava + "_" + this.os + "_" + this.engineMachine;
			return this.jarsDirectory + File.separator + vv;
		} else {
			// TODO create exception
			throw new Exception();
		}
	}
	
	/**
	 * Method that finds if the machine has a GPU or not available,
	 * or if it is MKL. If there is no GPU or MKL, it returns "cpu",
	 * if there is MKL, "mkl", if there is GPU, "gpu", and if there is
	 * everything "gpu".
	 * @return whether the program runs on CPU, GPU or MKL
	 */
	public static String findCpuGpuOrMkl() {
		return "cpu";
	}
	
	/**
	 * Method that finds the operating system where the program is running
	 * @return the operating system
	 */
	public static String findOs() {
		String os = null;
		String operSys = System.getProperty("os.name").toLowerCase();
        if (operSys.contains("win")) {
            os = "windows";
        } else if (operSys.contains("nix") || operSys.contains("nux")
                || operSys.contains("aix")) {
            os = "linux";
        } else if (operSys.contains("mac")) {
            os = "mac";
        } else if (operSys.contains("sunos")) {
            os = "solaris";
        }
        return os;
	}
	
	/**
	 * Finds the version of Deep Learning framework (engine) equivalent or compatible
	 * with the one used to train the model. This is done because sometimes APIs for
	 * different languages are named differently
	 * @return
	 * 	corresponding compatible version of the DL framework Java version
	 */
	public String findCorrespondingJavaVersion() {
		try {
			return this.supportedVersions.getCorrespondingJavaVersion(this.version);
		} catch (Exception e) {
			// TODO Refine exception
			e.printStackTrace();
			return "";
		}
	}
	
	/**
	 * Create the object that contains all the supported versions for the 
	 * Deep Learning framework (engine) selected
	 */
	private void setSupportedVersions() {
		this.supportedVersions = new SupportedVersions(this.engine);
	}
	
	/**
	 * Set the Deep Learning framework (engine) of the model
	 * @param engine
	 * 	Deep Learning framework used for the model
	 */
	public void setEngine(String engine) {
		if (engine.toLowerCase().contentEquals(tensorflowEngineName))
			this.engine = tensorflowEngineName;
		else if (engine.toLowerCase().contentEquals(pytorchEngineName))
			this.engine = pytorchEngineName;
	}
	
	/**
	 * Set the directory where the program will look for the 
	 * Deep Learning framework jars
	 * See {@link jarsDirectory} for more explanation
	 * @param jarsDirectory
	 * 	directory where all the folders containing the JARs are
	 * stored
	 */
	public void setJarsDirectory(String jarsDirectory) {
		this.jarsDirectory = jarsDirectory;
	}
	
	/**
	 * Return the String path to the directory where all the jars to load
	 * a Deep Learning framework (engine) are stored.
	 * See {@link jarsDirectory} for more explanation
	 * @return String path to the directory where all the jars are stored
	 */
	public String getJarsDirectory() {
		return this.jarsDirectory;
	}
	
	/**
	 * Set the tags needed to load a Tensorflow model. These fields
	 * are useless for other models
	 * @param tag
	 * @param sigDef
	 */
	public void setTags(String tag, String sigDef) {
		if (this.engine.toLowerCase().contentEquals(tensorflowEngineName)) {
			this.tfTag = tag;
			this.tfSigDef = sigDef;
		}
	}
	
	/**
	 * Get Tensorflow Signature Definition to open model
	 * @return Tensorflow Signature Definition to open model
	 */
	public String getTfSigDef() {
		return this.tfSigDef;
	}
	
	/**
	 * Get Tensorflow tag to open model
	 * @return Tensorflow tag to open model
	 */
	public String getTfTag() {
		return this.tfTag;
	}
	
	/**
	 * Return version of the Deep Learning framework (engine).
	 * The version corresponds to the one used to train the network.
	 * @return version of the engine where the model was trained
	 */
	public String getVersion() {
		return this.version;
	}
	
	/**
	 * Return version of the Deep Learning framework (engine).
	 * The version corresponds to the one used to run the model in Java.
	 * @return version of the engine where the model was trained
	 */
	public String getJavaVersion() {
		return this.versionJava;
	}
	
	/**
	 * Returns the machine that the jar is developed for.
	 * If there is no MKL or GPU available, returns "cpu",
	 * if there is Mkl and no GPU, "mkl" and if there is GPU,
	 * always "gpu"
	 * @return the available machines for the engine
	 */
	public String getEngineMachine() {
		return this.engineMachine;
	}
	
	/**
	 * Get the operating system of the machine
	 * @return the operation system. If can be either: "windows",
	 * "linux", "solaris" or "mac"
	 */
	public String getOS() {
		return this.os;
	}
	
	/**
	 * Sets which versions have already been loaed to avoid
	 * errors trying to load another version from the same engine,
	 * which always crashes the application
	 */
	public void setLoadedVersion() {
		if (this.engine.equals(tensorflowEngineName) 
				&& this.version.startsWith("1")) {
			loadedTf1Version = this.version;
		} else if (this.engine.equals(tensorflowEngineName) 
				&& this.version.startsWith("2")) {
			loadedTf2Version = this.version;
		} else if (this.engine.equals(pytorchEngineName)) {
			loadedPytorchVersion = this.version;
		}
	}
	
	/**
	 * REturns which versions have been already been loaded to 
	 * avoid errors of overlapping versions
	 * @param engine
	 * 	the Deep Learning framework of interest
	 * @param version
	 * 	the Deep LEarning version of interest
	 * @return the loaded version of the selected engine or null if 
	 * 	no version has been loaded
	 * @throws IllegalArgumentException if the engine is not supported yet
	 */
	public static String getLoadedVersions(String engine, String version)
				throws IllegalArgumentException {
		if (engine.toLowerCase().contains(tensorflowEngineName)  
				&& version.startsWith("1")) {
			return loadedTf1Version;
		} else if (engine.toLowerCase().contains(tensorflowEngineName)  
				&& version.startsWith("2")) {
			return loadedTf2Version;
		} else if (engine.toLowerCase().contains(pytorchEngineName)) {
			return loadedPytorchVersion;
		} else {
			throw new IllegalArgumentException("The selected engine '" 
								+ engine + "' is not supported yet.");
		}
	}
	
	/**
	 * Set in a static manner the {@link #STATIC_JARS_DIRECTORY} if it is not going
	 * to change during the execution of the program
	 * @param jarsDirectory
	 * 	the permanent jars directory
	 */
	public static void setStaticJarsDirectory(String jarsDirectory) {
		STATIC_JARS_DIRECTORY = jarsDirectory;
	}
	
	/**
	 * Method that returns the name with which Tensorflow is defined needed at
	 * some points to differentiate between tf1 and tf2
	 * @return the String used for tensorflow 
	 */
	public static String getTfKey() {
		return tensorflowEngineName;
	}
}
