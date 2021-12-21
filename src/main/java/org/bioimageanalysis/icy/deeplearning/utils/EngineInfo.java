package org.bioimageanalysis.icy.deeplearning.utils;

import java.io.File;

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
	 * Object containing all the supported versions for the selected
	 * Deep Learning framework (engine)
	 */
	private SupportedVersions supportedVersions;
	/**
	 * Variable containing the name used to refer to Tensorflow
	 * in the program
	 */
	private static String tensorflowEngineName = "tensorflow";
	/**
	 * Variable containing the name used to refer to Pytorch
	 * in the program
	 */
	private static String pytorchEngineName = "pytorch";
	
	private EngineInfo(String engine, String version, String jarsDirectory) {
		setEngine(engine);
		this.version = version;
		this.jarsDirectory = jarsDirectory;
		this.os = findOs();
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
	 * 	directory the JARs needed to launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a 
	 * 	Deep learning framework
	 */
	public static EngineInfo defineDLEngine(String engine, String version, String jarsDirectory) {
		return new EngineInfo(engine, version, jarsDirectory);
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
}
