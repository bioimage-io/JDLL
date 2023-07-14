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
package io.bioimage.modelrunner.engine;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.versionmanagement.AvailableEngines;
import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import io.bioimage.modelrunner.versionmanagement.SupportedVersions;
import io.bioimage.modelrunner.versionmanagement.VersionStringUtils;

/**
 * Class to create an object that contains all the information about a Deep
 * Learning framework (engine) that is needed to launch the engine in an
 * independent ClassLoader
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class EngineInfo
{
	/**
	 * Deep Learning framework (engine)., for example Pytorch or Tensorflow
	 */
	private String engine;

	/**
	 * Version of the Deep Learning framework (engine) used to train the model.
	 * This version usually corresponds to the Python API.
	 */
	private String version;

	/**
	 * Version of the Deep Learning framework (engine) of the Java API, which is
	 * going to be used to load the model.
	 */
	private String versionJava;

	/**
	 * True if the engine supports gpu or false otherwise. False by default.
	 */
	private boolean gpu = false;

	/**
	 * True if the engine supports cpu or false otherwise. True by default
	 * because at the moment of development all engines support cpu.
	 */
	private boolean cpu = true;

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
	 * Directory where the all the jars needed to load a version are stored. It
	 * should be organized in the following way:
	 * <pre>
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
	 * </pre>
	 */
	private String jarsDirectory;
	/**
	 * URL where the wanted Bioengine is hosted
	 */
	private String serverURL;
	
	/**
	 * Error message that will be thrown if the user tries to load an engine that cannot
	 * be loaded together with the engines that are already loaded
	 */
	 private static final String ENGINE_ERR = "The program will not be able to load '%s %s' because another "
	 		+ "version (%s) of the same framework has already been loaded." + System.lineSeparator()
            + "If loading the wanted version (%s) is strictly necessary, please "
            + "restart the JVM. However, if the previously loaded version (%s) "
            + "can be used, " + System.lineSeparator()
            + "please call EngineInfo.defineCompatibleDLEngine(...) to avoid restarting.";

	/**
	 * If the JARs directory is not going to change during the execution of the
	 * program
	 */
	private static String STATIC_JARS_DIRECTORY;

	/**
	 * Object containing all the supported versions for the selected Deep
	 * Learning framework (engine)
	 */
	private SupportedVersions supportedVersions;

	/**
	 * TODO change by official bioimageio tags? Variable containing the name
	 * used to refer to Tensorflow in the program
	 */
	private static final String TENSORFLOW_ENGINE_NAME = "tensorflow";

	/**
	 * Variable containing the name used to refer to Pytorch in the program
	 */
	private static final String PYTORCH_ENGINE_NAME = "pytorch";

	/**
	 * Variable containing the name used to refer to Onnx in the program
	 */
	private static final String ONNX_ENGINE_NAME = "onnx";

	/**
	 * Variable containing the name used to refer to Keras in the program
	 */
	private static final String KERAS_ENGINE_NAME = "keras";

	/**
	 * Variable containing the name used to refer to Tensorflow in the program
	 */
	private static final String TENSORFLOW_JAVA_BIOIMAGEIO_TAG = "tensorflow_saved_model_bundle";

	/**
	 * Variable containing the name used to refer to Pytorch in the program
	 */
	private static final String PYTORCH_JAVA_BIOIMAGEIO_TAG = "torchscript";

	/**
	 * Variable containing the name used to refer to the Pytorch that works with state dictionaries
	 */
	private static final String PYTORCH_STATE_DICT_BIOIMAGEIO_TAG = "pytorch_state_dict";

	/**
	 * Variable containing the name used to refer to Pytorch in the program
	 */
	private static final String ONNX_JAVA_BIOIMAGEIO_TAG = "onnx";

	/**
	 * Variable containing the name used to refer to Keras in the program
	 */
	private static final String KERAS_JAVA_BIOIMAGEIO_TAG = "keras_hdf5";
	/**
	 * List that contains all the supported weight tags by the bioengine
	 */
	private static final ArrayList<String> SUPPORTED_BIOENGINE_WEIGHTS;
	static {
		SUPPORTED_BIOENGINE_WEIGHTS = new ArrayList<String>();
		SUPPORTED_BIOENGINE_WEIGHTS.add(KERAS_JAVA_BIOIMAGEIO_TAG);
		SUPPORTED_BIOENGINE_WEIGHTS.add(ONNX_JAVA_BIOIMAGEIO_TAG);
		SUPPORTED_BIOENGINE_WEIGHTS.add(PYTORCH_JAVA_BIOIMAGEIO_TAG);
		SUPPORTED_BIOENGINE_WEIGHTS.add(TENSORFLOW_JAVA_BIOIMAGEIO_TAG);
		SUPPORTED_BIOENGINE_WEIGHTS.add(PYTORCH_STATE_DICT_BIOIMAGEIO_TAG);
	}

	/**
	 * Variable that stores which version of Tensorflow 1 has been already
	 * loaded to avoid errors for loading two different native libraries in the
	 * same namespace
	 */
	private static String loadedTf1Version;

	/**
	 * Variable that stores which version of Tensorflow 2 has been already
	 * loaded to avoid errors for loading two different native libraries in the
	 * same namespace
	 */
	private static String loadedTf2Version;

	/**
	 * Variable that stores which version of Pytorch has been already loaded to
	 * avoid errors for loading two different native libraries in the same
	 * namespace
	 */
	private static String loadedPytorchVersion;

	/**
	 * Variable that stores which version of Onnx has been already loaded to
	 * avoid errors for loading two different native libraries in the same
	 * namespace
	 */
	private static String loadedOnnxVersion;

	/**
	 * Information needed to know how to launch the corresponding Deep Learning
	 * framework
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 */
	private EngineInfo( String engine, String version, String jarsDirectory )
	{
		Objects.requireNonNull( engine, "The Deep Learning engine should not be null." );
		Objects.requireNonNull( version, "The Deep Learning engine version should not be null." );
		Objects.requireNonNull( jarsDirectory, "The Jars directory should not be null." );
		setEngine( engine );
		this.version = version;
		checkEngineAreadyLoaded();
		this.jarsDirectory = jarsDirectory;
		this.os = new PlatformDetection().toString();
		setSupportedVersions();
		this.versionJava = findCorrespondingJavaVersion();
		if (this.versionJava == null)
			throw new IllegalArgumentException("The DL framework version chosen is not supported.");
	}

	/**
	 * Information needed to know how to launch the Bioengine
	 * 
	 * @param engine
	 * 	weights to be run on the Bioengine
	 * @param serverURL
	 * 	url where the Bioengine is hosted
	 */
	private EngineInfo( String engine, String serverURL)
	{
		Objects.requireNonNull( engine, "The Deep Learning engine should not be null." );
		Objects.requireNonNull( serverURL, "The Deep Learning engine version should not be null." );
		this.engine = engine;
		this.serverURL = serverURL;
	}
	
	/**
	 * Check if the engine has already been loaded or not.
	 * If it is not possible to load the wanted version because another has already been
	 * loaded, an exception is thrown
	 * @throws IllegalArgumentException if an incompatible engine has already been loaded
	 */
	private void checkEngineAreadyLoaded() throws IllegalArgumentException {
		String versionedEngine = this.engine + this.getMajorVersion();
		boolean onnxLoaded = EngineLoader.getLoadedVersions().keySet().stream()
				.filter(en -> en.startsWith(ONNX_ENGINE_NAME) && !en.equals(versionedEngine))
				.findFirst().orElse(null) != null;
		boolean ptLoaded = EngineLoader.getLoadedVersions().keySet().stream()
				.filter(en -> en.startsWith(PYTORCH_ENGINE_NAME) && !en.equals(versionedEngine))
				.findFirst().orElse(null) != null;
		if (engine.equals(ONNX_ENGINE_NAME) && onnxLoaded) {
			String confV = EngineLoader.getLoadedVersions().entrySet().stream()
					.filter(en -> en.getKey().startsWith(ONNX_ENGINE_NAME) && !en.getKey().equals(versionedEngine))
					.map(en -> en.getValue()).findFirst().get();
			throw new IllegalArgumentException(
					String.format(ENGINE_ERR, engine, version, confV, version, confV));
		} else if (engine.equals(PYTORCH_ENGINE_NAME) && ptLoaded) {
			String confV = EngineLoader.getLoadedVersions().entrySet().stream()
					.filter(en -> en.getKey().startsWith(PYTORCH_ENGINE_NAME) && !en.getKey().equals(versionedEngine))
					.map(en -> en.getValue()).findFirst().get();
			throw new IllegalArgumentException(
					String.format(ENGINE_ERR, engine, version, confV, version, confV));
		} else if (!versionedEngine.equals("tensorflow2") && EngineLoader.getLoadedVersions().get(versionedEngine) != null
					&& !EngineLoader.getLoadedVersions().get(versionedEngine).equals(version)) {
			throw new IllegalArgumentException(String.format(ENGINE_ERR, engine, version, 
					EngineLoader.getLoadedVersions().get(versionedEngine), version, 
					EngineLoader.getLoadedVersions().get(versionedEngine)));
		}
	}

	/**
	 * Set the parameters to launch the wanted Bioengine instance.
	 * 
	 * @param engine
	 * 	DL framework we want to use in the Bioengine
	 * @param serverURL
	 * 	server where the instance of the Bioengine we want to connect to is hosted
	 * @return an object containing all the information needed to launch a Deep
	 * 	learning framework in the bioengine
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineBioengine( String engine, String serverURL ) throws IllegalArgumentException
	{	
		if (!SUPPORTED_BIOENGINE_WEIGHTS.contains(engine))
			throw new IllegalArgumentException("The only supported engine keys are: " + SUPPORTED_BIOENGINE_WEIGHTS);
		if (!DownloadModel.checkURL(serverURL))
			throw new IllegalArgumentException("The provided url does not exist: " + serverURL);
		return new EngineInfo(engine, serverURL);
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load DL models.
	 * 
	 * In this case neither CPU, nor GPU compatibility are not defined. The method will try to 
	 * find an engine compatible with CPU and GPU, but they are not required.
	 * 
	 * If the engine specified is not installed, the method will return null.
	 * The engine of interest needs to be installed first.
	 * A good way to check whether the engine of interest exists or not
	 * is: {@link InstalledEngines#checkEngineWithArgsInstalledForOS(String, String, Boolean, Boolean, String)}
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the wanted version engine is not installed
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineDLEngine( String engine, String version, String jarsDirectory ) throws IllegalArgumentException
	{	
		if (AvailableEngines.modelRunnerToBioimageioKeysMap().keySet().contains(engine))
			engine = AvailableEngines.modelRunnerToBioimageioKeysMap().get(engine);
		List<DeepLearningVersion> vs = 
				InstalledEngines.checkEngineWithArgsInstalledForOS(engine, version, 
						null, null, jarsDirectory);
		if (vs.size() == 0) {
			String jV = SupportedVersions.getJavaVersionForPythonVersion(engine, version);
			if (jV == null)
				return null;
			return defineDLEngineWithJavaVersion(engine, jV, null, null, jarsDirectory);
		}
		boolean cpu = false;
		boolean gpu = false;
		if (vs.stream().filter(v -> v.getCPU() && v.getGPU()).collect(Collectors.toList()).size() > 0) {
			cpu = true;
			gpu = true;
		} else if (vs.stream().filter(v -> v.getCPU()).collect(Collectors.toList()).size() > 0) {
			cpu = true;
		} else {
			gpu = true;
		}
		
		try {
			return defineDLEngine( engine, version, cpu, gpu, jarsDirectory );
		} catch (IllegalArgumentException ex) {
			ex.printStackTrace();
			return null;
		}
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load DL models.
	 * 
	 * In this case CPU compatibility is not defined. The method will try to 
	 * find an engine compatible with CPU but it is not required.
	 * 
	 * If the engine specified is not installed, the method will return null.
	 * The engine of interest needs to be installed first.
	 * A good way to check whether the engine of interest exists or not
	 * is: {@link InstalledEngines#checkEngineWithArgsInstalledForOS(String, String, Boolean, Boolean, String)}
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param gpu
	 *            whether the engine can use GPU or not
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the wanted version engine is not installed
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineDLEngine( String engine, String version, boolean gpu, String jarsDirectory )
												throws IllegalArgumentException
	{
		if (AvailableEngines.modelRunnerToBioimageioKeysMap().keySet().contains(engine))
			engine = AvailableEngines.modelRunnerToBioimageioKeysMap().get(engine);
		List<DeepLearningVersion> vs = 
				InstalledEngines.checkEngineWithArgsInstalledForOS(engine, version, null, 
						gpu, jarsDirectory);
		if (vs.size() == 0) {
			String jV = SupportedVersions.getJavaVersionForPythonVersion(engine, version);
			if (jV == null)
				return null;
			return defineDLEngineWithJavaVersion(engine, jV, null, gpu, jarsDirectory);
		}
		boolean cpu = false;
		if (vs.stream().filter(v -> v.getCPU()).collect(Collectors.toList()).size() > 0) 
			cpu = true;
		return defineDLEngine( engine, version, cpu, gpu, jarsDirectory );
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load DL models.
	 * 
	 * If the engine specified is not installed, the method will return null.
	 * The engine of interest needs to be installed first.
	 * A good way to check whether the engine of interest exists or not
	 * is: {@link InstalledEngines#checkEngineWithArgsInstalledForOS(String, String, Boolean, Boolean, String)}
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param cpu
	 *            whether the engine can use CPU or not
	 * @param gpu
	 *            whether the engine can use GPU or not
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the wanted version engine is not installed
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineDLEngine( String engine, String version, boolean cpu,
			boolean gpu, String jarsDirectory ) throws IllegalArgumentException
	{
		if (AvailableEngines.modelRunnerToBioimageioKeysMap().keySet().contains(engine))
			engine = AvailableEngines.modelRunnerToBioimageioKeysMap().get(engine);
		List<DeepLearningVersion> vvs =
				InstalledEngines.checkEngineWithArgsInstalledForOS(engine, version, cpu, 
						gpu, jarsDirectory);
		if (vvs.size() == 0) {
			String jV = SupportedVersions.getJavaVersionForPythonVersion(engine, version);
			if (jV == null)
				return null;
			return defineDLEngineWithJavaVersion(engine, jV, cpu, gpu, jarsDirectory);
		}
		EngineInfo engineInfo = new EngineInfo(engine, version, jarsDirectory);
		engineInfo.cpu = cpu;
		engineInfo.gpu = gpu;
		return engineInfo;
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load models.
	 * 
	 * In this case neither CPU, nor GPU compatibility are not defined. The method will try to 
	 * find an engine compatible with CPU and GPU, but they are not required.
	 * 
	 * If the engine specified is not installed, the method will return null.
	 * The engine of interest needs to be installed first.
	 * A good way to check whether the engine of interest exists or not
	 * is: {@link InstalledEngines#checkEngineWithArgsInstalledForOS(String, String, Boolean, Boolean, String)}
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the wanted version engine is not installed
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineDLEngine( String engine, String version ) throws IllegalArgumentException
	{
		if (AvailableEngines.modelRunnerToBioimageioKeysMap().keySet().contains(engine))
			engine = AvailableEngines.modelRunnerToBioimageioKeysMap().get(engine);
		Objects.requireNonNull( STATIC_JARS_DIRECTORY, "The Jars directory should not be null." );
		return defineDLEngine( engine, version, STATIC_JARS_DIRECTORY );
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load DL models.
	 * 
	 * In this case CPU compatibility is not defined. The method will try to 
	 * find an engine compatible with CPU but it is not required.
	 * 
	 * If the engine specified is not installed, the method will return null.
	 * The engine of interest needs to be installed first.
	 * A good way to check whether the engine of interest exists or not
	 * is: {@link InstalledEngines#checkEngineWithArgsInstalledForOS(String, String, Boolean, Boolean, String)}
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param gpu
	 *            whether the engine can use GPU or not
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the wanted version engine is not installed
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineDLEngine( String engine, String version, boolean gpu ) throws IllegalArgumentException
	{
		if (AvailableEngines.modelRunnerToBioimageioKeysMap().keySet().contains(engine))
			engine = AvailableEngines.modelRunnerToBioimageioKeysMap().get(engine);
		Objects.requireNonNull( STATIC_JARS_DIRECTORY, "The Jars directory should not be null." );
		List<DeepLearningVersion> vs = 
				InstalledEngines.checkEngineWithArgsInstalledForOS(engine, version, null, 
						gpu, STATIC_JARS_DIRECTORY);
		if (vs.size() == 0) {
			String jV = SupportedVersions.getJavaVersionForPythonVersion(engine, version);
			if (jV == null)
				return null;
			return defineDLEngineWithJavaVersion(engine, jV, null, gpu, STATIC_JARS_DIRECTORY);
		}
		boolean cpu = false;
		if (vs.stream().filter(v -> v.getCPU()).collect(Collectors.toList()).size() > 0) 
			cpu = true;
		return defineDLEngine( engine, version, cpu, gpu, STATIC_JARS_DIRECTORY );
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load models
	 * 
	 * If the engine specified is not installed, the method will return null.
	 * The engine of interest needs to be installed first.
	 * A good way to check whether the engine of interest exists or not
	 * is: {@link InstalledEngines#checkEngineWithArgsInstalledForOS(String, String, Boolean, Boolean, String)}
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param cpu
	 *            whether the engine can use CPU or not
	 * @param gpu
	 *            whether the engine can use GPU or not
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the wanted version engine is not installed
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineDLEngine( String engine, String version, 
			boolean cpu, boolean gpu ) throws IllegalArgumentException
	{
		if (AvailableEngines.modelRunnerToBioimageioKeysMap().keySet().contains(engine))
			engine = AvailableEngines.modelRunnerToBioimageioKeysMap().get(engine);
		Objects.requireNonNull( STATIC_JARS_DIRECTORY, "The Jars directory should not be null." );
		return defineDLEngine( engine, version, cpu, gpu, STATIC_JARS_DIRECTORY );
	}
	
	/**
	 * Method that finds an engine among the installed ones using the version of the
	 * engine in Java instead of the version of the engine in Python.
	 * 
	 * This method allows cpu = null and gpu = null. If any of those fields is null,
	 * the method will try to find the engines for which the field is tru, however, 
	 * if it does not find it it will continue with the field being false.
	 * 
	 * @param engine
	 * 	the Deep Learning framework of interest, cannot be null
	 * @param javaVersion
	 * 	version of the Java Deep Learning framework. It is equivalent to a set of 
	 * 	Python versions, cannot be null
	 * @param cpu
	 * 	whether the engine supports cpu or not, can be null
	 * @param gpu
	 * 	whether the engine supports gpu or not, can be null
	 * @param jarsDirectory
	 * 	Directory where the engines are located, cannot be null
	 * @return the engine info for the correspinding Java version, null if it is not
	 * 	installed for theat version.
	 */
	private static EngineInfo defineDLEngineWithJavaVersion(String engine, String javaVersion,
			Boolean cpu, Boolean gpu, String jarsDirectory) {
		Objects.requireNonNull(engine);
		Objects.requireNonNull(javaVersion);
		Objects.requireNonNull(jarsDirectory);
		List<DeepLearningVersion> vvs = InstalledEngines.checkEngineWithArgsInstalledForOS(
				engine, null, cpu, gpu, jarsDirectory);
		if (vvs.size() == 0)
			return null;
		List<DeepLearningVersion> compVersions = vvs.stream()
				.filter(vv -> vv.getVersion().equals(javaVersion)).collect(Collectors.toList());
		if (compVersions.size() == 0)
			return null;
		List<DeepLearningVersion> cpuComp = compVersions.stream()
				.filter(v -> v.getCPU() && (gpu == null || v.getGPU() == gpu))
				.collect(Collectors.toList());
		boolean ncpu;
		if (cpu == null && cpuComp.size() > 0)
			ncpu = true;
		else if (cpu == null)
			ncpu = false;
		else
			ncpu = cpu;
		List<DeepLearningVersion> gpuComp = compVersions.stream()
				.filter(v -> v.getCPU() == ncpu && v.getGPU()).collect(Collectors.toList());
		boolean ngpu;
		if (gpu == null && gpuComp.size() != 0)
			ngpu = true;
		else if (gpu == null)
			ngpu = false;
		else
			ngpu = gpu;
		String version = compVersions.stream()
				.filter(v -> v.getCPU() == ncpu && v.getGPU() == ngpu).findFirst().orElse(null).getPythonVersion();
		return defineDLEngine(engine, version, ncpu, ngpu, jarsDirectory);
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine)
	 * and load DL models.
	 * 
	 * In this method, the version defined is orientative to some extent. 
	 * If the version provided in the arguments is not installed, and there is another
	 * installed version of the same framework which has the same major version 
	 * (for example pytorch 1.13 and pytorch 1.9), the version installed will be loaded 
	 * directly instead of requiring the installation of the original version.
	 * 
	 * Also, for Pytorch if there is already another engine of the same framework, 
	 * same major version (same as before) but different overall version, 
	 * the previously loaded version will be used. This is because loading different versions
	 * of the Pytorch native libraries produce conflicts.
	 * 
	 *  Regard, that this method looks for engines compatible with CPU at least,
	 *  thus GPU support is not guaranteed.
	 *  
	 *  The method {@link #defineCompatibleDLEngineGPU(String, String, String)} does
	 *  the same but requiring GPU support.
	 *  
	 *  To know if the EngineInfo object has support for GPU, call {@link #isGPU()}
	 *  and if it returns false, and you want GPU support 
	 *  install the engine with GPU support if available.
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the engine of interest is not installed
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineCompatibleDLEngine( String engine, String version, 
			String jarsDirectory ) throws IOException, IllegalArgumentException 
	{
		InstalledEngines manager = InstalledEngines.buildEnginesFinder(jarsDirectory);
		String compatibleVersion = manager.getMostCompatibleVersionForEngine(engine, version);
		if (compatibleVersion == null)
			return null;
		List<DeepLearningVersion> vv = manager.getDownloadedForVersionedEngine(engine, compatibleVersion);
		boolean gpu = vv.stream().filter(v -> v.getGPU()).findFirst().orElse(null) != null;
		return EngineInfo.defineDLEngine(engine, compatibleVersion, true, gpu, jarsDirectory);
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine) compatible with GPU
	 * and load DL models.
	 * in the program.
	 * 
	 * In this method, the version defined is orientative to some extent. 
	 * If the version provided in the arguments is not installed, and there is another
	 * installed version of the same framework which has the same major version 
	 * (for example pytorch 1.13 and pytorch 1.9), the version installed will be loaded 
	 * directly instead of requiring the installation of the original version.
	 * 
	 * TODO if the engine already loaded supports GPU and we are askig for 
	 * TODO GPU support off, this should fail. Same thing for method that requires GPU
	 * TODO below
	 * 
	 * Also, for Pytorch if there is already another engine of the same framework, 
	 * same major version (same as before) but different overall version, 
	 * the previously loaded version will be used. This is because loading different versions
	 * of the Pytorch native libraries produce conflicts.
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param cpu
	 * 			  whether the engine has to support cpu or not
	 * @param gpu
	 * 			  whether the engine has to support gpu or not
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the engine of interest is not installed
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineCompatibleDLEngine( String engine, String version,
			boolean cpu, boolean gpu, String jarsDirectory ) throws IOException, IllegalArgumentException
	{
		List<DeepLearningVersion> possibles = 
				InstalledEngines.checkEngineWithArgsInstalledForOS(engine, null, cpu, gpu, 
						jarsDirectory);
		if (possibles.size() == 0)
			return null;
		List<String> possibleStrs = 
				possibles.stream().map(DeepLearningVersion::getPythonVersion).collect(Collectors.toList());
		List<String> compatibleVersion = 
				VersionStringUtils.getCompatibleEngineVersionsInOrder(version, possibleStrs, engine);
		if (compatibleVersion == null || compatibleVersion.size() == 0)
			return null;
		return EngineInfo.defineDLEngine(engine, compatibleVersion.get(0), cpu, true, jarsDirectory);
	}

	/**
	 * Set the parameters to launch the wanted Deep Learning framework (engine) compatible with GPU
	 * and load DL models.
	 * in the program.
	 * 
	 * In this method, the version defined is orientative to some extent. 
	 * If the version provided in the arguments is not installed, and there is another
	 * installed version of the same framework which has the same major version 
	 * (for example pytorch 1.13 and pytorch 1.9), the version installed will be loaded 
	 * directly instead of requiring the installation of the original version.
	 * 
	 * Also, for Pytorch if there is already another engine of the same framework, 
	 * same major version (same as before) but different overall version, 
	 * the previously loaded version will be used. This is because loading different versions
	 * of the Pytorch native libraries produce conflicts.
	 * 
	 *  Regard, that this method looks for engines compatible with GPU specifically,
	 *  thus CPU support is not guaranteed.
	 *  
	 *  To know if the EngineInfo object has support for CPU, call {@link #isCPU()}
	 *  and if it returns false, and you want CPU support 
	 *  install the engine with CPU support if available.
	 * 
	 * @param engine
	 *            name of the Deep Learning framework (engine). For example:
	 *            Pytorch, Tensorflow....
	 * @param version
	 *            version of the training Deep Learning framework (engine)
	 * @param jarsDirectory
	 *            directory where the folder containing the JARs needed to
	 *            launch the corresponding engine are located
	 * @return an object containing all the information needed to launch a Deep
	 *         learning framework or null if the engine of interest is not installed
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineCompatibleDLEngineGPU( String engine, String version, 
			String jarsDirectory ) throws IOException, IllegalArgumentException
	{
		List<DeepLearningVersion> possibles = 
				InstalledEngines.checkEngineWithArgsInstalledForOS(engine, null, null, true,
						jarsDirectory);
		if (possibles.size() == 0)
			return null;
		List<String> possibleStrs = 
				possibles.stream().map(DeepLearningVersion::getPythonVersion).collect(Collectors.toList());
		List<String> compatibleVersion = 
				VersionStringUtils.getCompatibleEngineVersionsInOrder(version, possibleStrs, engine);
		if (compatibleVersion == null || compatibleVersion.size() == 0)
			return null;
		boolean cpu = InstalledEngines.checkEngineWithArgsInstalledForOS(engine, compatibleVersion.get(0), 
				true, true, jarsDirectory).size() > 0;
		return EngineInfo.defineDLEngine(engine, compatibleVersion.get(0), cpu, true, jarsDirectory);
	}
	
	/**
	 * Create an {@link EngineInfo} object from an specific weigth definition of the rdf.yaml file
	 * This method assumes that the directory where the engine folders are downloaded to is 
	 * a directory called "engines" inside the application folder of the main program.
	 * 
	 * The version of the weights does not need to match exactly the version of the
	 * engine installed to enable loading Pytorch 1.11.0 models with Pytorch 1.13.1
	 * 
	 * @param weight
	 * 	the weights of a model for a specific single engine (DL framework)
	 * @return the {@link EngineInfo} object if there are compatible installed engines or null
	 * 	if they do not exist
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineCompatibleDLEngineWithRdfYamlWeights(WeightFormat weight) 
			throws IOException, IllegalArgumentException {
		return defineCompatibleDLEngineWithRdfYamlWeights(weight, InstalledEngines.getEnginesDir());
	}
	
	/**
	 * Create an {@link EngineInfo} object from an specific weigth definition of the rdf.yaml file
	 * 
	 * The version of the weights does not need to match exactly the version of the
	 * engine installed to enable loading Pytorch 1.11.0 models with Pytorch 1.13.1
	 * 
	 * @param weight
	 * 	the weights of a model for a specific single engine (DL framework)
	 * @param enginesDir
	 * 	directory where all the engine folders are downloaded
	 * @return the {@link EngineInfo} object if there are compatible installed engines or null
	 * 	if they do not exist
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineCompatibleDLEngineWithRdfYamlWeights(WeightFormat weight, 
			String enginesDir) throws IOException, IllegalArgumentException {
		String compatibleVersion = null;
		String engine = weight.getWeightsFormat();
		String version = weight.getTrainingVersion();
		InstalledEngines manager = InstalledEngines.buildEnginesFinder(enginesDir);
		compatibleVersion = manager.getMostCompatibleVersionForEngine(engine, version);
		if (compatibleVersion == null)
			return null;
		List<DeepLearningVersion> vv = manager.getDownloadedForVersionedEngine(engine, compatibleVersion);
		boolean gpu = vv.stream().filter(v -> v.getGPU()).findFirst().orElse(null) != null;
		return EngineInfo.defineDLEngine(engine, compatibleVersion, true, gpu, enginesDir);
	}
	
	/**
	 * Create an {@link EngineInfo} object from an specific weigth definition of the rdf.yaml file
	 * This method assumes that the directory where the engine folders are downloaded to is 
	 * a directory called "engines" inside the application folder of the main program.
	 * 
	 * The version of the weights needs to match exactly the version of the
	 * engine installed. The major and minor versions need to match.
	 * Only Pytorch 1.11 can be used to load Pytorch 1.11
	 * 
	 * @param weight
	 * 	the weights of a model for a specific single engine (DL framework)
	 * @return the {@link EngineInfo} object if there are compatible installed engines or null
	 * 	if they do not exist
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineExactDLEngineWithRdfYamlWeights(WeightFormat weight)
			throws IOException, IllegalArgumentException {
		return defineExactDLEngineWithRdfYamlWeights(weight, InstalledEngines.getEnginesDir());
	}
	
	/**
	 * Create an {@link EngineInfo} object from an specific weigth definition of the rdf.yaml file
	 * 
	 * The version of the weights needs to match exactly the version of the
	 * engine installed. The major and minor versions need to match.
	 * Only Pytorch 1.11 can be used to load Pytorch 1.11
	 * 
	 * @param weight
	 * 	the weights of a model for a specific single engine (DL framework)
	 * @param enginesDir
	 * 	directory where all the engine folders are downloaded
	 * @return the {@link EngineInfo} object if there are compatible installed engines or null
	 * 	if they do not exist
	 * @throws IOException if the engines directory does not exist
	 * @throws IllegalArgumentException if an engine that cannot be loaded together with the wanted engine
	 * 	has already been loaded
	 */
	public static EngineInfo defineExactDLEngineWithRdfYamlWeights(WeightFormat weight,
			String enginesDir) throws IOException, IllegalArgumentException {
		String engine = weight.getWeightsFormat();
		String version = weight.getTrainingVersion();
		InstalledEngines manager = InstalledEngines.buildEnginesFinder(enginesDir);
		if (version == null)
			return null;
		List<DeepLearningVersion> vv = manager.getDownloadedForVersionedEngine(engine, version);
		if (vv.size() == 0)
			return null;
		boolean gpu = vv.stream().filter(v -> v.getGPU()).findFirst().orElse(null) != null;
		return EngineInfo.defineDLEngine(engine, version, true, gpu, enginesDir);
	}

	/**
	 * Retrieve the complete name of the Deep Learning framework (engine)
	 * version. It includes the engine, the Java version, the os and the the
	 * machine. It should be the name of the directory where the needed JARs are
	 * stored.
	 * 
	 * @return a String with all the characteristics of the Deep Learning engine
	 */
	public String getDeepLearningVersionJarsDirectory()
	{
		final String vv = this.engine + "-" + this.version + "-" + this.versionJava + "-" + this.os
					+ ( this.cpu ? "-cpu" : "" ) + ( this.gpu ? "-gpu" : "" );
		return this.jarsDirectory + File.separator + vv;
	}

	/**
	 * Set whether the engine supports GPU or not. Does not support GPU by
	 * default.
	 * 
	 * @param support
	 *            true if it supports GPU and false otherwise
	 */
	public void supportGPU( boolean support )
	{
		gpu = support;
	}

	/**
	 * Set whether the engine supports CPU or not. By default supports CPU
	 * 
	 * @param support
	 *            true if it supports CPU and false otherwise
	 */
	public void supportCPU( boolean support )
	{
		cpu = support;
	}

	/**
	 * Finds the version of Deep Learning framework (engine) equivalent or
	 * compatible with the one used to train the model. This is done because
	 * sometimes APIs for different languages are named differently
	 * 
	 * @return corresponding compatible version of the DL framework Java version
	 */
	public String findCorrespondingJavaVersion()
	{
		return this.supportedVersions.getCorrespondingJavaVersion( this.version );
	}

	/**
	 * Create the object that contains all the supported versions for the Deep
	 * Learning framework (engine) selected
	 */
	private void setSupportedVersions()
	{
		this.supportedVersions = new SupportedVersions( this.engine );
	}

	/**
	 * REturn the name of the engine (Deep Learning framework)
	 * 
	 * @return the name of the Deep Learning framework
	 */
	public String getEngine()
	{
		return engine;
	}

	/**
	 * Set the Deep Learning framework (engine) of the model
	 * 
	 * @param engine
	 *            Deep Learning framework used for the model
	 */
	public void setEngine( String engine )
	{
		if ( engine.contentEquals( TENSORFLOW_JAVA_BIOIMAGEIO_TAG ) )
			this.engine = TENSORFLOW_ENGINE_NAME;
		else if ( engine.contentEquals( PYTORCH_JAVA_BIOIMAGEIO_TAG ) )
			this.engine = PYTORCH_ENGINE_NAME;
		else if ( engine.contentEquals( ONNX_JAVA_BIOIMAGEIO_TAG ) )
			this.engine = ONNX_ENGINE_NAME;
	}

	/**
	 * Set the directory where the program will look for the Deep Learning
	 * framework jars See {@link #jarsDirectory} for more explanation
	 * 
	 * @param jarsDirectory
	 *            directory where all the folders containing the JARs are stored
	 */
	public void setJarsDirectory( String jarsDirectory )
	{
		this.jarsDirectory = jarsDirectory;
	}

	/**
	 * Return the String path to the directory where all the jars to load a Deep
	 * Learning framework (engine) are stored. See {@link #jarsDirectory} for
	 * more explanation
	 * 
	 * @return String path to the directory where all the jars are stored
	 */
	public String getJarsDirectory()
	{
		return this.jarsDirectory;
	}

	/**
	 * Set the tags needed to load a Tensorflow model. These fields are useless
	 * for other models
	 * 
	 * @param tag
	 *            tad used to open a Tf model
	 * @param sigDef
	 *            signature definition used to open a tf model
	 */
	public void setTags( String tag, String sigDef )
	{
		if ( this.engine.contentEquals( TENSORFLOW_ENGINE_NAME ) )
		{
			this.tfTag = tag;
			this.tfSigDef = sigDef;
		}
	}

	/**
	 * Get Tensorflow Signature Definition to open model
	 * 
	 * @return Tensorflow Signature Definition to open model
	 */
	public String getTfSigDef()
	{
		return this.tfSigDef;
	}

	/**
	 * Get Tensorflow tag to open model
	 * 
	 * @return Tensorflow tag to open model
	 */
	public String getTfTag()
	{
		return this.tfTag;
	}

	/**
	 * Return version of the Deep Learning framework (engine). The version
	 * corresponds to the one used to train the network.
	 * 
	 * @return version of the engine where the model was trained
	 */
	public String getVersion()
	{
		return this.version;
	}

	/**
	 * Return version of the Deep Learning framework (engine). The version
	 * corresponds to the one used to run the model in Java.
	 * 
	 * @return version of the engine where the model was trained
	 */
	public String getJavaVersion()
	{
		return this.versionJava;
	}

	/**
	 * True if the engine allows running on GPU or false otherwise. By default
	 * false
	 * 
	 * @return True if the engine allows running on GPU or false otherwise
	 */
	public boolean isGPU()
	{
		return this.gpu;
	}

	/**
	 * True if the engine allows running on CPU or false otherwise. True by
	 * default
	 * 
	 * @return True if the engine allows running on CPU or false otherwise
	 */
	public boolean isCPU()
	{
		return this.cpu;
	}

	/**
	 * Get the operating system of the machine
	 * 
	 * @return the operation system. If can be either: "windows", "linux",
	 *         "solaris" or "mac"
	 */
	public String getOS()
	{
		return this.os;
	}

	/**
	 * Sets which versions have already been loaed to avoid errors trying to
	 * load another version from the same engine, which always crashes the
	 * application
	 */
	public void setLoadedVersion()
	{
		if ( this.engine.equals( TENSORFLOW_ENGINE_NAME ) && this.version.startsWith( "1" ) )
		{
			loadedTf1Version = this.version;
		}
		else if ( this.engine.equals( TENSORFLOW_ENGINE_NAME ) && this.version.startsWith( "2" ) )
		{
			loadedTf2Version = this.version;
		}
		else if ( this.engine.equals( PYTORCH_ENGINE_NAME ) )
		{
			loadedPytorchVersion = this.version;
		}
		else if ( this.engine.equals( ONNX_ENGINE_NAME ) )
		{
			loadedOnnxVersion = this.version;
		}
	}

	/**
	 * REturns which versions have been already been loaded to avoid errors of
	 * overlapping versions
	 * 
	 * @param engine
	 *            the Deep Learning framework of interest
	 * @param version
	 *            the Deep LEarning version of interest
	 * @return the loaded version of the selected engine or null if no version
	 *         has been loaded
	 * @throws IllegalArgumentException
	 *             if the engine is not supported yet
	 */
	public static String getLoadedVersions( String engine, String version ) throws IllegalArgumentException
	{
		if ( engine.equals( TENSORFLOW_JAVA_BIOIMAGEIO_TAG ) && version.startsWith( "1" ) )
		{
			return loadedTf1Version;
		}
		else if ( engine.equals( TENSORFLOW_JAVA_BIOIMAGEIO_TAG ) && version.startsWith( "2" ) )
		{
			return loadedTf2Version;
		}
		else if ( engine.equals( PYTORCH_JAVA_BIOIMAGEIO_TAG ) )
		{
			return loadedPytorchVersion;
		}
		else if ( engine.equals( ONNX_JAVA_BIOIMAGEIO_TAG ) )
		{
			return loadedOnnxVersion;
		}
		else
		{
			throw new IllegalArgumentException( "The selected engine '" + engine + "' is not supported yet." );
		}
	}

	/**
	 * Get the major version of the Deep Learning framework. This is the first
	 * number of the version until the first dot.
	 * 
	 * @return the major version of the engine
	 */
	public String getMajorVersion()
	{
		int ind = version.indexOf( "." );
		String majorVersion = "" + version;
		if ( ind != -1 )
			majorVersion = version.substring( 0, ind );
		return majorVersion;
	}
	
	/**
	 * Method that checks if the Deep Learning engine specified by the {@link EngineInfo}
	 * object is 
	 * @return true if the engine is installed and false otherwise
	 */
	public boolean isEngineInstalled() {
		File file = new File(this.getDeepLearningVersionJarsDirectory());
		try {
			boolean missingJars = (DeepLearningVersion.fromFile(file).checkMissingJars().size() == 0);
			if (!missingJars)
				return false;
		} catch (Exception e) {
			return false;
		}
		return true;
	}
	
	/**
	 * Find the installed engine that has a version closes to the one defined for this instance.
	 * If this instance has version 1.4 and the only installed engine for the same DL framework
	 * is 1.8, the result will be an engine info object with the same characteristics and different
	 * engine version.
	 * If the original engine was cpu gpu and there are cpu installed but not gpu, a cpu will be returned.
	 * 
	 * If there is an installed engine for the original EngineInfo instance, the same EngineInfo will
	 * be returned
	 * 
	 * @return the most compatible engine with the one defined, if it exists
	 * @throws IOException if no engine of the same DL framework is found
	 */
	public EngineInfo getEngineInfoOfTheClosestInstalledEngineVersion() throws IOException {
		String newV = InstalledEngines.getMostCompatibleVersionForEngine(jarsDirectory, engine, version);
		String msg = "There are no installed engines of the DL framework: "  + engine + version.split("\\.")[0];
		if (newV == null)
			throw new IOException(msg);
		EngineInfo newInfo = EngineInfo.defineDLEngine(engine, newV, this.isCPU(), this.isGPU(), jarsDirectory);
		if (!newInfo.isEngineInstalled())
			newInfo.gpu = !this.gpu;
		if (!newInfo.isEngineInstalled())
			throw new IOException(msg);
		return newInfo;
	}
	
	/**
	 * Return whether the {@link EngineInfo} object is designed to load a model 
	 * remotely on the Bioengine or not
	 * @return true if loading on the Bioengine is the objective or false otherwise
	 */
	public boolean isBioengine() {
		return this.serverURL != null;
	}
	
	/**
	 * 
	 * @return the server url where the bioengine is hosted.
	 * @throws IllegalArgumentException if the method is called when the {@link EngineInfo}
	 * 	object has not been created for the Bioengine, with the method {@link #defineBioengine(String, String)}
	 */
	public String getServer() {
		if (!isBioengine())
			throw new IllegalArgumentException("The Bioengine server can only be retrieved if the "
					+ "method EngineInfo.defineBioengine(String, String) has been used to create the object.");
		return this.serverURL;
	}

	/**
	 * Set in a static manner the {@link #STATIC_JARS_DIRECTORY} if it is not
	 * going to change during the execution of the program
	 * 
	 * @param jarsDirectory
	 *            the permanent jars directory
	 */
	public static void setStaticJarsDirectory( String jarsDirectory )
	{
		STATIC_JARS_DIRECTORY = jarsDirectory;
	}

	/**
	 * Method that returns the name with which Tensorflow is defined needed at
	 * some points to differentiate between tf1 and tf2
	 * 
	 * @return the String used for tensorflow
	 */
	public static String getTensorflowKey()
	{
		return TENSORFLOW_ENGINE_NAME;
	}

	/**
	 * Method that returns the name with which Pytorch is defined
	 * 
	 * @return the String used for Pytorch
	 */
	public static String getPytorchKey()
	{
		return PYTORCH_ENGINE_NAME;
	}

	/**
	 * Method that returns the name with which Onnx is defined
	 * 
	 * @return the String used for Onnx
	 */
	public static String getOnnxKey()
	{
		return ONNX_ENGINE_NAME;
	}

	/**
	 * Method that returns the name with which Keras is defined 
	 * 
	 * @return the String used for Keras
	 */
	public static String getKerasKey()
	{
		return KERAS_ENGINE_NAME;
	}

	/**
	 * Method that returns the name with which Tensorflow is defined in the
	 * Bioimage.io.
	 * 
	 * @return the String used for tensorflow in the Bioimage.io
	 */
	public static String getBioimageioTfKey()
	{
		return TENSORFLOW_JAVA_BIOIMAGEIO_TAG;
	}

	/**
	 * Method that returns the name with which Pytorch is defined in the
	 * Bioimage.io.
	 * 
	 * @return the String used for Pytorch in the Bioimage.io
	 */
	public static String getBioimageioPytorchKey()
	{
		return PYTORCH_JAVA_BIOIMAGEIO_TAG;
	}

	/**
	 * Method that returns the name with which Onnx is defined in the
	 * Bioimage.io.
	 * 
	 * @return the String used for Onnx in the Bioimage.io
	 */
	public static String getBioimageioOnnxKey()
	{
		return ONNX_JAVA_BIOIMAGEIO_TAG;
	}

	/**
	 * Method that returns the name with which Keras is defined in the
	 * Bioimage.io.
	 * 
	 * @return the String used for Onnx in the Bioimage.io
	 */
	public static String getBioimageioKerasKey()
	{
		return KERAS_JAVA_BIOIMAGEIO_TAG;
	}
}
