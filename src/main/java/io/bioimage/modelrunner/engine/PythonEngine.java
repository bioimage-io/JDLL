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

import io.bioimage.modelrunner.versionmanagement.SupportedVersions;

/**
 * Class to create an object that contains all the information about a Deep
 * Learning framework (engine) that is needed to launch the engine in an
 * independent ClassLoader
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class PythonEngine
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
	private PythonEngine( String engine, String version, String jarsDirectory )
	{
	}
}
