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
package io.bioimage.modelrunner.engine.installation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.download.DownloadTracker;
import io.bioimage.modelrunner.bioimageio.download.DownloadTracker.TwoParameterConsumer;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.utils.Log;
import io.bioimage.modelrunner.versionmanagement.AvailableEngines;
import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;

/**
 * Class that manages the dl-modelrunner engines.
 * This class checks that the required engines are installed and installs them if they are not.
 * There is one required engine per DL framework. It can be either the latest one or the one specified
 * in the variable {@link EngineManagement#ENGINES_VERSIONS}.
 * This class also contains the methods to install engines on demand.
 * @author Carlos Garcia Lopez de Haro, Ivan Estevez Albuja and Caterina Fuster Barcelo
 *
 */
public class EngineManagement {
	/**
	 * Directory where the engines shold be installed
	 */
	private static final String ENGINES_DIR = new File("engines").getAbsolutePath();
	/**
	 * Map containing which version should always be installed per framework
	 */
	public static LinkedHashMap<String, String> ENGINES_VERSIONS = new LinkedHashMap<String, String>();
	
	static {
		ENGINES_VERSIONS.put(EngineInfo.getTensorflowKey() + "_2", "2.7.0");
		ENGINES_VERSIONS.put(EngineInfo.getTensorflowKey() + "_1", "1.15.0");
		ENGINES_VERSIONS.put(EngineInfo.getOnnxKey() + "_17", "17");
		ENGINES_VERSIONS.put(EngineInfo.getPytorchKey() + "_1", "1.13.1");
	}
	
	/**
	 * Map containing the reference from bioimage.io key to the engine key used
	 * to name the engine folder
	 */
	private static final Map<String, String> ENGINES_MAP = 
			AvailableEngines.bioimageioToModelRunnerKeysMap().entrySet()
			.stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
	
	/**
	 * Key word that substitutes the OS part of the engine folder name in some
	 * of the engines installed from the update site
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  - "tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another. 
	 */
	private static final String GENERAL_KEYWORD = "-general";
	/**
	 * Keyword used to identify the engine being installed
	 */
	public static final String PROGRESS_ENGINE_KEYWORD = "Engine: ";
	/**
	 * Whether the minimum required engines are installed or not
	 */
	private boolean everythingInstalled = false;
	/**
	 * Which of the required engines are not installed
	 */
	private LinkedHashMap<String, String> missingEngineFolders;
	/**
	 * String that communicates the progress made downloading engines
	 */
	private String progressString;
	/**
	 * Flag to communicate if the management of engines is already finished
	 */
	private boolean isManagementFinished = false;
	/**
	 * For the automatic installation of basic engines, consumer that contains 
	 * a consumer for each of the engines that is going to be installed.
	 * Key is the engine installed, the value is the consumer for that engine
	 */
	private Map<String, TwoParameterConsumer<String, Double>> consumersMap;
	
	/**
	 * Constructor that checks whether the minimum engines are installed
	 * or not.
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  - "tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another. 
	 */
	private EngineManagement() {
	}
	
	/**
	 * Creates an {@link EngineManagement} object to check if the required engines are installed.
	 * 
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  - "tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another. 
	 * @return a manager to handle the installation of basic engines
	 */
	public static EngineManagement createManager() {
		return new EngineManagement();
	}
	
	/**
	 * Checks if the minimal required engines to execute the majority of models
	 * are installed, if not it manages the installation of the missing ones.
	 *  
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  - "tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another.  
	 * 
	 */
	public void checkAndSetBasicEngineInstallation() {
		isManagementFinished = false;
		readEnginesJSON();
		checkEnginesInstalled();
		if (!this.everythingInstalled)
			manageMissingEngines();
		isManagementFinished = true;
	}
	
	/**
	 * Checks if the minimal required engines to execute the majority of models
	 * are installed.
	 *  
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  - "tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another. 
	 */
	public void checkBasicEngineInstallation() {
		isManagementFinished = false;
		readEnginesJSON();
		checkEnginesInstalled();
	}
	
	/**
	 * Manages the installation of the minimal required engines to execute the majority of models.
	 *  
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  - "tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  - "tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another. 
	 * 
	 */
	public void basicEngineInstallation() {
		if (!this.everythingInstalled)
			manageMissingEngines();
		isManagementFinished = true;
	}
	
	/**
	 * Returns a map containing which of the main engines, those that
	 * together can run the majority of the models (one for pytorch 1, one for
	 * tf1, one for tf2 and one for onnx) are not installed
	 * @return map containing which of the main engines have not been installed
	 */
	public Map<String, String> getNotInstalledMainEngines(){
		if (missingEngineFolders == null)
			checkEnginesInstalled();
		return missingEngineFolders;
	}
	
	/**
	 * Retrieve a map that contains a {@link TwoParameterConsumer} for each of the 
	 * engines installed. 
	 * The engines will be download in order of the keys in the map.
	 * @return a map where each key corresponds to each of the basic engines
	 *  ({@link #ENGINES_VERSIONS}) missing and its value will be the consumer 
	 *  used to track the progress of its download
	 */
	public Map<String, TwoParameterConsumer<String, Double>> getBasicEnginesProgress() {
		if (consumersMap != null && consumersMap.size() != 0)
			return consumersMap;
		if (missingEngineFolders == null)
			checkEnginesInstalled();
		consumersMap = new LinkedHashMap<String, TwoParameterConsumer<String, Double>>();
		for (String missing : missingEngineFolders.values()) {
			this.consumersMap.put(missing, DownloadTracker.createConsumerProgress());
		}
		return consumersMap;
	}
	
	/**
	 * Returns the required engines that have not been yet correctly installed
	 * @return a list with the required engines for the plugin that have not been installed
	 * 	yet.
	 */
	public ArrayList<String> getMissingEngines() {
		if (missingEngineFolders == null)
			checkEnginesInstalled();
		return new ArrayList<>(missingEngineFolders.keySet());
	}
	
	/**
	 * Read the engines JSON and finds if there is any new framework that is not required
	 * by the {@link #ENGINES_VERSIONS} dictionary. 
	 * If a new framework is found, the {@link #ENGINES_VERSIONS} dictionary is updated with
	 * the latest version available in the JSON for that framework.
	 * Regard that there is a workaround for Onnx versions due to its weird versioning,
	 * instead of being 1.1.0, 1.2.1, 1.3.0, it changes as 1, 2, 3, 4...
	 * 
	 * The engines JSOn is located at {@code /src/main/resources/availableDLVersions.json}
	 * at the JAR file {@code dl-modelrunner-X-Y-Z.jar}.
	 * The link to a github repo is: https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 */
	private void readEnginesJSON() {
		Map<String, String> versionsNotInRequired = getListOfSingleVersionsPerFrameworkNotInRequired();
		List<String> uniqueFrameworks = versionsNotInRequired.keySet().stream()
				 .map(f -> f.substring(0, f.lastIndexOf("_"))).distinct()
				 .collect(Collectors.toList());
		Comparator<String> versionComparator = (v1, v2) -> {
			 // Multiply by -1 because we want to return 1 if v1 is bigger and -1 otherwise
			 // and the used method does the opposite
			 return DeepLearningVersion.stringVersionComparator(v1, v2) * -1;
	        };
        for (String f : uniqueFrameworks) {
			 String selectedVersion = versionsNotInRequired.entrySet().stream()
					 .filter( v -> v.getKey().startsWith(f + "_"))
					 .map(v -> v.getValue()).max(versionComparator).orElse(null);
			 ENGINES_VERSIONS.put(f + "_" + selectedVersion.indexOf("."), selectedVersion);
		 }
		 
		
	}
	
	/**
	 * Method to retrieve a list of single python versions for the system OS. If there exist GPU
	 * and CPU versions, it chooses the GPU one
	 * @return a list of all the versions available for each framework that is not contained in 
	 * the {@link #ENGINES_MAP} map of required versions
	 */
	public static Map<String, String> getListOfSingleVersionsPerFrameworkNotInRequired() {
		List<DeepLearningVersion> vList = AvailableEngines
				.loadCompatibleOnly().getVersions().stream()
				.filter( v -> !v.getEngine().startsWith(EngineInfo.getOnnxKey())
						&& !ENGINES_VERSIONS.keySet().contains( v.getEngine() 
						+ "_" + v.getPythonVersion().substring(0, v.getPythonVersion().indexOf(".")) ) 
						&& v.getOs().equals(new PlatformDetection().toString()))
				.collect(Collectors.groupingBy(DeepLearningVersion::getPythonVersion)).values().stream()
			    .flatMap(sizeGroup -> {
			        List<DeepLearningVersion> uniquePythonVersions = sizeGroup.stream()
			            .filter(v -> sizeGroup.stream()
			            		.noneMatch(otherV -> v != otherV && v.getPythonVersion().equals(otherV.getPythonVersion())))
			            .collect(Collectors.toList());

			        List<DeepLearningVersion> guVersions = sizeGroup.stream()
			            .filter(obj -> obj.getGPU()).limit(1).collect(Collectors.toList());

			        uniquePythonVersions.addAll(guVersions);
			        return uniquePythonVersions.stream();
			    })
			    .collect(Collectors.toList());

		Map<String, String> versionsNotInRequired = vList.stream().collect(Collectors.toMap(
							v -> v.getEngine() + "_" + v.getPythonVersion(), v -> v.getPythonVersion()));;
		return versionsNotInRequired;
	}
	
	/**
	 * Checks which of the required engines are not installed.
	 */
	public void checkEnginesInstalled() {
		Map<String, String> engineFolders = ENGINES_VERSIONS.entrySet().stream()
				.collect(Collectors.toMap( v -> v.getKey(), v -> {
					String framework = v.getKey().substring(0, v.getKey().lastIndexOf("_"));
					if (ENGINES_MAP.get(framework) != null)
						framework = ENGINES_MAP.get(framework);
					String pythonVersion = v.getValue();
					try {
						boolean gpu = true;
						if (!isEngineSupported(framework, pythonVersion, true, gpu))
							gpu = false;
						EngineInfo engineInfo = 
							EngineInfo.defineDLEngine(framework, pythonVersion, ENGINES_DIR, true, gpu);
						return engineInfo.getDeepLearningVersionJarsDirectory();
					} catch (Exception ex) {
						return "";
					}
					},(u, v) -> u, LinkedHashMap::new));

		missingEngineFolders = engineFolders.entrySet().stream()
				.filter( dir -> {
					try {
						File dirFile = new File(dir.getValue());
						return !dirFile.isDirectory() || DeepLearningVersion.fromFile(dirFile).checkMissingJars().size() != 0;
					} catch (Exception e) {
						return true;
					}
				} ).collect(Collectors.toMap(dir -> dir.getKey(), dir -> dir.getValue(),
						(u, v) -> u, LinkedHashMap::new));
		
		if (missingEngineFolders.entrySet().size() == 0)
			everythingInstalled = true;
	}
	
	/**
	 * Manages the missing engines, either renaming the OS-general engine folder
	 * to OS-specific or directly installing the engine from scratch.
	 * 
	 * This method tries to find if there is any engine with the {@link #GENERAL_KEYWORD}
	 * tag and renames it following the dl-modelrunner naming convention.
	 * 
	 * In order to reduce repetition and to reduce the number of deps downloaded
	 * by the user when deepImageJ is installed, a selection of engines is created.
	 * There are some of the engines that are the same for every operating system,
	 * for example TF1 and Pytorch.
	 * To reduce the number of deps donwloaded, the TF1 and Pytorch engines are
	 * installed as for example:
	 *  - "tensorflow-1.15.0-1.15.0" + {@link #GENERAL_KEYWORD}
	 * Just one of the above folder is downloaded for all the OS.
	 * If it was not done like this, as the Fiji update site does not recognise the
	 * OS of the user, the three following engines would be required:
	 *  -"tensorflow-1.15.0-1.15.0-windows-x86_64-cpu-gpu"
	 *  -"tensorflow-1.15.0-1.15.0-linux-x86_64-cpu-gpu"
	 *  -"tensorflow-1.15.0-1.15.0-macos-x86_64-cpu"
	 * however, the 3 engines would contain exactly the same deps.
	 * This is the reason why we make the workaround to download a single
	 * engine for certain engines and then rename it follwoing the corresponding
	 * convention
	 * 
	 * Regard, that for certain engines, downloading all the OS depending engines
	 * is necessary, as the dependencies vary from one system to another. 
	 * 
	 */
	private void manageMissingEngines() {
		if (missingEngineFolders == null)
			checkEnginesInstalled();
		if (missingEngineFolders.entrySet().size() == 0)
			return;
		missingEngineFolders = missingEngineFolders.entrySet().stream()
			.filter(v -> {
				String value = v.getValue();
				String generalName = value.substring(0, value.indexOf(new PlatformDetection().toString()) - 1);
				generalName += GENERAL_KEYWORD;
				File generalFile = new File(generalName);
				if (generalFile.isDirectory() && generalFile.renameTo(new File(v.getValue())))
					return false;
				return true;
			}).collect(Collectors.toMap(v -> v.getKey(), v -> v.getValue(),
					(u, v) -> u, LinkedHashMap::new));
		installMissingBasicEngines();
	}
	
	/**
	 * Install the missing engines from scratch
	 */
	private void installMissingBasicEngines() {
		if (missingEngineFolders == null)
			checkEnginesInstalled();
		if (missingEngineFolders.entrySet().size() == 0)
			return;
		missingEngineFolders = missingEngineFolders.entrySet().stream()
				.filter(v -> {
					TwoParameterConsumer<String, Double> consumer = DownloadTracker.createConsumerProgress();
					if (this.consumersMap != null && this.consumersMap.get(v.getValue()) != null)
						consumer = this.consumersMap.get(v.getValue());
					return!installEngineByCompleteName(v.getValue(), consumer);
				})
				.collect(Collectors.toMap(v -> v.getKey(), v -> v.getValue(),
						(u, v) -> u, LinkedHashMap::new));
	}
	
	/**
	 * Install the engine that should be located in the engine dir specified
	 * @param engineDir
	 * 	directory where the specific engine shuold be installed. Regard that this 
	 * 	is the whole path to the folder, and that the folder name should follow the 
	 * 	dl-modelrunner naming convention (https://github.com/bioimage-io/model-runner-java#readme)
	 * @return true if the installation was successful and false otherwise
	 */
	public static boolean installEngineByCompleteName(String engineDir) {
		return installEngineByCompleteName(engineDir, null);
	}
	
	/**
	 * Install the engine that should be located in the engine dir specified
	 * @param engineDir
	 * 	directory where the specific engine shuold be installed. Regard that this 
	 * 	is the whole path to the folder, and that the folder name should follow the 
	 * 	dl-modelrunner naming convention (https://github.com/bioimage-io/model-runner-java#readme)
	 * @param consumer
	 * 	consumer used to communicate the progress made donwloading files. It can be null
	 * @return true if the installation was successful and false otherwise
	 */
	public static boolean installEngineByCompleteName(String engineDir,
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) {
		File engineFileDir = new File(engineDir);
		if (!engineFileDir.isDirectory() && engineFileDir.mkdirs() == false)
			return false;
		DeepLearningVersion dlVersion;
		try {
			dlVersion = DeepLearningVersion.fromFile(engineFileDir);
			return installEngine(dlVersion, consumer);
		} catch (Exception e) {
			return false;
		}
	}
	
	/**
	 * Install the DL engine corresponding to the weights provided.
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param ww
	 * 	{@link WeightFormat} object containing the weights as in the yaml file
	 * @return true if the DL engine was installed or false otherwise
	 * @throws IOException if there is any error creating the folder for the engine
	 * @throws InterruptedException if the thread of the download is interrumpted abruptly
	 */
	public static boolean installEngineForWeights(WeightFormat ww) throws IOException, InterruptedException {
		return installEngineForWeightsInDir(ww, InstalledEngines.getEnginesDir(), null);
	}
	
	/**
	 * Install the DL engine corresponding to the weights provided.
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param ww
	 * 	{@link WeightFormat} object containing the weights as in the yaml file
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if the DL engine was installed or false otherwise
	 * @throws IOException if there is any error creating the folder for the engine
	 * @throws InterruptedException if the thread of the download is interrumpted abruptly
	 */
	public static boolean installEngineForWeights(WeightFormat ww, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		return installEngineForWeightsInDir(ww, InstalledEngines.getEnginesDir(), consumer);
	}

	/**
	 * Install the DL engine corresponding to the weights provided.
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param ww
	 * 	{@link WeightFormat} object containing the weights as in the yaml file
	 * @param enginesDir
	 * 	directory where the engine is installed
	 * @return true if the DL engine was installed or false otherwise
	 * @throws IOException if there is any error creating the folder for the engine
	 * @throws InterruptedException if the thread of the download is interrumpted abruptly
	 */
	public static boolean installEngineForWeightsInDir(WeightFormat ww, String enginesDir) throws IOException, InterruptedException {
		return installEngineForWeightsInDir(ww, enginesDir, null);
	}


	/**
	 * Install the DL engine corresponding to the weights provided.
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param ww
	 * 	{@link WeightFormat} object containing the weights as in the yaml file
	 * @param enginesDir
	 * 	directory where the engine is installed
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if the DL engine was installed or false otherwise
	 * @throws IOException if there is any error creating the folder for the engine
	 * @throws InterruptedException if the thread of the download is interrumpted abruptly
	 */
	public static boolean installEngineForWeightsInDir(WeightFormat ww, String enginesDir, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		InstalledEngines manager = InstalledEngines.buildEnginesFinder(enginesDir);
		String engine = ww.getWeightsFormat();
		String version = ww.getTrainingVersion();
		List<DeepLearningVersion> vs = manager.getDownloadedCompatibleForVersionedEngine(engine, version);
		if (vs.size() != 0)
			return true;
		if (AvailableEngines.isEngineSupported(engine, version, true, true)) {
			DeepLearningVersion dlv = 
					AvailableEngines.getEngineForOsByParams(engine, version, true, true);
			return installEngineInDir(dlv, enginesDir, consumer);
		} else if (AvailableEngines.isEngineSupported(engine, version, true, false)) {
			DeepLearningVersion dlv = 
					AvailableEngines.getEngineForOsByParams(engine, version, true, false);
			return installEngineInDir(dlv, enginesDir, consumer);
		} else if (AvailableEngines.isEngineSupported(engine, version, false, true)) {
			DeepLearningVersion dlv = 
					AvailableEngines.getEngineForOsByParams(engine, version, false, true);
			return installEngineInDir(dlv, enginesDir, consumer);
		}
		return false;
	}

	/**
	 * Install all the engines specified by all the weights specified in the
	 * model rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param descriptor
	 * 	{@link ModelDescriptor} object containing the information from the rdf.yaml
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModel(ModelDescriptor descriptor) throws IOException {
		return installEnginesForModelInDir(descriptor, InstalledEngines.getEnginesDir());
	}
	
	/**
	 * Install all the engines specified by all the weights specified in the
	 * model rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param descriptor
	 * 	{@link ModelDescriptor} object containing the information from the rdf.yaml
	 * 	successfully installed
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModel(ModelDescriptor descriptor, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException {
		return installEnginesForModelInDir(descriptor, InstalledEngines.getEnginesDir(), consumer);
	}
	
	/**
	 * Install all the engines specified by all the weights specified in the
	 * model rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param descriptor
	 * 	{@link ModelDescriptor} object containing the information from the rdf.yaml
	 * 	successfully installed
	 * @param enginesDir
	 * 	directory where the engine is installed
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelInDir(ModelDescriptor descriptor, String enginesDir) throws IOException {
		return installEnginesForModelInDir(descriptor, enginesDir, null);
	}
	
	/**
	 * Install all the engines specified by all the weights specified in the
	 * model rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param descriptor
	 * 	{@link ModelDescriptor} object containing the information from the rdf.yaml
	 * 	successfully installed
	 * @param enginesDir
	 * 	directory where the engine is installed
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelInDir(ModelDescriptor descriptor, String enginesDir,
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException {
		boolean installed = true;
		for (WeightFormat ww : descriptor.getWeights().getSupportedWeights()) {
			try {
				boolean status = installEngineForWeightsInDir(ww, enginesDir, consumer);
				if (!status)
					System.out.println("DL engine not supported by JDLL: " + ww.getWeightsFormat());
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
				installed = false;
			}
		}
		return installed;
	}
	
	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the modelID provided. The modelId is the field 'id' in
	 * the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param modelID
	 * 	ID of the model as defined in the rdf.yaml file
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByID(String modelID) throws IOException {
		return installEnginesForModelByIDInDir(modelID, InstalledEngines.getEnginesDir(), null);
	}
	
	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the modelID provided. The modelId is the field 'id' in
	 * the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param modelID
	 * 	ID of the model as defined in the rdf.yaml file
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByID(String modelID, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException {
		return installEnginesForModelByIDInDir(modelID, InstalledEngines.getEnginesDir(), consumer);
	}
	
	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the modelID provided. The modelId is the field 'id' in
	 * the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param modelID
	 * 	ID of the model as defined in the rdf.yaml file
	 * @param enginesDir
	 *  directory where the engines are installed
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByIDInDir(String modelID, String enginesDir) throws IOException {
		return installEnginesForModelByIDInDir(modelID, enginesDir, null);
	}
	
	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the modelID provided. The modelId is the field 'id' in
	 * the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param modelID
	 * 	ID of the model as defined in the rdf.yaml file
	 * @param enginesDir
	 *  directory where the engines are installed
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByIDInDir(String modelID, String enginesDir, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException {
		ModelDescriptor descriptor = BioimageioRepo.connect().selectByID(modelID);
		if (descriptor == null)
			return false;
		return installEnginesForModelInDir(descriptor, enginesDir, consumer);
	}
	
	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the model name provided. The model name is the field 
	 * 'name' in the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param modelName
	 * 	model name as defined in the rdf.yaml file
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByName(String modelName) throws IOException {
		return installEnginesForModelByNameinDir(modelName, InstalledEngines.getEnginesDir(), null);
	}
	
	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the model name provided. The model name is the field 
	 * 'name' in the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param modelName
	 * 	model name as defined in the rdf.yaml file
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByName(String modelName, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException {
		return installEnginesForModelByNameinDir(modelName, InstalledEngines.getEnginesDir(), consumer);
	}

	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the model name provided. The model name is the field 
	 * 'name' in the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param modelName
	 * 	model name as defined in the rdf.yaml file
	 * @param enginesDir
	 * 	dir where the engines will be installed
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByNameinDir(String modelName, String enginesDir) throws IOException {
		return installEnginesForModelByNameinDir(modelName, enginesDir, null);
	}

	/**
	 * Install the DL engines corresponding to the weights defined in the yaml.rdf
	 * file corresponding to the model name provided. The model name is the field 
	 * 'name' in the rdf.yaml
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param modelName
	 * 	model name as defined in the rdf.yaml file
	 * @param enginesDir
	 * 	dir where the engines will be installed
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelByNameinDir(String modelName, String enginesDir, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException {
		ModelDescriptor descriptor = BioimageioRepo.connect().selectByName(modelName);
		if (descriptor == null)
			return false;
		return installEnginesForModelInDir(descriptor, enginesDir, consumer);
	}
	
	/**
	 * Install the DL engine corresponding to the weights of the Bioimage.io
	 * model contained in the provided folder. The engines are read from the 
	 * model rdf.yaml file
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param modelFolder
	 * 	folder containing a Bioimage.io model with a valid rdf.yaml
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelInFolder(String modelFolder) throws Exception {
		return installEnginesinDirForModelInFolder(modelFolder, InstalledEngines.getEnginesDir(), null);
	}
	
	/**
	 * Install the DL engine corresponding to the weights of the Bioimage.io
	 * model contained in the provided folder. The engines are read from the 
	 * model rdf.yaml file
	 * 
	 * The DL engine is downloaded automatically into a directory called "models"
	 * inside the application folder.
	 * 
	 * @param modelFolder
	 * 	folder containing a Bioimage.io model with a valid rdf.yaml
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesForModelInFolder(String modelFolder, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws Exception {
		return installEnginesinDirForModelInFolder(modelFolder, InstalledEngines.getEnginesDir(), consumer);
	}
	
	/**
	 * Install the DL engine corresponding to the weights of the Bioimage.io
	 * model contained in the provided folder. The engines are read from the 
	 * model rdf.yaml file
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param modelFolder
	 * 	folder containing a Bioimage.io model with a valid rdf.yaml
	 * @param enginesDir
	 * 	directory where the engines will be installed
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesinDirForModelInFolder(String modelFolder, String enginesDir) throws Exception {
		return installEnginesinDirForModelInFolder(modelFolder, enginesDir, null);
	}
	
	/**
	 * Install the DL engine corresponding to the weights of the Bioimage.io
	 * model contained in the provided folder. The engines are read from the 
	 * model rdf.yaml file
	 * 
	 * The DL engine is downloaded automatically into a directory provided.
	 * 
	 * @param modelFolder
	 * 	folder containing a Bioimage.io model with a valid rdf.yaml
	 * @param enginesDir
	 * 	directory where the engines will be installed
	 * @param consumer
	 * 	consumer used to keep track of the process of download of the weights
	 * @return true if at least one DL engine of the model weights defined in the rdf.yaml is
	 * 	successfully installed
	 * @throws IOException if there is any error creating the folder for the engine
	 */
	public static boolean installEnginesinDirForModelInFolder(String modelFolder, String enginesDir, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws Exception {
		if (new File(modelFolder, "rdf.yaml").isFile() == false)
			throw new IOException("A Bioimage.io model folder should contain its corresponding rdf.yaml file.");
		ModelDescriptor descriptor = 
				ModelDescriptor.readFromLocalFile(modelFolder + File.separator + "rdf.yaml");
		return installEnginesForModelInDir(descriptor, enginesDir, consumer);
	}
	
	/**
	 * Install the engine specified by the {@link DeepLearningVersion} object
	 * @param engine
	 * 	the {@link DeepLearningVersion} object specifying the wanted engine
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if there is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumpted abruptly while downloading
	 */
	public static boolean installEngine(DeepLearningVersion engine) throws IOException, InterruptedException {
		return installEngine(engine, null);
	}
	
	/**
	 * Install the engine specified by the {@link DeepLearningVersion} object
	 * @param engine
	 * 	the {@link DeepLearningVersion} object specifying the wanted engine
	 * @param consumer
	 * 	consumer used to communicate the progress made donwloading files
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if there is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumpted abruptly while downloading
	 */
	public static boolean installEngine(DeepLearningVersion engine, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		return installEngineInDir(engine, ENGINES_DIR, consumer);
	}
	
	/**
	 * Install the engine specified by the {@link DeepLearningVersion} object
	 * @param engine
	 * 	the {@link DeepLearningVersion} object specifying the wanted engine
	 * @param engineDir
	 * 	directory where the engines are downloaded. Inside the dir specififed the 
	 * 	corresponding folder for the engine will be created, which will contain the 
	 * 	files needed to run the engine
	 * @param consumer
	 * 	consumer used to communicate the progress made donwloading files
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if there is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumped abruptly while downloading
	 */
	public static boolean installEngineInDir(DeepLearningVersion engine, String engineDir, 
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		Log.addProgressAndShowInTerminal(null, PROGRESS_ENGINE_KEYWORD + engine.folderName(), true);
		if (consumer == null)
			consumer = DownloadTracker.createConsumerProgress();
		String folder = engineDir + File.separator + engine.folderName();
		if (!new File(folder).isDirectory() && !new File(folder).mkdir())
			throw new IOException("Unable to create the folder where the engine "
					+ "will be installed: " + folder);
		
		Thread downloadThread = new Thread(() -> {
			downloadEngineFiles(engine, folder);
        });
		downloadThread.start();
		
		DownloadTracker tracker = DownloadTracker.getFilesDownloadTracker(folder,
				consumer, engine.getJars(), downloadThread);
		Thread trackerThread = new Thread(() -> {
            try {
            	tracker.track();
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}
        });
		trackerThread.start();
		DownloadTracker.printProgress(downloadThread, consumer);
		List<String> badDownloads = tracker.findMissingDownloads();
		if (badDownloads.size() > 0)
			throw new IOException("The following files of engine '" + engine.folderName()
			+ "' where downloaded incorrectly: " + badDownloads.toString());
		return true;
	}
	
	/**
	 * Method that just downloads all the files that form a given engine into the
	 * directory provided
	 * @param engine
	 * 	engine to be downloaded
	 * @param engineDir
	 * 	directory where the files will be downloaded
	 */
	private static void downloadEngineFiles(DeepLearningVersion engine, String engineDir) {
		for (String jar : engine.getJars()) {
			try {
				URL website = new URL(jar);
				Path filePath = Paths.get(website.getPath()).getFileName();
				try (ReadableByteChannel rbc = Channels.newChannel(website.openStream());
						FileOutputStream fos = new FileOutputStream(new File(engineDir, filePath.toString()))){
						FileDownloader downloader = new FileDownloader(rbc, fos);
						downloader.call();
				} catch (IOException e) {
					String msg = "The link for the file: " + filePath.getFileName() + " is broken." + System.lineSeparator() 
								+ "JDLL will continue with the download but the model might be "
								+ "downloaded incorrectly. The link is '" + jar + "'.";
					new IOException(msg, e).printStackTrace();
				}
			} catch (IOException e) {
				new IOException("The following URL is wrong: " + jar, e).printStackTrace();
			}
		}
	}
	
	/**
	 * Install the engine specified by the arguments of the method
	 * The engine will be installed in the default engines directory, which is the
	 * folder named 'engines' inside the application directory
	 * @param framework
	 * 	DL framework as specified by the Bioimage.io model zoo ()https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md)
	 * @param version
	 * 	the version of the framework
	 * @param cpu
	 * 	whether the engine supports cpu or not
	 * @param gpu
	 * 	whether the engine supports gpu or not
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if tehre is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumped abruptly while downloading
	 */
	public static  boolean installEngineWithArgs(String framework, String version, boolean cpu, boolean gpu) throws IOException, InterruptedException {
		return installEngineWithArgsInDir(framework, version, cpu, gpu, ENGINES_DIR, null);
	}
	
	/**
	 * Install the engine specified by the arguments of the method
	 * The engine will be installed in the provided directory
	 * folder named 'engines' inside the application directory
	 * @param framework
	 * 	DL framework as specified by the Bioimage.io model zoo ()https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md)
	 * @param version
	 * 	the version of the framework
	 * @param cpu
	 * 	whether the engine supports cpu or not
	 * @param gpu
	 * 	whether the engine supports gpu or not
	 * @param dir
	 * 	directory where the engine will be installed
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if tehre is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumped abruptly while downloading
	 */
	public static  boolean installEngineWithArgsInDir(String framework, String version, 
			boolean cpu, boolean gpu, String dir) throws IOException, InterruptedException {
		return installEngineWithArgsInDir(framework, version, cpu, gpu, dir, null);
	}
	
	/**
	 * Install the engine specified by the arguments of the method.
	 * The engine will be installed in the default engines directory, which is the
	 * folder named 'engines' inside the application directory
	 * @param framework
	 * 	DL framework as specified by the Bioimage.io model zoo ()https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md)
	 * @param version
	 * 	the version of the framework
	 * @param cpu
	 * 	whether the engine supports cpu or not
	 * @param gpu
	 * 	whether the engine supports gpu or not
	 * @param consumer
	 * 	consumer used to communicate the progress made donwloading files
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if tehre is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumped abruptly while downloading
	 */
	public static  boolean installEngineWithArgs(String framework, String version, 
			boolean cpu, boolean gpu, DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		return installEngineWithArgsInDir(framework, version, cpu, gpu, ENGINES_DIR, consumer);
	}
	
	/**
	 * Install the engine specified by the arguments of the method in the wanted folder
	 * @param framework
	 * 	DL framework as specified by the Bioimage.io model zoo ()https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md)
	 * @param version
	 * 	the version of the framework
	 * @param cpu
	 * 	whether the engine supports cpu or not
	 * @param gpu
	 * 	whether the engine supports gpu or not
	 * @param dir
	 * 	folder where the engine will be installed
	 * @param consumer
	 * 	consumer used to communicate the progress made donwloading files
	 * @return true if the installation was successful and false otherwise
	 * @throws IOException if tehre is any error downloading the engine
	 * @throws InterruptedException if the main thread is interrumped abruptly while downloading
	 */
	public static  boolean installEngineWithArgsInDir(String framework, String version, 
			boolean cpu, boolean gpu, String dir,
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws IOException, InterruptedException {
		if (AvailableEngines.bioimageioToModelRunnerKeysMap().get(framework) != null)
			framework = AvailableEngines.bioimageioToModelRunnerKeysMap().get(framework);
		DeepLearningVersion engine = AvailableEngines.getAvailableVersionsForEngine(framework).getVersions()
				.stream().filter(v -> (v.getPythonVersion() == version)
					&& (v.getCPU() == cpu)
					&& (v.getGPU() == gpu)).findFirst().orElse(null);
		return installEngineInDir(engine, dir, consumer);
	}
    
    /**
     * Retrieve the progress String
     * @return progress String that updates the progress about installing engines
     */
    public String getProgressString() {
    	return progressString;
    }
    
    /**
     * Check whether the management of the engines is finished or not
     * @return true if it is finished or false otherwise
     */
    public boolean isManagementDone() {
    	return isManagementFinished;
    }
    
    /**
     * Check if an engine is supported by the dl-modelrunner or not
     * @param framework
	 * 	DL framework as specified by the Bioimage.io model zoo ()https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md)
	 * @param version
	 * 	the version of the framework
	 * @param cpu
	 * 	whether the engine supports cpu or not
	 * @param gpu
	 * 	whether the engine supports gpu or not
     * @return true if the engine exists and false otherwise
     */
    public static boolean isEngineSupported(String framework, String version, boolean cpu, boolean gpu) {
    	if (ENGINES_MAP.get(framework) != null)
			framework = AvailableEngines.bioimageioToModelRunnerKeysMap().get(framework);
    	DeepLearningVersion engine = AvailableEngines.getAvailableVersionsForEngine(framework).getVersions()
				.stream().filter(v -> v.getPythonVersion().equals(version) 
						&& v.getOs().equals(new PlatformDetection().toString())
						&& v.getCPU() == cpu
						&& v.getGPU() == gpu
						&& (!(new PlatformDetection().isUsingRosseta()) || v.getRosetta()))
				.findFirst().orElse(null);
		if (engine == null) 
			return false;
		return true;
    }
	
    /**
     * Get file size of the file located in the given url
     * @param url
     * 		url of interest
     * @return the size of the file at the url or -1 if there is any error
     */
	public static long getFileSize(URL url) {
		long totSize = -1;
		HttpURLConnection conn = null;
		try {
			conn = (HttpURLConnection) url.openConnection();
			conn.setRequestMethod("HEAD");
			totSize = conn.getContentLengthLong();
		} catch (IOException e) {
		}
		return totSize;
	}
	
    /**
     * Get file size of the file located in the given url
     * @param strUrl
     * 		url of interest as a String 
     * @return the size of the file at the url or -1 if there is any error
     */
	public long getFileSize(String strUrl) {
		long totSize = -1;
		try {
			URL url = new URL(strUrl);
			totSize = getFileSize(url);
		} catch (IOException e) {
		}
		return totSize;
	}
}
