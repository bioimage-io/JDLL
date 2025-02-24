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
package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.download.MultiFileDownloader;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.ZipUtils;

/**
 * 
 * Class to manage the downloading of models from the BioImage.io
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class DownloadModel {
	/**
	 * Map that contain all the links needed to download a BioImage.io model
	 */
	private Map<String, String> downloadableLinks;
	/**
	 * Model descriptor containing all the information about the model
	 */
	private ModelDescriptor descriptor;
	/**
	 * Folder where the model is going to be downloaded
	 */
	private String modelsDir;
	/**
	 * Map containing the size of each of the files defined by the links
	 */
	private LinkedHashMap<String, Long> map;
    /**
     * Consumer used to send info about the download to other threads
     */
    private Consumer<Double> consumer;
    /**
     * String that logs the file that it is being downloaded at the current moment
     */
    private String progressString = "";
    /**
     * Whether a file has to be unzipped or not
     */
    private boolean unzip = false;
    /**
     * Progress unzipping the tensorflow model zip
     */
    private double unzippingProgress = 0.0;
    /**
     * Consumer used to tackthe progress of unzipping the model weights
     * if they are stored in a zip
     */
    private Consumer<Double> unzippingConsumer;
    /**
     * String that announces that certain file is just begining to be downloaded
     */
    public static final String START_DWNLD_STR = "START: ";
    /**
     * String that announces that certain file is just begining to be downloaded
     */
    public static final String END_DWNLD_STR = " -- END" + System.lineSeparator();
    /**
     * String that announces that certain file is just begining to be downloaded
     */
    public static final String FILE_SIZE_STR = " ** FILE_SIZE **";
    /**
     * String to communicate that there has been a download error
     * with a given file
     */
    public static final String DOWNLOAD_ERROR_STR = " --**ERROR**-- ";
    /**
     * String that communcates that the whole String was finished downloading
     */
    public static final String FINISH_STR = " --**END MODEL DOWNLOAD**-- ";
	/**
	 * Key for the map that contains the attachments of the model 
	 */
	private static String ATTACH_KEY = "attachments";
	/**
	 * Key for the map that contains the weights of the model 
	 */
	private static String WEIGHTS_KEY = "weights";
	/**
	 * Key for the map that contains the rdf URL of the model 
	 */
	private static String RDF_KEY = "rdf_source";
	/**
	 * Key for the map that contains the sample inputs of the model 
	 */
	private static String SAMPLE_INPUTS_KEY = "sample_input";
	/**
	 * Key for the map that contains the sample outputs of the model 
	 */
	private static String SAMPLE_OUTPUTS_KEY = "sample_output";
	/**
	 * Key for the map that contains the test inputsof the model 
	 */
	private static String TEST_INPUTS_KEY = "test_input";
	/**
	 * Key for the map that contains the test outputs of the model 
	 */
	private static String TEST_OUTPUTS_KEY = "test_output";
	/**
	 * Key for the map that contains the covers for the model
	 */
	private static String COVERS_KEY = "covers";
	
	/**
	 * Constructor that contains all the info and is able to download a BioImage.io model
	 * @param descriptor
	 * 	information about the model from the rdf.yaml
	 */
	private DownloadModel(ModelDescriptor descriptor, String modelsDir) {
		this.descriptor = descriptor;
		String fname = addTimeStampToFileName(descriptor.getName(), true);
		this.modelsDir = modelsDir + File.separator + getValidFileName(fname);
    	this.unzippingConsumer = new Consumer<Double>() {
    		@Override
            public void accept(Double d) {
        		unzippingProgress = d;
            }
        };
		retriveDownloadModelLinks();
	}
	
	public void setProgressConsumer(Consumer<Double> consumer) {
		this.consumer = consumer;
	}
	
	/**
	 * If the Sting contains any forbidden character, this method substitutes them 
	 * by "_"
	 * @param fileName
	 * 	filename to be validated
	 * @return a valid filename, if the input does not contain any forbidden character 
	 * 	it will be the same, if it does contain them, they will be replaced by "_"
	 */
	public static String getValidFileName(String fileName) {
        Pattern pattern = Pattern.compile("[\\\\/:*?\"<>|]");
        String name = new File(fileName).getName();
        String validFileName = pattern.matcher(name).replaceAll("_");
        validFileName = fileName.substring(0, fileName.lastIndexOf(name)) + validFileName;
        return validFileName;
	}
	
	/**
	 * REtrieve the model folder that contains the bmz model files
	 * @return folder containing the BMZ model files
	 */
	public String getModelFolder() {
		return this.modelsDir;
	}

	/**
	 * Build a constructor that contains all the info and is able to download a BioImage.io model
	 * @param descriptor
	 * 	information about the model from the rdf.yaml
	 * @param modelsDir
	 * 	directory where the model will be downloaded
	 * @return object that can be handles the download of all the files needed for a model
	 */
	public static DownloadModel build(ModelDescriptor descriptor, String modelsDir) {
		return new DownloadModel(descriptor, modelsDir);
	}

	/**
	 * Build a constructor that contains all the info and is able to download a BioImage.io model
	 * @param descriptor
	 * 	information about the model from the rdf.yaml
	 * @return object that can be handles the download of all the files needed for a model
	 */
	public static DownloadModel build(ModelDescriptor descriptor) {
		return new DownloadModel(descriptor, new File("models").getAbsolutePath());
	}
	
	/**
	 * Retrieve the list of strings (URLs) that need to be donloaded to create the 
	 * BioImage.io model specified in the rdf.yaml
	 */
	private void retriveDownloadModelLinks() {
		downloadableLinks = new HashMap<String, String>();
		addAttachments();
		addRDF();
		addCovers();
		addSampleInputs();
		addSampleOutputs();
		addTestInputs();
		addTestOutputs();
		addWeights();
		//checkModelWasDownloaded();
	}
	
	/**
	 * Method that checks whether the model was downloaded or not. In order
	 * for the model to be considered as downlaoded, it has to include the rdf.yaml
	 * and the weight supported by DeepIcy that are specified in the rdf.yaml
	 * @throws IOException if the web file cannot be accessed or if the downloaded file's size
	 * 	is different to the file in the website
	 */
	public void checkModelWasDownloaded() throws IOException {
		long websize = -1;
		for (String link : getListOfLinks()) {
			String fileName;
			try {
				fileName = FileDownloader.getFileNameFromURLString(link);
				websize = FileDownloader.getFileSize(new URL(link));
			} catch (MalformedURLException e) {
				throw new MalformedURLException("URL: '" + link + "' specified in the "
						+ "model rdf.yaml does not exist");
			}
			String name = modelsDir + File.separator + fileName;
			long localSize = new File(name).length();
			if (localSize != websize)
				throw new IOException("Downloaded file: '" + name + "' is not the same size as "
						+ "the file at: '" + link + "'.");
		}
	}
	
	/**
	 * Add weight links to the downloadable links
	 */
	private void addWeights() {
		ModelWeight weights = descriptor.getWeights();
		int c = 0;
		for (WeightFormat w : weights.gettAllSupportedWeightObjects()) {
			try {
				if (w.getSource() != null) {
					downloadableLinks.put(WEIGHTS_KEY + "_" + c ++, descriptor.getModelURL() + w.getSource());
					if (w.getSourceFileName().endsWith(".zip"))
						unzip = true;
				}
				if (w.getArchitecture() != null && w.getArchitecture().getSource() != null)
					downloadableLinks.put(WEIGHTS_KEY + "_" + c ++, descriptor.getModelURL() + w.getArchitecture().getSource());
				if (w.getEnvDependencies() != null && w.getEnvDependencies().getSource() != null)
					downloadableLinks.put(WEIGHTS_KEY + "_" + c ++, descriptor.getModelURL() + w.getEnvDependencies().getSource());
			} catch (Exception ex) {
				// The exception is thrown whenever the weight format is not present.
				// This exception will not be thrown here because the weight formats are retrieved from the same object
			}
		}
	}
	
	/**
	 * Add the model covers to the downloadable links
	 */
	private void addCovers() {
		int c = 0;
		for (String ss : descriptor.getCovers()) {
			if (ss != null && !checkURL(ss))
				downloadableLinks.put(COVERS_KEY + "_" + c ++, this.descriptor.getModelURL() + ss);
			else if (ss != null)
				downloadableLinks.put(COVERS_KEY + "_" + c ++, ss);
		}
	}
	
	/**
	 * Add the test inputs to the downloadable links
	 */
	private void addTestInputs() {
		List<String> fileNames = descriptor.getInputTensors().stream()
				.map(tt -> tt.getTestTensorName()).collect(Collectors.toList());
		int c = 0;
		for (String ss : fileNames) {
			if (ss == null)
				continue;
			downloadableLinks.put(TEST_INPUTS_KEY + "_" + c ++, this.descriptor.getModelURL() + ss);
		}
	}
	
	/**
	 * Add the test outputs to the dowloadable links
	 */
	private void addTestOutputs() {
		List<String> fileNames = descriptor.getOutputTensors().stream()
				.map(tt -> tt.getTestTensorName()).collect(Collectors.toList());
		int c = 0;
		for (String ss : fileNames) {
			if (ss == null)
				continue;
			downloadableLinks.put(TEST_OUTPUTS_KEY + "_" + c ++, this.descriptor.getModelURL() + ss);
		}
	}
	
	/**
	 * Add the sample inputs to the dowloadable links
	 */
	private void addSampleInputs() {
		List<String> fileNames = descriptor.getInputTensors().stream()
				.map(tt -> tt.getSampleTensorName()).collect(Collectors.toList());
		int c = 0;
		for (String ss : fileNames) {
			if (ss == null)
				continue;
			downloadableLinks.put(SAMPLE_INPUTS_KEY + "_" + c ++, this.descriptor.getModelURL() + ss);
		}
	}
	
	/**
	 * Add the sample outputs to the dowloadable links
	 */
	private void addSampleOutputs() {
		List<String> fileNames = descriptor.getOutputTensors().stream()
				.map(tt -> tt.getSampleTensorName()).collect(Collectors.toList());
		int c = 0;
		for (String ss : fileNames) {
			if (ss == null)
				continue;
			downloadableLinks.put(SAMPLE_OUTPUTS_KEY + "_" + c ++, this.descriptor.getModelURL() + ss);
		}
	}
	
	/**
	 * Add the rdf.yaml file to the downloadable links of the model
	 */
	private void addRDF() {
		downloadableLinks.put(RDF_KEY, descriptor.getModelURL() + Constants.RDF_FNAME);
	}
	
	/**
	 * Add the attachment files to the downloadable links of the model
	 */
	private void addAttachments() {
		List<String> attachments = descriptor.getAttachments();
		if (attachments == null)
			return;
		int c = 0;
		for (String kk : attachments)
			downloadableLinks.put(ATTACH_KEY + "_" + c ++, descriptor.getModelURL() + kk);
	}
	
	/**
	 * Check if a String contains a valid URL
	 * @param str
	 * 	str that might contain an URL
	 * @return true if the String corresponds to an URL and false otherwise
	 */
	public static boolean checkURL(String str) {
		try {
			URL url = new URL(str);
		    return true;
		} catch (MalformedURLException e) {
			return false;
		}
	}
	
	/**
	 * Get list of links that need to be downloaded to create a model
	 * @return the list of links to be downloaded
	 */
	public List<String> getListOfLinks() {
		List<String> links = new ArrayList<String>();
		for (String kk : downloadableLinks.keySet())
			links.add(downloadableLinks.get(kk));
		return links;
	}
	
	/**
	 * Add the timestamp to the String given
	 * @param str
	 * 	String to add the time stamp
	 * @param isDir
	 * 	whether the file name represents a directory or not
	 * @return string with the timestamp
	 */
	public static String addTimeStampToFileName(String str, boolean isDir) {
		// Add timestamp to the model name. 
		// The format consists on: modelName + date as ddmmyyyy + time as hhmmss
        Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		if (isDir)
			return str + "_" + dateString;
		int ind = str.lastIndexOf(File.separator);
		String fileName = str;
		if (ind != -1)
			fileName = str.substring(ind + 1);
		int extensionPos = fileName.lastIndexOf(".");
		if (extensionPos == -1)
			return str + "_" + dateString;
		String nameNoExtension = str.substring(0, extensionPos);
		String extension = str.substring(extensionPos);
		return nameNoExtension + "_" + dateString + extension;
	}
	
	/**
	 * Add the timestamp to the String given
	 * @param str
	 * 	String to add the time stamp
	 * @return string with the timestamp
	 */
	public static String addTimeStampToFileName(String str) {
		// Add timestamp to the model name. 
		// The format consists on: modelName + date as ddmmyyyy + time as hhmmss
        Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		int ind = str.lastIndexOf(File.separator);
		String fileName = str;
		if (ind != -1)
			fileName = str.substring(ind + 1);
		int extensionPos = fileName.lastIndexOf(".");
		if (extensionPos == -1)
			return str + "_" + dateString;
		String nameNoExtension = str.substring(0, extensionPos);
		String extension = str.substring(extensionPos);
		return nameNoExtension + "_" + dateString + extension;
	}
	
	/**
	 * Download a model downloading one by one all the files that should be inside
	 * the model folder into a created folder inside the models repo
	 * @throws IOException if there is any error creating the folder or downloading the files
	 * @throws InterruptedException if the thread was stopped by the user
	 * @throws ExecutionException if anything in the download goes wrong
	 */
	public void downloadModel() throws IOException, InterruptedException, ExecutionException {
		downloadModel(Thread.currentThread());
	}
	
	/**
	 * Download a model downloading one by one all the files that should be inside
	 * the model folder into a created folder inside the models repo
	 * @param parentThread
	 * 	reference thread for the download. If it is stopped the download will stop too.
	 * @throws IOException if there is any error creating the folder or downloading the files
	 * @throws InterruptedException if the thread was stopped by the user
	 * @throws ExecutionException if anything in the download goes wrong
	 */
	public void downloadModel(Thread parentThread) throws IOException, InterruptedException, ExecutionException {
		File folder = new File(modelsDir);
		if (!folder.isDirectory() && !folder.mkdirs())
			throw new IOException("The provided directory where the model is going to "
					+ "be downloaded does not exist and cannot be created ->" + modelsDir);
		List<URL> urls = new ArrayList<URL>();
		for (String link : getListOfLinks()) 
			urls.add(new URL(link));
		MultiFileDownloader mfd = new MultiFileDownloader(urls, folder, parentThread);
		mfd.setPartialProgressConsumer(consumer);
		mfd.download();
		
		if (unzip)
			unzipTfWeights();
	}
	
	/**
	 * Method that unzips the tensorflow model zip into the variables
	 * folder and .pb file, if they are saved in a zip
	 * @throws IOException if there is any error unzipping
	 */
	private void unzipTfWeights() throws IOException {
		if (descriptor.getWeights().getAllSuportedWeightNames()
				.contains(EngineInfo.getBioimageioTfKey())
				&& !(new File(this.modelsDir, "variables").isDirectory())) {
			String source = descriptor.getWeights().gettAllSupportedWeightObjects().stream()
					.filter(ww -> ww.getFramework().equals(EngineInfo.getBioimageioTfKey()))
					.findFirst().get().getSource();
			try {
				source = FileDownloader.getFileNameFromURLString(source);
			} catch (Exception ex) {}
			System.out.println("Unzipping model...");
			unzippingConsumer.accept(0.);
			ZipUtils.unzipFolder(this.modelsDir + File.separator + source, this.modelsDir,
					this.unzippingConsumer);
		}
		unzip = false;
	}

	/**
	 * Get the final size of the downloadable model by getting the size of 
	 * all the links that are going to be downloaded
	 * @param recalculate
	 * 	whether to recalculate the file size or not
	 * @return the total size to be downloaded in bytes
	 * @throws MalformedURLException if any of the urls for the files in the rdf.yaml is not correct
	 */
	public LinkedHashMap<String, Long> getModelSizeFileByFile(boolean recalculate) throws MalformedURLException {
		if (map != null && !recalculate) {
			return map;
		}
		map = new LinkedHashMap<String, Long>();
		for (String link : getListOfLinks()) {
			map.put(link, FileDownloader.getFileSize(new URL(link)));
		}
		return map;
	}
	
	/**
	 * REtrieve a formated string containing each of the files that have been
	 * downloaded
	 * @return string containing each of the files that have been
	 * downloaded
	 */
	public String getProgress() {
		return progressString;
	}

	/**
	 * 
	 * @return a consumer that tracks the progress of unzipping the
	 * model weights if they are stored in a zip file. It returns the 
	 * fraction of the file that has been unzipped
	 */
	public Consumer<Double> getUnzippingConsumer() {
		return unzippingConsumer;
	}
	
	/**
	 * 
	 * @return whether the model needs unzipping to be done or not, 
	 * 	or it has already been done
	 */
	public boolean needsUnzipping() {
		return unzip;
	}
	
	/**
	 * 
	 * @return the progress unzipping the tensorflow model .zip file 
	 * if it exists, as a fraction of what has already been unziiped over the total
	 */
	public double getUnzippingProgress() {
		return unzippingProgress;
	}
}
