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
package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.SampleImage;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.engine.installation.FileDownloader;

/**
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
    private Consumer<String> consumer;
    /**
     * String that logs the file that it is being downloaded at the current moment
     */
    private String progressString = "";
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
	 * Constructor that contains all the info and is able to download a BioImage.io model
	 * @param descriptor
	 * 	information about the model from the rdf.yaml
	 */
	private DownloadModel(ModelDescriptor descriptor, String modelsDir) {
		this.descriptor = descriptor;
		String fname = addTimeStampToFileName(descriptor.getName());
		this.modelsDir = modelsDir + File.separator + getValidFileName(fname);
		this.consumer = (String b) -> {
    		progressString += b;
    		};
		retriveDownloadModelLinks();
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
				fileName = getFileNameFromURLString(link);
				websize = getFileSize(new URL(link));
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
	 * Method that deletes the folder where the model has been downloaded with all
	 * its contents
	 */
	public void deleteModel() {
		// TODO
	}
	
	/**
	 * Add weight links to the downloadable links
	 */
	private void addWeights() {
		ModelWeight weights = descriptor.getWeights();
		int c = 0;
		for (String ww : weights.getEnginesListWithVersions()) {
			try {
				WeightFormat w = weights.getWeightsByIdentifier(ww);
				if (w.getSource() != null && checkURL(w.getSource())) {
					downloadableLinks.put(WEIGHTS_KEY + "_" + c ++, w.getSource());
				}
			} catch (Exception ex) {
				// The exception is thrown whenever the weight format is not present.
				// This exception will not be thrown here because the weight formats are retrieved from the same object
			}
		}
	}
	
	/**
	 * Add the test inputs to the downloadable links
	 */
	private void addTestInputs() {
		List<String> sampleInps = descriptor.getTestInputs();
		if (sampleInps == null)
			return;
		int c = 0;
		for (String ss : sampleInps) {
			if (ss != null && checkURL(ss)) {
				downloadableLinks.put(TEST_INPUTS_KEY + "_" + c ++, ss);
			}
		}
	}
	
	/**
	 * Add the test outputs to the dowloadable links
	 */
	private void addTestOutputs() {
		List<String> sampleOuts = descriptor.getTestOutputs();
		if (sampleOuts == null)
			return;
		int c = 0;
		for (String ss : sampleOuts) {
			if (ss != null && checkURL(ss)) {
				downloadableLinks.put(TEST_OUTPUTS_KEY + "_" + c ++, ss);
			}
		}
	}
	
	/**
	 * Add the sample inputs to the dowloadable links
	 */
	private void addSampleInputs() {
		List<SampleImage> sampleInps = descriptor.getSampleInputs();
		if (sampleInps == null)
			return;
		int c = 0;
		for (SampleImage ss : sampleInps) {
			if (ss != null && ss.getUrl() != null) {
				downloadableLinks.put(SAMPLE_INPUTS_KEY + "_" + c ++, ss.getString());
			}
		}
	}
	
	/**
	 * Add the sample outputs to the dowloadable links
	 */
	private void addSampleOutputs() {
		List<SampleImage> sampleOuts = descriptor.getSampleOutputs();
		if (sampleOuts == null)
			return;
		int c = 0;
		for (SampleImage ss : sampleOuts) {
			if (ss != null && ss.getUrl() != null) {
				downloadableLinks.put(SAMPLE_OUTPUTS_KEY + "_" + c ++, ss.getString());
			}
		}
	}
	
	/**
	 * Add the rdf.yaml file to the downloadable links of the model
	 */
	private void addRDF() {
		String rdf = descriptor.getRDFSource();
		if (rdf != null && checkURL(rdf)) {
			downloadableLinks.put(RDF_KEY, rdf);
		}
	}
	
	/**
	 * Add the attachment files to the downloadable links of the model
	 */
	private void addAttachments() {
		Map<String, Object> attachments = descriptor.getAttachments();
		if (attachments == null)
			return;
		int c = 0;
		for (String kk : attachments.keySet()) {
			if (attachments.get(kk) instanceof String && checkURL((String) attachments.get(kk))) {
				downloadableLinks.put(ATTACH_KEY + "_" + c ++, (String) attachments.get(kk));
			} else if (attachments.get(kk) instanceof URL) {
				downloadableLinks.put(ATTACH_KEY + "_" + c ++, ((URL) attachments.get(kk)).toString());
			}
		}
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
	 */
	public void downloadModel() throws IOException, InterruptedException {
		File folder = new File(modelsDir);
		if (!folder.isDirectory() && !folder.mkdirs())
			throw new IOException("The provided directory where the model is going to "
					+ "be downloaded does not exist and cannot be created ->" + modelsDir);
		for (int i = 0; i < getListOfLinks().size(); i ++) {
        	if (Thread.interrupted())
                throw new InterruptedException("Interrupted before downloading the remaining files: "
            		+ Arrays.toString(IntStream.range(i, getListOfLinks().size())
            									.mapToObj(j -> getListOfLinks().get(j)).toArray()));
			String item = getListOfLinks().get(i);
			String fileName = getFileNameFromURLString(item);
			downloadFileFromInternet(item, new File(modelsDir, fileName));
		}
		consumer.accept(FINISH_STR);
	}
	
	/**
	 * Gets the filename of the file in an URL from the url String
	 * @param str
	 * 	the URL string
	 * @return the file name of the file in the URL
	 * @throws MalformedURLException if the String does not correspond to an URL
	 */
	public static String getFileNameFromURLString(String str) throws MalformedURLException {
		URL url = new URL(str);
		return new File(url.getPath()).getName();
	}
	
	/**
	 * Method that downloads the model selected from the internet,
	 * copies it and unzips it into the models folder
	 * @param downloadURL
	 * 	url of the file to be downloaded
	 * @param targetFile
	 * 	file where the file from the url will be downloaded too
	 */
	public void downloadFileFromInternet(String downloadURL, File targetFile) {
		FileOutputStream fos = null;
		ReadableByteChannel rbc = null;
		try {
			URL website = new URL(downloadURL);
			rbc = Channels.newChannel(website.openStream());
			// Create the new model file as a zip
			fos = new FileOutputStream(targetFile);
			// Send the correct parameters to the progress screen
			FileDownloader downloader = new FileDownloader(rbc, fos);
			consumer.accept(START_DWNLD_STR + targetFile + FILE_SIZE_STR + map.get(downloadURL));
			downloader.call();
			consumer.accept(END_DWNLD_STR);
		} catch (IOException e) {
			consumer.accept(DOWNLOAD_ERROR_STR);
			String msg = "The link for the file: " + targetFile.getName() + " is broken." + System.lineSeparator() 
						+ "JDLL will continue with the download but the model might be "
						+ "downloaded incorrectly.";
			new IOException(msg, e).printStackTrace();
		} finally {
			try {
				if (fos != null)
						fos.close();
				if (rbc != null)
					rbc.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
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
			map.put(link, getFileSize(new URL(link)));
		}
		return map;
	}

	/**
	 * Get the size of the file stored in the given URL
	 * @param url
	 * 	url where the file is stored
	 * @return the size of the file
	 */
	public static long getFileSize(URL url) {
		HttpURLConnection conn = null;
		try {
			conn = (HttpURLConnection) url.openConnection();
			if (conn.getResponseCode() > 300 && conn.getResponseCode() < 304)
				return getFileSize(redirectedURL(url, conn));
			if (conn.getResponseCode() != 200)
				throw new Exception("Unable to connect to: " + url.toString());
			return conn.getContentLengthLong();
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (Exception ex) {
			ex.printStackTrace();
			String msg = "Unable to connect to " + url.toString();
			System.out.println(msg);
			return 1;
		}
	}
	
	/**
	 * This method shuold be used when we get the following response codes from 
	 * a {@link HttpURLConnection}:
	 * - {@link HttpURLConnection#HTTP_MOVED_TEMP}
	 * - {@link HttpURLConnection#HTTP_MOVED_PERM}
	 * - {@link HttpURLConnection#HTTP_SEE_OTHER}
	 * 
	 * If that is not the response code or the connection does not work, the url
	 * returned will be the same as the provided.
	 * If the method is used corretly, it will return the URL to which the original URL
	 * has been redirected
	 * @param url
	 * 	original url. Connecting to that url must give a 301, 302 or 303 response code
	 * @param conn
	 * 	connection to the url
	 * @return the redirected url
	 */
	public static URL redirectedURL(URL url, HttpURLConnection conn) {
		int statusCode;
		try {
			statusCode = conn.getResponseCode();
		} catch (IOException ex) {
			return url;
		}
		if (statusCode != HttpURLConnection.HTTP_MOVED_TEMP
            && statusCode != HttpURLConnection.HTTP_MOVED_PERM
            && statusCode != HttpURLConnection.HTTP_SEE_OTHER)
			return url;
		String newURL = conn.getHeaderField("Location");
		try {
			return new URL(newURL);
		} catch (MalformedURLException ex) {
		}
        try {
			String mainDomain = url.toURI().getHost();
			return new URL(mainDomain + newURL);
		} catch (URISyntaxException | MalformedURLException e) {
			return null;
		}
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
}
