package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.SampleImage;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormatInterface;
import io.bioimage.modelrunner.utils.Log;

/**
 * Class to manage the downloading of models from the BioImage.io
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class DownloadableModel {
	/**
	 * Map that contain all the links needed to download a BioImage.io model
	 */
	private Map<String, String> downloadableLinks;
	/**
	 * Model descriptor containing all the information about the model
	 */
	private ModelDescriptor descriptor;
	/**
	 * String path to the folder where the model has been downloaded
	 */
	private String modelFolder;
	/**
	 * Key for the map that contains the download URL of the model 
	 */
	private static String downloadURLKey = "download_url";
	/**
	 * Key for the map that contains the attachments of the model 
	 */
	private static String attachmentsKey = "attachments";
	/**
	 * Key for the map that contains the weights of the model 
	 */
	private static String weightsKey = "weights";
	/**
	 * Key for the map that contains the rdf URL of the model 
	 */
	private static String rdfKey = "rdf_source";
	/**
	 * Key for the map that contains the sample inputs of the model 
	 */
	private static String sampleInputsKey = "sample_input";
	/**
	 * Key for the map that contains the sample outputs of the model 
	 */
	private static String sampleOutputsKey = "sample_output";
	/**
	 * Key for the map that contains the test inputsof the model 
	 */
	private static String testInputsKey = "test_input";
	/**
	 * Key for the map that contains the test outputs of the model 
	 */
	private static String testOutputsKey = "test_output";
	/**
	 * Name of the folder where the models are downloaded and stored
	 */
	private static String MODELS_PATH = new File("models").getAbsolutePath();
    /**
     * Consumer used to send info about the download to other threads
     */
    private Consumer<String> consumer;
	
	/**
	 * Constructor that contains all the info and is able to download a BioImage.io model
	 * @param descriptor
	 * 	information about the model from the rdf.yaml
	 */
	private DownloadableModel(ModelDescriptor descriptor) {
		this.descriptor = descriptor;
		retriveDownloadModelLinks();
	}

	/**
	 * Build a constructor that contains all the info and is able to download a BioImage.io model
	 * @param descriptor
	 * 	information about the model from the rdf.yaml
	 */
	public static DownloadableModel build(ModelDescriptor descriptor) {
		return new DownloadableModel(descriptor);
	}
	
	/**
	 * Set a consumer to provide info about the download to other threads
	 * @param consumer
	 * 	consumer where info will be sent
	 */
	public void setConsumer(Consumer<String> consumer) {
		this.consumer = consumer;
	}
	
	/**
	 * Retrieve the list of strings (URLs) that need to be donloaded to create the 
	 * BioImage.io model specified in the rdf.yaml
	 */
	private void retriveDownloadModelLinks() {
		downloadableLinks = new HashMap<String, String>();
		String downloadURL = descriptor.getDownloadUrl();
		if (downloadURL != null && checkURL(downloadURL)) {
			downloadableLinks.put(downloadURLKey, downloadURL);
			return;
		}
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
	 * @throws Exception if the model is not correctly downloaded
	 */
	public void checkModelWasDownloaded() throws Exception {
		// TODO
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
				WeightFormatInterface w = weights.getWeightsByIdentifier(ww);
				if (w.getSource() != null && checkURL(w.getSource())) {
					downloadableLinks.put(weightsKey + "_" + c ++, w.getSource());
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
				downloadableLinks.put(testInputsKey + "_" + c ++, ss);
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
				downloadableLinks.put(testOutputsKey + "_" + c ++, ss);
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
				downloadableLinks.put(sampleInputsKey + "_" + c ++, ss.getString());
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
				downloadableLinks.put(sampleOutputsKey + "_" + c ++, ss.getString());
			}
		}
	}
	
	/**
	 * Add the rdf.yaml file to the downloadable links of the model
	 */
	private void addRDF() {
		String rdf = descriptor.getRDFSource();
		if (rdf != null && checkURL(rdf)) {
			downloadableLinks.put(rdfKey, rdf);
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
				downloadableLinks.put(attachmentsKey + "_" + c ++, (String) attachments.get(kk));
			} else if (attachments.get(kk) instanceof URL) {
				downloadableLinks.put(attachmentsKey + "_" + c ++, ((URL) attachments.get(kk)).toString());
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
	 * Download the model from the BioImage.io insicating the progress that is being made file
	 * by file
	 * @throws IOException if there is any error creating the folder or downloading the files
	 * @throws InterruptedException if the thread was stopped by the user
	 */
	public void downloadModelAsFiles() throws IOException, InterruptedException {
		boolean downloadZip = downloadableLinks.keySet().contains(downloadURLKey);
		if (downloadZip) {
			downloadZipModel();
		} else {
			downloadModelFileByFile(false);
		}
	}
	
	/**
	 * Download the model from the BioImage.io indicating the progress of the download as a whole
	 * @throws IOException if there is any error creating the folder or downloading the files
	 * @throws InterruptedException if the thread was stopped by the user
	 */
	public void downloadModelAsWhole() throws IOException, InterruptedException {
		downloadModelFileByFile(true);
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
		File ff = new File(str);
		String fileName = ff.getName();
		int extensionPos = fileName.lastIndexOf(".");
		if (extensionPos == -1)
			return fileName + "_" + dateString;
		String nameNoExtension = fileName.substring(0, extensionPos);
		String extension = fileName.substring(extensionPos);
		return nameNoExtension + "_" + dateString + extension;
	}
	
	/**
	 * Download a model downloading one by one all the files that should be inside
	 * the model folder into a created folder inside the models repo
	 * @param asWhole
	 * 	whether the model download is notifies file by file or as a whole
	 * @throws IOException if there is any error creating the folder or downloading the files
	 * @throws InterruptedException if the thread was stopped by the user
	 */
	private void downloadModelFileByFile(boolean asWhole) throws IOException, InterruptedException {
		modelFolder = MODELS_PATH + File.separator + addTimeStampToFileName(descriptor.getName());
		File folder = new File(modelFolder);
		boolean created = folder.mkdirs();
		if (!created)
			throw new IOException("Unable to create model folder ->" + modelFolder);
		if (asWhole)
			createUniqueTrackerForAllModelFiles();
		for (int i = 0; i < getListOfLinks().size(); i ++) {
        	if (Thread.interrupted())
                throw new InterruptedException("Interrupted before downloading the remaining files: "
            		+ Arrays.toString(IntStream.range(i, getListOfLinks().size())
            									.mapToObj(j -> getListOfLinks().get(j)).toArray()));
			String item = getListOfLinks().get(i);
			String fileName = getFileNameFromURLString(item);
			if (!asWhole)
				createSingleFileDownloaderTracker(item);
			downloadFileFromInternet(item, new File(modelFolder, fileName));
		}
		
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
		int ind = str.lastIndexOf("/");
		return str.substring(ind + 1);
	}
	
	/**
	 * Method that downloads the model selected from the internet,
	 * copies it and unzips it into the models folder
	 * @throws InterruptedException 
	 */
	public void downloadFileFromInternet(String downloadURL, File targetFile) throws InterruptedException {
		FileOutputStream fos = null;
		ReadableByteChannel rbc = null;
		// TODO if apache works fine, get rid of the other download class
		try {
			URL website = new URL(downloadURL);
			rbc = Channels.newChannel(website.openStream());
			// Create the new model file as a zip
			fos = new FileOutputStream(targetFile);
			// Send the correct parameters to the progress screen
			FileDownloader downloader = new FileDownloader(rbc, fos);
			System.out.println(Log.getCurrentTime() + " -- Downloading " + targetFile.getName());
			// TODO remove FileUtils.copyURLToFile(website, targetFile);
			downloader.call();
		} catch (IOException e) {
			MessageDialog.showDialog("The link for the file: " + targetFile.getName() 
									+ " is broken.\nDeepIcy will continue with the "
									+ "download but the model might be downloaded incorrectly.", MessageDialog.ERROR_MESSAGE);
			e.printStackTrace();
		} finally {
			fdt = null;
			try {
				if (fos != null)
						fos.close();
				if (rbc != null)
					rbc.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

        if (Thread.interrupted()) {
            throw new InterruptedException(
                    "Interrupted while downloading: " + targetFile.getName());
        }
	}
	
	/**
	 * Create a tracker that monitors the download of all the files of a 
	 * model
	 * @throws MalformedURLException if any of the URLs for the files is incorrect
	 */
	private void createUniqueTrackerForAllModelFiles() throws MalformedURLException {
		String msg = System.lineSeparator() + Log.getCurrentTime() + " -- Getting size"
				+ " of the files to be downloaded." + System.lineSeparator()
				+ "  If it takes too long maybe the connection is unstable. Try: "
				+ System.lineSeparator() + "  " + getListOfLinks().get(0);
		System.out.println(msg);
		long totSize = 0;
		for (String item : getListOfLinks()) {
			URL website = new URL(item);
			totSize += getFileSize(website);
		}
		setFileDownloaderTracker(modelFolder, totSize);
	}
	
	/**
	 * Create a tracker that monitors the download of the file defined by the input
	 * parameter, that should always correspond to an URL
	 * @param downloadURL
	 * 	the url for the given file
	 * @throws MalformedURLException if there is an error in the url of the file
	 */
	private void createSingleFileDownloaderTracker(String downloadURL) throws MalformedURLException {
		Objects.requireNonNull(downloadURL);
		Objects.requireNonNull(modelFolder);
		String fileName = getFileNameFromURLString(downloadURL);
		String msg = System.lineSeparator() + Log.getCurrentTime() + " -- Getting size"
				+ " of the file to be downloaded." + System.lineSeparator()
				+ "  If it takes too long maybe the connection is unstable. Try: "
				+ System.lineSeparator() + "  " +  downloadURL;
		System.out.println(msg);
		URL website = new URL(downloadURL);
		setFileDownloaderTracker(new File(modelFolder, fileName).getAbsolutePath(), getFileSize(website));
	}
	
	/**
	 * Get the final size of the downloadable model by getting the size of 
	 * all the links that are going to be downloaded
	 * @return the total size to be downloaded in bytes
	 * @throws MalformedURLException 
	 */
	public HashMap<String, Long> getModelSizeFileByFile() throws MalformedURLException {
		HashMap<String, Long> map = new HashMap<String, Long>();
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
	 * MEthod that returns the directory that corresponds to the model downloaded
	 * @return the directory corresponding to the model downloaded
	 */
	public String getModelFolder(){
		return this.modelFolder;
	}
}
