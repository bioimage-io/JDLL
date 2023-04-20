package io.bioimage.modelrunner.bioimageio.download;

import java.util.HashMap;
import java.util.UUID;
import java.util.function.Consumer;

public class ModelDownloadTracker {
	/**
	 * Consumer used to report the download progress to the main thread
	 */
	Consumer<String> consumer;
	/**
	 * Map containing the size of each of the files to be downloaded
	 */
	HashMap<String, Long> sizeFiles;
	/**
	 * Whether a file or a folder are being downloaded
	 */
	private boolean isFile = false;
	/**
	 * Thread where the download is being done
	 */
	private Thread downloadThread;
	/**
	 * Unique identifier associated to the download
	 */
	private final String uniqueIdentifier;
	/**
	 * Name of the folder that is going to be donwloaded
	 */
	public static final String DOWNLOAD_FOLDERNAME = "-* FOLDER NAME *-";
	/**
	 * Name of the file that is going to be donwloaded
	 */
	public static final String DOWNLOAD_FILERNAME = "-* FILE NAME *-";
	/**
	 * Keyword identifying the beginning of a download
	 */
	public static final String START_KEY = "-* START *-";
	/**
	 * Keyword identifying the end of a download
	 */
	public static final String END_KEY = "-* END *-";
	
	/**
	 * 
	 * @param consumer
	 * @param sizeFiles
	 */
	public ModelDownloadTracker(Consumer<String> consumer, HashMap<String, Long> sizeFiles, Thread thread) {
		this.consumer = consumer;
		this.sizeFiles = sizeFiles;
		this.uniqueIdentifier = createID();
		if (sizeFiles.entrySet().size() == 1) {
			isFile = true;
		}
		this.downloadThread = thread;
	}
	
	/**
	 * Create a unique identifier for this download
	 */
	private String createID() {
		return UUID.randomUUID().toString();
	}
	
	/**
	 * REturn the unique identifier associated to the download.
	 * The identifier is used to identify the download in the text passed to the consumer
	 * @return
	 */
	public String getID() {
		return this.uniqueIdentifier;
	}

}
