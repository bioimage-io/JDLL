package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.stream.Collectors;


public class ModelDownloadTracker {
	/**
	 * Consumer used to report the download progress to the main thread
	 */
	Consumer<HashMap<String, Long>> consumer;
	/**
	 * Map containing the size of each of the files to be downloaded
	 */
	HashMap<String, Long> sizeFiles;
	/**
	 * Files that have not been downloaded yet
	 */
	List<File> remainingFiles;
	/**
	 * Thread where the download is being done
	 */
	private Thread downloadThread;
	/**
	 * Class that downloads the files that compose the model
	 */
	private DownloadableModel dm;
	
	/**
	 * URL to check if the access to zenodo is fine
	 */
	public static final String ZENODO_URL = "https://zenodo.org/record/6559475/files/README.md?download=1";
	
	/**
	 * 
	 * @param consumer
	 * @param sizeFiles
	 */
	public ModelDownloadTracker(Consumer<HashMap<String, Long>> consumer, HashMap<String, Long> sizeFiles, Thread thread) {
		this.consumer = consumer;
		this.sizeFiles = sizeFiles;
		this.remainingFiles = sizeFiles.keySet().stream().map(i -> new File(i)).collect(Collectors.toList());
		this.downloadThread = thread;
	}
	
	public ModelDownloadTracker(Consumer<HashMap<String, Long>> consumer, DownloadableModel dm, Thread thread) {
		this.consumer = consumer;
		this.dm = dm;
		this.remainingFiles = sizeFiles.keySet().stream().map(i -> new File(i)).collect(Collectors.toList());
		this.downloadThread = thread;
	}
	
	/**
	 * Create a unique identifier for this download
	 */
	private String createID() {
		return UUID.randomUUID().toString();
	}
	
	public void trackBMZModelDownload() throws IOException {
		if (dm == null) {
			trackDownloadofFilesFromFileSystem();
		} else {
			trackBMZModelDownloadWithDm();
		}
	}
	
	private void trackBMZModelDownloadWithDm() {
		dm.getProgress();
	}
	
	public void trackDownloadofFilesFromFileSystem() throws IOException {
		HashMap<String, Long> infoMap = new HashMap<String, Long>();
		int nTimesWoChange = 0;
		long downloadSize = 0;
		while (this.downloadThread.isAlive() && remainingFiles.size() > 0) {
			for (int i = 0; i < this.remainingFiles.size(); i ++) {
				File ff = remainingFiles.get(i);
				if (ff.isFile() && ff.length() != this.sizeFiles.get(ff.getAbsolutePath())){
					infoMap.put(ff.getAbsolutePath(), ff.length());
					break;
				} else if (remainingFiles.get(i).isFile()) {
					infoMap.put(ff.getAbsolutePath(), ff.length());
					remainingFiles.remove(i);
					break;
				}
			}
			long totDownload = infoMap.values().stream().mapToLong(Long::longValue).sum();
			if (downloadSize != totDownload) {
				downloadSize = totDownload;
				nTimesWoChange = 0;
			} else {
				nTimesWoChange += 1;
			}
			if (nTimesWoChange > 30 && !checkInternet(ZENODO_URL)) {
				throw new IOException("The download seems to have stopped. There has been no "
						+ "progress during more than 10 seconds. The internet connection seems unstable.");
			} else if (nTimesWoChange > 60) {
				throw new IOException("The download seems to have stopped. There has been no "
						+ "progress during more than 20 seconds, please review your internet connection or computer permissions");
			}
		}
	}
	
	public static boolean checkInternet(String urlStr) {
        try {
			URL url = new URL(urlStr);
	        HttpURLConnection connection = (HttpURLConnection)url.openConnection();
	        connection.setRequestMethod("GET");
	        connection.connect();
	        int code = connection.getResponseCode();
	        if (code != 200)
	        	throw new IOException();
        } catch (Exception ex) {
        	return false;
        }
        return true;
    }

}
