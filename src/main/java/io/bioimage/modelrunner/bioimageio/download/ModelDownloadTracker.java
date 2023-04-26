package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.stream.Collectors;


public class ModelDownloadTracker {
	/**
	 * Consumer used to report the download progress to the main thread
	 */
	TwoParameterConsumer<String, Double> consumer;
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
	private DownloadModel dm;
	/**
	 * String that tracks the model downoad progress String
	 */
	private String trackString = "";
	
	/**
	 * URL to check if the access to zenodo is fine
	 */
	public static final String ZENODO_URL = "https://zenodo.org/record/6559475/files/README.md?download=1";
	
	/**
	 * 
	 * @param consumer
	 * @param sizeFiles
	 */
	public ModelDownloadTracker(TwoParameterConsumer<String, Double> consumer, HashMap<String, Long> sizeFiles, Thread thread) {
		this.consumer = consumer;
		this.sizeFiles = sizeFiles;
		this.remainingFiles = sizeFiles.keySet().stream().map(i -> new File(i)).collect(Collectors.toList());
		this.downloadThread = thread;
	}
	
	public ModelDownloadTracker(TwoParameterConsumer<String, Double> consumer, DownloadModel dm, Thread thread) {
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
	
	/**
	 * Method that tracks the download of BMZ model files, if the {@link TwoParameterConsumer}
	 * {@link #consumer} retrieves the progress that is being made in the download of the model
	 */
	private void trackBMZModelDownloadWithDm() {
		while (!trackString.contains(DownloadModel.FINISH_STR) && this.downloadThread.isAlive()) {
			String progressStr = dm.getProgress();
			String infoStr = progressStr.substring(trackString.length());
			int startInd = infoStr.indexOf(DownloadModel.START_DWNLD_STR);
			int endInd = infoStr.indexOf(DownloadModel.END_DWNLD_STR);
			int finishInd = infoStr.indexOf(DownloadModel.FINISH_STR);
			int errInd = infoStr.indexOf(DownloadModel.DOWNLOAD_ERROR_STR);
			if (endInd == -1 && startInd == -1 && finishInd == -1)
				continue;
			else if (startInd == -1 && endInd == -1) {
				trackString = progressStr;
				continue;
			} 
			int fileSizeInd = infoStr.indexOf(DownloadModel.FILE_SIZE_STR);
			String file = infoStr.substring(startInd + DownloadModel.START_DWNLD_STR.length(), 
					fileSizeInd).trim();
			String fileSizeStr = infoStr.substring(fileSizeInd + DownloadModel.FILE_SIZE_STR.length(),
					endInd).trim();
			long fileSize = Long.parseLong(fileSizeStr);
			double progress = (new File(file).length()) / fileSize;
			if (consumer != null && errInd == -1)
				consumer.accept(file, progress);
			else if (consumer != null && (errInd != -1 && (endInd == -1 || errInd < endInd))) {
				consumer.accept(file, 0.0);
				trackString += infoStr.substring(0, endInd + DownloadModel.DOWNLOAD_ERROR_STR.length());
				continue;
			}
			if (endInd != -1 && (errInd == -1 || errInd > endInd))
				trackString += infoStr.substring(0, endInd + DownloadModel.END_DWNLD_STR.length());			
		}
	}
	
	public void trackDownloadofFilesFromFileSystem() {
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
	
	/**
	 * Functional interface to create a consumer that accepts two args and
	 * can be used to retrieve an underlying map
	 * @author Carlos Garcia Lopez de Haro
	 *
	 * @param <T>
	 * 	key 
	 * @param <U>
	 * 	value
	 */
	public static class TwoParameterConsumer<T, U> {
		/**
		 * Map where the values are stored
		 */
		private LinkedHashMap<T, U> map = new LinkedHashMap<T, U>();
		
		/**
		 * Add the key value pair
		 * @param t
		 * 	key
		 * @param u
		 * 	value
		 */
	    public void accept(T t, U u) {
	        map.put(t, u);
	    }
	    
	    /**
	     * Retrieve the map
	     * @return the map
	     */
	    public LinkedHashMap<T, U> get() {
	    	return map;
	    }
	}
	
	/**
	 * Create consumer used to be used with the {@link ModelDownloadTracker}.
	 * This consumer will be where the info about the files downloaded is written.
	 * The key will be the name of the file and the value the size in bytes already
	 * downloaded
	 * @return a consumer to track downloaded files
	 */
	public static TwoParameterConsumer<String, Long> createConsumerTotalBytes() {
		return new TwoParameterConsumer<String, Long>();
	}
	
	/**
	 * Create consumer used to be used with the {@link ModelDownloadTracker}.
	 * This consumer will be where the info about the files downloaded is written.
	 * The key will be the name of the file and the value the porcentage of
	 * the file already downloaded.
	 * @return a consumer to track downloaded files
	 */
	public static TwoParameterConsumer<String, Double> createConsumerProgress() {
		return new TwoParameterConsumer<String, Double>();
	}

}
