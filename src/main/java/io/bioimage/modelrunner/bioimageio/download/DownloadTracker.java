package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Class that contains the methods to track the progress downloading files.
 * It can be used to track the download of any file or list of files. In addition
 * there is a special constructor to track more easily the dowload of Bioimage.io models.
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class DownloadTracker {
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
	 * Total size of the files to download
	 */
	private long totalSize;
	/**
	 * Number of times in a row that the download has been checked without 
	 * change. The progress is checked every {@link #TIME_INTERVAL_MILLIS}
	 */
	private int nTimesNoChange = 0;
	/**
	 * Aux variable used to keep track of the model download
	 */
	private double downloadSize = 0;
	/**
	 * URL to check if the access to zenodo is fine
	 */
	public static final String ZENODO_URL = "https://zenodo.org/record/6559475/files/README.md?download=1";
	/**
	 * Key for the consumer map that has the total progress of the download
	 */
	public static final String TOTAL_PROGRESS_KEY = "total";
	/**
	 * Millisecond time interval that passes between checks of the download.
	 */
	public static final long TIME_INTERVAL_MILLIS = 300;
	
	/**
	 * 
	 * @param consumer
	 * @param sizeFiles
	 * @throws IOException 
	 */
	private DownloadTracker(String folder, TwoParameterConsumer<String, Double> consumer, List<String> links, Thread thread) throws IOException {
		Objects.requireNonNull(folder, "Please provide teh folder where the files are going to be "
				+ "downloaded.");
		Objects.requireNonNull(consumer);
		Objects.requireNonNull(links, "Please provide the links to the files that are going to be downloaded.");
		Objects.requireNonNull(thread);
		for (String link : links) {
			try {
				sizeFiles.put(folder + File.separator + DownloadModel.getFileNameFromURLString(link), 
						DownloadModel.getFileSize(new URL(link)));
			} catch (MalformedURLException e) {
				throw new IOException("The URL '" + link + "' cannot be found.");
			}
		}
		this.consumer = consumer;
		this.remainingFiles = sizeFiles.keySet().stream().map(i -> new File(i)).collect(Collectors.toList());
		this.downloadThread = thread;
	}
	
	private DownloadTracker(TwoParameterConsumer<String, Double> consumer, DownloadModel dm, Thread thread) throws MalformedURLException {
		Objects.requireNonNull(consumer);
		Objects.requireNonNull(dm);
		Objects.requireNonNull(thread);
		this.consumer = consumer;
		this.dm = dm;
		this.totalSize = dm.getModelSizeFileByFile(false).values().stream()
				.mapToLong(Long::longValue).sum();
		if (this.totalSize < 1)
			this.totalSize = dm.getModelSizeFileByFile(true).values().stream()
			.mapToLong(Long::longValue).sum();
		this.downloadThread = thread;
	}
	
	public static DownloadTracker getBMZModelDownloadTracker(TwoParameterConsumer<String, Double> consumer, DownloadModel dm, Thread thread) throws MalformedURLException {
		return new DownloadTracker(consumer, dm, thread);
	}
	
	public static DownloadTracker getFilesDownloadTracker(String folder, TwoParameterConsumer<String, Double> consumer, 
			List<String> links, Thread thread) throws IOException {
		return new DownloadTracker(folder, consumer, links, thread);
	}
	
	public void track() throws IOException, InterruptedException {
		if (dm == null) {
			trackDownloadOfFilesFromFileSystem();
		} else {
			trackBMZModelDownloadWithDm();
		}
	}
	
	/**
	 * Method that tracks the download of BMZ model files, if the {@link TwoParameterConsumer}
	 * {@link #consumer} retrieves the progress that is being made in the download of the model
	 * @throws IOException if the download stopped
	 * @throws InterruptedException if the current thread is stopped by other threads
	 */
	private void trackBMZModelDownloadWithDm() throws IOException, InterruptedException {
		nTimesNoChange = 0;
		HashMap<String, Long> infoMap = new HashMap<String, Long>();
		while (!trackString.contains(DownloadModel.FINISH_STR) && this.downloadThread.isAlive()) {
			consumer.accept(TOTAL_PROGRESS_KEY, 
					(double) (infoMap.values().stream().mapToLong(Long::longValue).sum()) / (double) this.totalSize);
			didDownloadStop();
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
					endInd != -1 ? endInd : infoStr.length()).trim();
			long fileSize = Long.parseLong(fileSizeStr);
			double progress = (new File(file).length()) / (double) fileSize;
			if (consumer != null && errInd == -1) {
				infoMap.put(file, new File(file).length());
				consumer.accept(file, progress);
			} else if (consumer != null && (errInd != -1 && (endInd == -1 || errInd < endInd))) {
				consumer.accept(file, 0.0);
				trackString += infoStr.substring(0, endInd + DownloadModel.DOWNLOAD_ERROR_STR.length());
				continue;
			}
			if (endInd != -1 && (errInd == -1 || errInd > endInd))
				trackString += infoStr.substring(0, endInd + DownloadModel.END_DWNLD_STR.length());			
			Thread.sleep(TIME_INTERVAL_MILLIS);
		}
		consumer.accept(TOTAL_PROGRESS_KEY, 
				(double) (infoMap.values().stream().mapToLong(Long::longValue).sum()) / (double) this.totalSize);
	}
	
	public void trackDownloadOfFilesFromFileSystem() throws IOException, InterruptedException {
		nTimesNoChange = 0;
		downloadSize = 0;
		long totalDownloadSize = 0;
		while (this.downloadThread.isAlive() && remainingFiles.size() > 0) {
			Thread.sleep(TIME_INTERVAL_MILLIS);
			for (int i = 0; i < this.remainingFiles.size(); i ++) {
				File ff = remainingFiles.get(i);
				if (ff.isFile() && ff.length() != this.sizeFiles.get(ff.getAbsolutePath())){
					consumer.accept(ff.getAbsolutePath(), (double) (ff.length()) / (double) this.sizeFiles.get(ff.getAbsolutePath()));
					consumer.accept(TOTAL_PROGRESS_KEY, (double) (totalDownloadSize + ff.length()) / (double) totalSize);
					break;
				} else if (remainingFiles.get(i).isFile()) {
					consumer.accept(ff.getAbsolutePath(), 1.0);
					totalDownloadSize += ff.length();
					consumer.accept(TOTAL_PROGRESS_KEY, (double) (totalDownloadSize) / (double) totalSize);
					remainingFiles.remove(i);
					break;
				}
			}
			didDownloadStop();
		}
	}
	
	private void didDownloadStop() throws IOException {
		double totDownload = consumer.get().get(TOTAL_PROGRESS_KEY);
		if (downloadSize != totDownload) {
			downloadSize = totDownload;
			nTimesNoChange = 0;
		} else {
			nTimesNoChange += 1;
		}
		if (nTimesNoChange > 30 && !checkInternet(ZENODO_URL)) {
			this.downloadThread.interrupt();
			throw new IOException("The download seems to have stopped. There has been no "
					+ "progress during more than 10 seconds. The internet connection seems unstable.");
		} else if (nTimesNoChange > 60) {
			this.downloadThread.interrupt();
			throw new IOException("The download seems to have stopped. There has been no "
					+ "progress during more than 20 seconds, please review your internet connection or computer permissions");
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
	 * Create consumer used to be used with the {@link DownloadTracker}.
	 * This consumer will be where the info about the files downloaded is written.
	 * The key will be the name of the file and the value the size in bytes already
	 * downloaded
	 * @return a consumer to track downloaded files
	 */
	public static TwoParameterConsumer<String, Long> createConsumerTotalBytes() {
		return new TwoParameterConsumer<String, Long>();
	}
	
	/**
	 * Create consumer used to be used with the {@link DownloadTracker}.
	 * This consumer will be where the info about the files downloaded is written.
	 * The key will be the name of the file and the value the porcentage of
	 * the file already downloaded.
	 * @return a consumer to track downloaded files
	 */
	public static TwoParameterConsumer<String, Double> createConsumerProgress() {
		return new TwoParameterConsumer<String, Double>();
	}

}
