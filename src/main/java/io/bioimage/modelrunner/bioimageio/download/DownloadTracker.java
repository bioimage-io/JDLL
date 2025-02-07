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
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.versionmanagement.JarInfo;

/**
 * Class that contains the methods to track the progress downloading files. 
 * The files have to be downloaded to the same folder.
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
	LinkedHashMap<String, Long> sizeFiles;
	/**
	 * Files that have not been downloaded yet
	 */
	List<File> remainingFiles;
	/**
	 * Thread where the download is being done
	 */
	private Thread downloadThread;
	/**
	 * Thread that called the whole model download method
	 */
	private Thread parentThread;
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
	 * List of all the links that are going to be downloaded
	 */
	private List<String> links;
	/**
	 * Folder where the files are going to be downloaded
	 */
	private String folder;
	/**
	 * URL to check if the access to zenodo is fine
	 */
	public static final String ZENODO_URL = "https://zenodo.org/record/6559475/files/README.md?download=1";
	/**
	 * Key for the consumer map that has the total progress of the download
	 */
	public static final String TOTAL_PROGRESS_KEY = "total";
	/**
	 * Key in the consumer map that specifies the progress unzipping a file,
	 * if there is any file to unzip
	 */
	public static final String UNZIPPING_PROGRESS_KEY = "unzipping progress";
	/**
	 * Millisecond time interval that passes between checks of the download.
	 */
	public static final long TIME_INTERVAL_MILLIS = 300;
	
	/**
	 * Create a download tracker taht can be used to get info about how a download is progressing
	 * 
	 * @param folder
	 * 	folder where the files specifies in the urls are going to be downloaded to.
	 * 	All the files need to be directly downloaded inside this folder. In order for the
	 * 	tracking to work, they cannot be downloaded to subfolders inside this folder
	 * @param consumer
	 * 	consumer that provides info bout the download
	 * @param links
	 * 	string links that correspond to the urls of the files that are going to be downloaded
	 * @param thread
	 * 	thread where the download is happening
	 * @throws IOException if there i any error related to the download
	 */
	private DownloadTracker(String folder, TwoParameterConsumer<String, Double> consumer, List<String> links, Thread thread) throws IOException {
		Objects.requireNonNull(folder, "Please provide the folder where the files are going to be "
				+ "downloaded.");
		Objects.requireNonNull(consumer);
		Objects.requireNonNull(links, "Please provide the links to the files that are going to be downloaded.");
		Objects.requireNonNull(thread);
		this.parentThread = Thread.currentThread();
		this.folder = folder;
		sizeFiles = new LinkedHashMap<String, Long>();
		JarInfo jarInfo = JarInfo.getInstance();
		for (String link : links) {
			String key = folder + File.separator + FileDownloader.getFileNameFromURLString(link);
			if (consumer.get().get(key + EngineInstall.NBYTES_SUFFIX) != null && consumer.get().get(key + EngineInstall.NBYTES_SUFFIX) != -1) {
				sizeFiles.put(key, consumer.get().get(key + EngineInstall.NBYTES_SUFFIX).longValue());
				continue;
			}
			try {
				sizeFiles.put(key, (jarInfo.get(link) != null ? jarInfo.get(link) : FileDownloader.getFileSize(new URL(link))));
			} catch (MalformedURLException e) {
				throw new IOException("The URL '" + link + "' cannot be found.");
			}
		}
		this.totalSize = sizeFiles.values().stream().mapToLong(Long::longValue).sum();
		this.links = links;
		this.consumer = consumer;
		this.remainingFiles = sizeFiles.keySet().stream().map(i -> new File(i)).collect(Collectors.toList());
		this.downloadThread = thread;
		this.consumer.accept(TOTAL_PROGRESS_KEY, 0.0);
	}

	
	/**
	 * Create an object that tracks the download of the files cresponding to a Bioimage.io
	 * model
	 * 
	 * @param consumer
	 * 	consumer used to communicate the progress of the download
	 * @param dm
	 * 	object that manages the download of a Bioimage.io model
	 * @param thread
	 * 	thread where the download is happening
	 * @throws MalformedURLException if any of the URL links is invalid
	 */
	private DownloadTracker(TwoParameterConsumer<String, Double> consumer, DownloadModel dm, Thread thread) throws MalformedURLException {
		Objects.requireNonNull(consumer);
		Objects.requireNonNull(dm);
		Objects.requireNonNull(thread);
		this.parentThread = Thread.currentThread();
		this.consumer = consumer;
		this.dm = dm;
		this.links = dm.getListOfLinks();
		this.folder = dm.getModelFolder();
		this.totalSize = dm.getModelSizeFileByFile(false).values().stream()
				.mapToLong(Long::longValue).sum();
		if (this.totalSize < 1)
			this.totalSize = dm.getModelSizeFileByFile(true).values().stream()
			.mapToLong(Long::longValue).sum();
		this.sizeFiles = dm.getModelSizeFileByFile(false);
		this.downloadThread = thread;
	}
	
	/**
	 * Create an object that tracks the download of the files cresponding to a Bioimage.io
	 * model
	 * 
	 * @param consumer
	 * 	consumer used to communicate the progress of the download
	 * @param dm
	 * 	object that manages the download of a Bioimage.io model
	 * @param thread
	 * 	thread where the download is happening
	 * @return the object used to track teh download. I order to track, execute in another
	 * 	thread tracker.track()
	 * @throws MalformedURLException if any of the URL links is invalid
	 */
	public static DownloadTracker getBMZModelDownloadTracker(TwoParameterConsumer<String, Double> consumer, DownloadModel dm, Thread thread) throws MalformedURLException {
		return new DownloadTracker(consumer, dm, thread);
	}

	/**
	 * Create a download tracker taht can be used to get info about how a download is progressing
	 * 
	 * @param folder
	 * 	folder where the files specifies in the urls are going to be downloaded to.
	 * 	All the files need to be directly downloaded inside this folder. In order for the
	 * 	tracking to work, they cannot be downloaded to subfolders inside this folder
	 * @param consumer
	 * 	consumer that provides info bout the download
	 * @param links
	 * 	string links that correspond to the urls of the files that are going to be downloaded
	 * @param thread
	 * 	thread where the download is happening
	 * @return the object used to track teh download. I order to track, execute in another
	 * 	thread tracker.track()
	 * @throws IOException if there i any error related to the download
	 */
	public static DownloadTracker getFilesDownloadTracker(String folder, TwoParameterConsumer<String, Double> consumer, 
			List<String> links, Thread thread) throws IOException {
		return new DownloadTracker(folder, consumer, links, thread);
	}
	
	/**
	 * Method used to start tracking the progress made by a download. In order to use it 
	 * efectively, it should be launch in a separate thread
	 * @throws IOException if there is any error in the download
	 * @throws InterruptedException if the thread is interrupted abruptly
	 */
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
		boolean alive = true;
		boolean keep = true;
		while (this.parentThread.isAlive() && (!trackString.contains(DownloadModel.FINISH_STR) && alive)) {
			if (!keep)
				alive = false;
			if (!this.downloadThread.isAlive())
				keep = false;
			Thread.sleep(TIME_INTERVAL_MILLIS);
			consumer.accept(TOTAL_PROGRESS_KEY, 
					(double) (infoMap.values().stream().mapToLong(Long::longValue).sum()) / (double) this.totalSize);
			didDownloadStop();
			String progressStr = "" + dm.getProgress();
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
		}
		consumer.accept(TOTAL_PROGRESS_KEY, 
				(double) (infoMap.values().stream().mapToLong(Long::longValue).sum()) / (double) this.totalSize);
		while (dm.needsUnzipping()) {
			Thread.sleep(TIME_INTERVAL_MILLIS);
			consumer.accept(UNZIPPING_PROGRESS_KEY, dm.getUnzippingProgress());
		}
		consumer.accept(UNZIPPING_PROGRESS_KEY, 1.);
	}
	
	/**
	 * Method to track the dowload of any file or list of files in a specific folder
	 * @throws IOException if there is any error with the download
	 * @throws InterruptedException if the thread is stopped abruptly
	 */
	private void trackDownloadOfFilesFromFileSystem() throws IOException, InterruptedException {
		nTimesNoChange = 0;
		downloadSize = 0;
		long totalDownloadSize = 0;
		boolean keep = true;
		
		while (this.parentThread.isAlive() && 
				((this.downloadThread.isAlive() && remainingFiles.size() > 0) || keep)) {
			Thread.sleep(keep == false ? TIME_INTERVAL_MILLIS: 1);
			keep = false;
			for (int i = 0; i < this.remainingFiles.size(); i ++) {
				File ff = remainingFiles.get(i);
				Long storedValue = this.sizeFiles.get( ff.getAbsolutePath() );
				long fileSize = storedValue != null ? storedValue : -1;
				if ( ff.isFile() && ff.length() < fileSize )
				{
					consumer.accept( ff.getAbsolutePath(), Math.min( 1, ( double ) ( ff.length() ) / fileSize ) );
					consumer.accept(TOTAL_PROGRESS_KEY, (double) (totalDownloadSize + ff.length()) / (double) totalSize);
					break;
				} else if (remainingFiles.get(i).isFile()) {
					consumer.accept(ff.getAbsolutePath(), 1.0);
					totalDownloadSize += fileSize;
					consumer.accept(TOTAL_PROGRESS_KEY, (double) (totalDownloadSize) / (double) totalSize);
					remainingFiles.remove(i);
					keep = true;
					break;
				}
				else if ( fileSize == -1 )
				{
					consumer.accept(ff.getAbsolutePath(), -0.01);
					remainingFiles.remove(i);
					keep = true;
					break;
				}
			}
			didDownloadStop();
		}
	}
	
	/**
	 * Check whether the download is stalled or not.
	 * If it has stopped, it also stops the thread of the download
	 * @throws IOException if the download has stopped without any notice
	 */
	private void didDownloadStop() throws IOException {
		if (consumer.get().get(TOTAL_PROGRESS_KEY) != null && downloadSize != consumer.get().get(TOTAL_PROGRESS_KEY)) {
			downloadSize = consumer.get().get(TOTAL_PROGRESS_KEY);
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
	
	/**
	 * Method that is useful to run after the download has completed to check if all
	 * the intended files have been correctly downloaded
	 * @return a list of the links that have not been downloaded correctly
	 */
	public List<String> findMissingDownloads() {
		List<String> badDownloads = new ArrayList<String>();
		if (links == null)
			return badDownloads;
		else if (this.sizeFiles == null || this.sizeFiles.keySet().size() == 0)
			return links;
		for (String link : links) {
			try {
				String fName = folder + File.separator + FileDownloader.getFileNameFromURLString(link);
				if (!(new File(fName).isFile())) {
					badDownloads.add(link);
					continue;
				}
				String key = this.sizeFiles.get(fName) != null ? fName : link;
				Long val = this.sizeFiles.get(key);
				long fSize = new File(fName).length();
				if (val != null && 
						((val > 1 && val == fSize) || (val == 1 && val <= fSize)) ) {
					continue;
				}
				badDownloads.add(link);
			} catch (MalformedURLException e) {
				badDownloads.add(link);
			}
		}
		return badDownloads;
	}
	
	/**
	 * Check whether a specific url is accessible or not
	 * @param urlStr
	 * 	the url of interest
	 * @return true if accessible or false otherwise
	 */
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
	
	/**
	 * Method that tracks the progress of a download happening in the 
	 * thread used as the first parameter and being tracked by the consumer
	 * used as the second parameter.
	 * The teminal output should look like the following for every file:
	 * 	file1.txt: [#######...................] 10%
	 * 
	 * @param downloadThread
	 * 	thread where the download is happening, when it stops the tracking stops
	 *  too
	 * @param consumer
	 * 	consumer that provides the info about the download
	 * @throws InterruptedException if the download is interrupted
	 */
	public static void printProgress(Thread downloadThread,
			DownloadTracker.TwoParameterConsumer<String, Double> consumer) throws InterruptedException {
		Set<String> already = new HashSet<String>();
		while (Thread.currentThread().isAlive() && downloadThread.isAlive()) {
			long waitMillis = 0;
			while ((consumer.get().get(TOTAL_PROGRESS_KEY) == null || consumer.get().get(TOTAL_PROGRESS_KEY) < 1.0)
					&& waitMillis < 3000) {
				Thread.sleep(30);
				waitMillis += 30;
			}
			for (String key : consumer.get().keySet()) {
				if (already.contains(key) || key.equals(DownloadTracker.TOTAL_PROGRESS_KEY) 
						|| key.contains(EngineInstall.NBYTES_SUFFIX)) {
					continue;
				}
				double progress = consumer.get().get(key);
				System.out.println(getStringToPrintProgress(key, progress));
				if (progress == 1) already.add(key);
				else break;
			}
			double progress = consumer.get().get(DownloadTracker.TOTAL_PROGRESS_KEY);
			System.out.println(getStringToPrintProgress(DownloadTracker.TOTAL_PROGRESS_KEY, progress));
		}
	}
	
	private static String getStringToPrintProgress(String kk, double progress) {
		int n = 30;
		int nProgressBar = (int) (progress * n);
		String progressStr = new File(kk).getName() + ": [";
		for (int i = 0; i < nProgressBar; i ++) progressStr += "#";
		for (int i = nProgressBar; i < n; i ++) progressStr += ".";
		progressStr += "] " + Math.round(progress * 100) + "%";
		return progressStr;
	}

}
