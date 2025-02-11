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
package io.bioimage.modelrunner.download;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.X509Certificate;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.JarInfo;

public class FileDownloader {
	
	private final URL website;
	
	private final File file;
	
	private final boolean printProgress;
	
	private final String name;
	
	private Long fileSize;
	
	private AtomicLong sizeDownloaded = new AtomicLong(0);
	
    private int lost_conn = 0;
	
    private long already = 0;
    
    private boolean complete = false;
	
	private Consumer<Long> totalProgress;
	
	private Consumer<Double> partialProgress;
	
	public void setTotalProgressConsumer(Consumer<Long> totalProgress) {
		this.totalProgress = totalProgress;
	}
	
	public void setPartialProgressConsumer(Consumer<Double> partialProgress) {
		this.partialProgress = partialProgress;
	}
	
	private static final long CHUNK_SIZE = 1024 * 1024 * 5;

	private static final int STALL_THRES = 10000;

	private static final int PROGRESS_INTER = 1000;
	
	private static final int MAX_RETRIES = 3;
	
	public FileDownloader(String url, File file, boolean printProgress) throws MalformedURLException {
		this.website = new URL(url);
		this.file = file;
		this.printProgress = printProgress;
		this.name = getFileNameFromURLString(url);
	}
	
	public FileDownloader(String url, File file) throws MalformedURLException {
		this(url, file, true);
	}
	
	public String getFileName() {
		return this.name;
	}
	
	public long getOnlineFileSize() {
		if (fileSize != null)
			return fileSize;
		try {
			JarInfo jarInfo = JarInfo.getInstance();
			if (jarInfo.get(website.toString()) != null) {
				fileSize = jarInfo.get(website.toString()).longValue();
				return fileSize;
			}
		} catch (IOException e) {
		}
		fileSize = getFileSize(website);
		return fileSize;
	}
	
	public long getSizeDownloaded() {
		return this.sizeDownloaded.get();
	}
	
	public void download(Thread parentThread) throws IOException, ExecutionException {
		already = 0;
		while (lost_conn < MAX_RETRIES && !complete && parentThread.isAlive()) {
			downloadAttempt(parentThread);
		}
		if (!complete) {
			throw new IOException("Unable to download file after " + MAX_RETRIES + " attempts.");
		}
	}
	
	private void downloadAttempt(Thread parentThread) throws IOException, ExecutionException {
		HttpsURLConnection conn = ( HttpsURLConnection ) website.openConnection();
		conn.setConnectTimeout(STALL_THRES);
		conn.setReadTimeout(STALL_THRES);
		if (already > 0) {
			conn.setRequestProperty("Range", "bytes=" + already + "-");
        }
		SSLContext sslContext;
		try {
			sslContext = getAllTrustingSSLContext();
		} catch (KeyManagementException | NoSuchAlgorithmException e) {
			throw new RuntimeException(e);
		}
		conn.setSSLSocketFactory( sslContext.getSocketFactory() );
		
		try (
				InputStream str = conn.getInputStream();
				ReadableByteChannel rbc = Channels.newChannel(str);
				FileOutputStream fos = new FileOutputStream(file, already != 0 ? true : false);
				){
			/**
			 * TODO remove
			Thread downloadThread = new Thread(() -> {
				try {
					call(rbc, fos, parentThread);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
			downloadThread.start();
			
			checkDownloadContinues(parentThread, downloadThread);
			 */
			performDownload(fos, rbc, parentThread);
		}
	}
	
	private void performDownload(FileOutputStream fos, ReadableByteChannel rbc, Thread parentThread) throws ExecutionException {

        ExecutorService downloadExecutor = Executors.newSingleThreadExecutor();
        ScheduledExecutorService monitorExecutor = Executors.newScheduledThreadPool(2);
        this.getOnlineFileSize();

        Callable<Void> downloadTask = () -> {
			call(rbc, fos);
            return null;
        };

        Future<Void> downloadFuture = downloadExecutor.submit(downloadTask);
        final long[] lastProgress = { already };

        Runnable stallDetectionTask = () -> {
            if (!parentThread.isAlive()) {
                downloadFuture.cancel(true);
                return;
            }
            long bytesDownloaded = file.length();
            if (bytesDownloaded == lastProgress[0]) {
                System.err.println("Connection lost. Time number: " + (lost_conn + 1));
                downloadFuture.cancel(true);
            } else {
                lastProgress[0] = bytesDownloaded;
            }
        };

        Runnable progressPrintTask = () -> {
            if (!parentThread.isAlive()) {
                downloadFuture.cancel(true);
                return;
            }
    		long totalPro = file.length();
    		double partialPro = totalPro / (double) this.fileSize;
            if (printProgress)
            	System.out.println(getStringToPrintProgress(file.getName(), partialPro));
            if (totalProgress != null)
            	this.totalProgress.accept(totalPro);
            if (partialProgress != null)
            	this.partialProgress.accept(partialPro);
        };

        monitorExecutor.scheduleAtFixedRate(
                stallDetectionTask, STALL_THRES, STALL_THRES, TimeUnit.MILLISECONDS);
        monitorExecutor.scheduleAtFixedRate(
                progressPrintTask, 0, PROGRESS_INTER, TimeUnit.MILLISECONDS);

        try {
            // Wait for the download task to complete.
            downloadFuture.get();
            complete = true;
        } catch (CancellationException | InterruptedException e) {
            lost_conn ++;
        } finally {
            downloadExecutor.shutdownNow();
            monitorExecutor.shutdownNow();
        }
        finalProgress();
	}
	
	private void finalProgress() {
		long totalPro = file.length();
		double partialPro = totalPro / (double) this.fileSize;
        if (printProgress)
        	System.out.println(getStringToPrintProgress(file.getName(), partialPro));
        if (totalProgress != null)
        	this.totalProgress.accept(totalPro);
        if (partialProgress != null)
        	this.partialProgress.accept(partialPro);
	}
	
	/**
	 * Download a file without the possibility of interrupting the download
	 * @throws IOException if there is any error downloading the file from the url
	 * @throws ExecutionException 
	 */
	public void download() throws IOException, ExecutionException  {
		download(Thread.currentThread());
	}
	
	/**
	 * Download a file with the possibility of interrupting the download if the parentThread is
	 * interrupted
	 * 
	 * @param parentThread
	 * 	thread from where the download was launched, it is the reference used to stop the download
	 * @throws IOException if there is any error downloading the file from the url
	 */
	public void call(ReadableByteChannel rbc, FileOutputStream fos) throws IOException {
		sizeDownloaded.set(already);
        while (true) {
            long transferred = fos.getChannel().transferFrom(rbc, sizeDownloaded.get(), CHUNK_SIZE);
            if (transferred == 0) {
                break;
            }

            sizeDownloaded.set(sizeDownloaded.get() + transferred);
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
        }
    }
	
	private void checkDownloadContinues(Thread parentThread, Thread downloadThread) throws IOException {
		long lastBytesDownloaded = 0;
        long lastCheckedTime = System.currentTimeMillis();
        this.getSizeDownloaded();
        while (parentThread.isAlive() && downloadThread.isAlive()) {
            try {Thread.sleep(1000);} catch (InterruptedException e) {return;}

        	long totalBytesDownloaded = file.length();
            if (System.currentTimeMillis() - lastCheckedTime > STALL_THRES) {
                if (lastBytesDownloaded == totalBytesDownloaded) {
                	downloadThread.interrupt();
                    System.err.println("Connection lost. Time number: " + (lost_conn + 1));
                    already = file.length();
                	lost_conn ++;
                    return;
                }
                lastBytesDownloaded = totalBytesDownloaded;
            }
            if (printProgress)
            	System.out.println(getStringToPrintProgress(name, lastBytesDownloaded / (double) this.fileSize));
            lastCheckedTime = System.currentTimeMillis();
        }
        complete = true;
	}

	/**
	 * Get the size of the file stored in the given URL
	 * @param url
	 * 	url where the file is stored
	 * @return the size of the file
	 */
	public static long getFileSize(URL url) {
		HttpsURLConnection conn = null;
		try {
			SSLContext sslContext = getAllTrustingSSLContext();
			conn = ( HttpsURLConnection ) url.openConnection();
			conn.setSSLSocketFactory( sslContext.getSocketFactory() );
			conn.setRequestProperty("User-Agent", CommonUtils.getJDLLUserAgent());
			if (conn.getResponseCode() >= 300 && conn.getResponseCode() <= 308)
				return getFileSize(redirectedURL(url));
			if (conn.getResponseCode() != 200)
				throw new Exception( "Unable to connect to: " + url );
			long size = conn.getContentLengthLong();
			conn.disconnect();
			return size;
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (Exception ex) {
			ex.printStackTrace();
			String msg = "Unable to connect to " + url.toString();
			System.out.println(msg);
			return 1;
		}
	}

	private static SSLContext getAllTrustingSSLContext() throws NoSuchAlgorithmException, KeyManagementException
	{
		// Create a trust manager that does not validate certificate chains
		TrustManager[] trustAllCerts = new TrustManager[] {
				new X509TrustManager()
				{
					public java.security.cert.X509Certificate[] getAcceptedIssuers()
					{
						return new X509Certificate[ 0 ];
					}

					public void checkClientTrusted( X509Certificate[] certs, String authType )
					{
						// Do nothing, since we trust all certificates here
					}

					public void checkServerTrusted( X509Certificate[] certs, String authType )
					{
						// Do nothing, since we trust all certificates here
					}
				}
		};

		// Create a SSLContext with an all-trusting trust manager
		SSLContext sslContext = SSLContext.getInstance( "SSL" );
		sslContext.init( null, trustAllCerts, new java.security.SecureRandom() );
		return sslContext;
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
	}
	
	/**
	 * This method should be used when we get the following response codes from 
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
	 * @return the redirected url
	 * @throws MalformedURLException if the url is invalid
	 * @throws URISyntaxException if the url is invalid
	 */
	public static URL redirectedURL(URL url) throws MalformedURLException, URISyntaxException {
		int statusCode;
		HttpURLConnection conn;
		try {
			conn = (HttpURLConnection) url.openConnection();
			conn.setRequestProperty("User-Agent", CommonUtils.getJDLLUserAgent());
			statusCode = conn.getResponseCode();
		} catch (IOException ex) {
			return url;
		}
		if (statusCode < 300 || statusCode > 308)
			return url;
		String newURL = conn.getHeaderField("Location");
		try {
			conn.disconnect();
			return redirectedURL(new URL(newURL));
		} catch (MalformedURLException ex) {
		}
		try {
			conn.disconnect();
			if (newURL.startsWith("//"))
				return redirectedURL(new URL("http:" + newURL));
			else
				throw new MalformedURLException();
		} catch (MalformedURLException ex) {
		}
        URI uri = url.toURI();
        String scheme = uri.getScheme();
        String host = uri.getHost();
        String mainDomain = scheme + "://" + host;
		conn.disconnect();
		return redirectedURL(new URL(mainDomain + newURL));
	}
	
	/**
	 * Gets the filename of the file in an URL from the url String
	 * @param str
	 * 	the URL string
	 * @return the file name of the file in the URL
	 * @throws MalformedURLException if the String does not correspond to an URL
	 */
	public static String getFileNameFromURLString(String str) throws MalformedURLException {
		if (str.startsWith(Constants.ZENODO_DOMAIN) && str.endsWith(Constants.ZENODO_ANNOYING_SUFFIX))
			str = str.substring(0, str.length() - Constants.ZENODO_ANNOYING_SUFFIX.length());
		URL url = new URL(str);
		return new File(url.getPath()).getName();
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
