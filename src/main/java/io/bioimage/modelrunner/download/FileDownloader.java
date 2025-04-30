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
import java.net.SocketTimeoutException;
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

/**
 * Class that implements the download of a file and the tracking of the download
 * @author Carlos Garcia
 */
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
	
	private static final long CHUNK_SIZE = 1024 * 1024 * 1;

	private static final int STALL_THRES = 10000;

	private static final int PROGRESS_INTER = 1000;
	
	private static final int MAX_RETRIES = 3;
	
	/**
	 * Create an instance of an object that can download the file at the wanted url in the wanted file
	 * 
	 * @param url
	 * 	the url where the file of interest is
	 * @param file
	 * 	file into which the contents of the url are going to be copied
	 * @param printProgress
	 * 	whether to print to the terminal the progress of the download or not
	 * @throws MalformedURLException if the provided url is not valid
	 */
	public FileDownloader(String url, File file, boolean printProgress) throws MalformedURLException {
		this.website = new URL(url);
		this.file = file;
		this.printProgress = printProgress;
		this.name = getFileNameFromURLString(url);
	}

	/**
	 * Create an instance of an object that can download the file at the wanted url in the wanted file.
	 * By default the progress of the file download is printed to the terminal.
	 * 
	 * @param url
	 * 	the url where the file of interest is
	 * @param file
	 * 	file into which the contents of the url are going to be copied
	 * @throws MalformedURLException if the provided url is not valid
	 */
	public FileDownloader(String url, File file) throws MalformedURLException {
		this(url, file, true);
	}
	
	/**
	 * Set a consumer that will receive the number of bytes that have been downloaded
	 * @param totalProgress
	 * 	consumer that will receive the number of bytes that have been downloaded
	 */
	public void setTotalProgressConsumer(Consumer<Long> totalProgress) {
		this.totalProgress = totalProgress;
	}
	
	/**
	 * Set a consumer that will receive the percentage of the bytes out of the total that have been downloaded
	 * @param partialProgress
	 * 	consumer that will receive the percentage of the bytes out of the total that have been downloaded
	 */
	public void setPartialProgressConsumer(Consumer<Double> partialProgress) {
		this.partialProgress = partialProgress;
	}
	
	/**
	 * 
	 * @return the name of the file at the url
	 */
	public String getFileName() {
		return this.name;
	}
	
	/**
	 * 
	 * @return the size in bytes of the object that wants to be downloaded.
	 */
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
	
	/**
	 * 
	 * @return the amount of bytes that have been downloaded at the moment
	 */
	public long getSizeDownloaded() {
		return this.sizeDownloaded.get();
	}
	
	/**
	 * Download the file of interest.
	 * 
	 * 
	 * @param parentThread
	 * 	if not null, whenever this thread is interrupted, the download will stop
	 * @throws IOException if there is any file related error in the download
	 * @throws ExecutionException if there is any error with the urls or the conection
	 */
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
			performDownload(fos, rbc, parentThread);
		} catch (SocketTimeoutException ex) {
			System.err.println("Socket timeout accessing: " + website);
			lost_conn ++;
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
	 * Download the file of interest.
	 * The download will stop when the thread that executed this method is interrupted
	 * or when the download is finished or when it fails
	 * 
	 * @throws IOException if there is any file related error in the download
	 * @throws ExecutionException if there is any error with the urls or the conection
	 */
	public void download() throws IOException, ExecutionException  {
		download(Thread.currentThread());
	}
	
	private void call(ReadableByteChannel rbc, FileOutputStream fos) throws IOException {
		sizeDownloaded.set(already);
        while (true) {
            long transferred = fos.getChannel().transferFrom(rbc, sizeDownloaded.get(), CHUNK_SIZE);
            if (transferred == 0) {
                break;
            }

            sizeDownloaded.set(sizeDownloaded.get() + transferred);
            System.out.println(this.name + ": " + sizeDownloaded);
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
        }
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
