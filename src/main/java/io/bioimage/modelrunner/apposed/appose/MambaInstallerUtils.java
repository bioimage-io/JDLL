/*-
 * #%L
 * Appose: multi-language interprocess cooperation with shared memory.
 * %%
 * Copyright (C) 2023 Appose developers.
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package io.bioimage.modelrunner.apposed.appose;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;

import org.apache.commons.compress.archivers.ArchiveException;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.compress.utils.IOUtils;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
/**
 * Utility methods  unzip bzip2 files
 */
public final class MambaInstallerUtils {
	
	private MambaInstallerUtils() {
		// Prevent instantiation of utility class.
	}
	
	/**
	 * DEcompress a bzip2 file into a new file.
	 * The method is needed because Micromamba is distributed as a .tr.bz2 file and
	 * many distributions do not have tools readily available to extract the required files 
	 * @param source
	 * 	.bzip2 file
	 * @param destination
	 * 	destination folder where the contents of the file are going to be decompressed
	 * @throws FileNotFoundException if the .bzip2 file is not found or does not exist
	 * @throws IOException if the source file already exists or there is any error with the decompression
	 */
	public static void unBZip2(File source, File destination) throws FileNotFoundException, IOException {
	    try (
	    		BZip2CompressorInputStream input = new BZip2CompressorInputStream(new BufferedInputStream(new FileInputStream(source)));
	    		FileOutputStream output = new FileOutputStream(destination);
	    		) {
	        IOUtils.copy(input, output);
	    }
	}
	
	/** Untar an input file into an output file.

	 * The output file is created in the output folder, having the same name
	 * as the input file, minus the '.tar' extension. 
	 * 
	 * @param inputFile     the input .tar file
	 * @param outputDir     the output directory file. 
	 * @throws IOException 
	 * @throws FileNotFoundException
	 * @throws ArchiveException 
	 */
	public static void unTar(final File inputFile, final File outputDir) throws FileNotFoundException, IOException, ArchiveException {

		try (
				InputStream is = new FileInputStream(inputFile);
				TarArchiveInputStream debInputStream = (TarArchiveInputStream) new ArchiveStreamFactory().createArchiveInputStream("tar", is);
				) {
		    TarArchiveEntry entry = null; 
		    while ((entry = (TarArchiveEntry)debInputStream.getNextEntry()) != null) {
		        final File outputFile = new File(outputDir, entry.getName());
		        if (entry.isDirectory()) {
		            if (!outputFile.exists()) {
		                if (!outputFile.mkdirs()) {
		                    throw new IllegalStateException(String.format("Couldn't create directory %s.", outputFile.getAbsolutePath()));
		                }
		            }
		        } else {
		        	if (!outputFile.getParentFile().exists()) {
		        	    if (!outputFile.getParentFile().mkdirs()) 
		        	        throw new IOException("Failed to create directory " + outputFile.getParentFile().getAbsolutePath());
		        	}
		            try (OutputStream outputFileStream = new FileOutputStream(outputFile)) {
		            	IOUtils.copy(debInputStream, outputFileStream);
		            }
		        }
		    }
		} 

	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException, ArchiveException, URISyntaxException {
		String url = Conda.MICROMAMBA_URL;
		final File tempFile = File.createTempFile( "miniconda", ".tar.bz2" );
		tempFile.deleteOnExit();
		URL website = MambaInstallerUtils.redirectedURL(new URL(url));
		ReadableByteChannel rbc = Channels.newChannel(website.openStream());
		try (FileOutputStream fos = new FileOutputStream(tempFile)) {
			long transferred = fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
			System.out.print(tempFile.length());
		}
		String tarPath = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\micromamba-1.5.1-1.tar";
		String mambaPath = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\mamba";
		unBZip2(new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\micromamba-1.5.1-1.tar.bz2"), 
			new File(tarPath));
		unTar(new File(tarPath), new File(mambaPath));
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
	 * @throws MalformedURLException 
	 * @throws URISyntaxException 
	 */
	public static URL redirectedURL(URL url) throws MalformedURLException, URISyntaxException {
		int statusCode;
		HttpURLConnection conn;
		try {
			conn = (HttpURLConnection) url.openConnection();
			statusCode = conn.getResponseCode();
		} catch (IOException ex) {
			return url;
		}
		if (statusCode < 300 || statusCode > 308)
			return url;
		String newURL = conn.getHeaderField("Location");
		try {
			return redirectedURL(new URL(newURL));
		} catch (MalformedURLException ex) {
		}
		try {
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
		return redirectedURL(new URL(mainDomain + newURL));
	}
}
