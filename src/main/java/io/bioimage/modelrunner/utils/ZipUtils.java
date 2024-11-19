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
package io.bioimage.modelrunner.utils;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;


/**
 * Class to unzip files and directories
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 */

public class ZipUtils
{
	
	private static final int BUFFER_SIZE = 8192;
	
	/**
	 * Unzip a zip file into the wanted path. This method does not track progress.
	 * In order to track progress, please use a consumer 
	 * with {@link #unzipFolder(String, String, Consumer)}
	 * @param sourcePath
	 * 	path to the zip file
	 * @param targetPath
	 * 	path to the file where everything will be extracted
	 * @throws IOException if there is any error extracting the files
	 */
    public static void unzipFolder(String sourcePath, String targetPath) throws IOException {
    	// Create empty consumer to work with unzipping method method
    	Consumer<Double> progressConsumer = new Consumer<Double>() {
    		@Override
            public void accept(Double d) {
            }
        };
    	unzipFolder(sourcePath, targetPath, progressConsumer);
    }
	
	/**
	 * Unzip a zip file into the wanted path. TRack the progress being made using 
	 * a {@link Consumer}
	 * @param sourcePath
	 * 	path to the zip file
	 * @param targetPath
	 * 	path to the file where everything will be extracted
	 * @param consumer
	 * 	track the unzipping of the files
	 * @throws IOException if there is any error extracting the files
	 */
    public static void unzipFolder(String sourcePath, String targetPath, Consumer<Double> consumer) throws IOException {
 	    
    	FileInputStream fis = new FileInputStream(new File(sourcePath));
    	ZipInputStream zis = new ZipInputStream(fis);

        ZipEntry entry = zis.getNextEntry();

        // Calculate the total size
        long totalSize = Files.size(Paths.get(sourcePath));
        long extractedSize = 0;

        while (entry != null) {
            File file = new File(targetPath, entry.getName());
            // Check if entry is directory (if the entry name ends with '\' or '/'
            if (entry.isDirectory()) {
                file.mkdirs();
            } else {
                File parent = file.getParentFile();
                if (!parent.exists()) 
                    parent.mkdirs();

                try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file))) {
                    byte[] buffer = new byte[BUFFER_SIZE];
                    int location;
                    while ((location = zis.read(buffer)) != -1 && !Thread.interrupted()) {
                        bos.write(buffer, 0, location);
           	          	extractedSize += location;
           	          	consumer.accept(((double) extractedSize) / ((double) totalSize));
                    }
     	           bos.close();
                } catch (ZipException e) {
    		 	    zis.close();
    		 	    fis.close();
    				throw e;
    			}
            }
            entry = zis.getNextEntry();
        }
 	    zis.close();
 	    fis.close();
 	}
    
    /**
     * Calculate the uncompressed size of a .zip file
     * @param zipFile
     * 	the zip file of interest
     * @return the size in bytes if the zip file was to be uncompressed
     * @throws IOException if there is any error finding the size
     */
    public static long getUncompressedSize(File zipFile) throws IOException {
    	return getUncompressedSize(zipFile, Thread.currentThread());
    }
    
    /**
     * Calculate the uncompressed size of a .zip file
     * @param zipFile
     * 	the zip file of interest
     * @param parentThread
     * 	thread from where this operation is being launched
     * @return the size in bytes if the zip file was to be uncompressed
     * @throws IOException if there is any error finding the size
     */
    public static long getUncompressedSize(File zipFile, Thread parentThread) throws IOException {
        long totalSize = 0;

        try (ZipFile zip = new ZipFile(zipFile)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();

            while (entries.hasMoreElements() && parentThread.isAlive()) {
                ZipEntry entry = entries.nextElement();

                // Skip directories
                if (!entry.isDirectory()) {
                    long size = entry.getSize();

                    // If size is not known (-1), read the entry to determine its size
                    if (size == -1) {
                        size = calculateEntrySize(zip, entry, parentThread);
                    }

                    totalSize += size;
                }
            }
        }

        return totalSize;
    }

    private static long calculateEntrySize(ZipFile zip, ZipEntry entry, Thread parentThread) throws IOException {
        long size = 0;
        byte[] buffer = new byte[8192];

        try (InputStream is = zip.getInputStream(entry)) {
            int read;
            while ((read = is.read(buffer)) != -1 && parentThread.isAlive()) {
                size += read;
            }
        }

        return size;
    }
}
