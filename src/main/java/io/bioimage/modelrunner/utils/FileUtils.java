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
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipInputStream;


/**
 * Class to unzip files and directories
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 */

public class FileUtils {
    
	/**
	 * Copy one folder into another folder creating the necessary subfolders
	 * 
	 * @param src
	 * 	folder to be copied
	 * @param dest
	 * 	destination folder
	 * @throws IOException if there is any error copying the contents of the folder
	 */
    public static void copyFolder(Path src, Path dest) throws IOException {
        // If the directory does not exist, create it
        if (!Files.exists(dest)) {
            Files.createDirectories(dest);
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(src)) {
            for (Path entry : stream) {
                Path newDest = dest.resolve(entry.getFileName());
                
                if (Files.isDirectory(entry)) {
                    copyFolder(entry, newDest);
                } else {
                    Files.copy(entry, newDest, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }
}
