/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
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
package io.bioimage.modelrunner.tensor.shm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;

public interface MacosHelpers extends Library {
	
	static final String LIBRARY_NAME = "libShmCreate";
	
	static final String LIBRARY_NAME_SUF = ".dylib";

    // Load the native library
	MacosHelpers INSTANCE = loadLibrary();

    static MacosHelpers loadLibrary() {
    	InputStream in = MacosHelpers.class.getClassLoader().getResourceAsStream(LIBRARY_NAME + LIBRARY_NAME_SUF);
        /*if (in == null) {
            throw new FileNotFoundException("Library file " + LIBRARY_NAME + " not found in JAR.");
        }*/

        File tempFile = null;
		try {
			tempFile = File.createTempFile(LIBRARY_NAME, LIBRARY_NAME_SUF);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
        tempFile.deleteOnExit();

        try (OutputStream out = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[1024];
            int readBytes;
            while ((readBytes = in.read(buffer)) != -1) {
                out.write(buffer, 0, readBytes);
            }
        } catch (IOException e) {
			e.printStackTrace();
		}

        return (MacosHelpers) Native.load(tempFile.getAbsolutePath(), MacosHelpers.class);
    }

    // Declare methods corresponding to the native functions
    int create_shared_memory(String name, int size);
    
    void unlink_shared_memory(String name);
}


