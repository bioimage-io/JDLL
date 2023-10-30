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
package io.bioimage.modelrunner.engine.installation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;

public class FileDownloader {
	private ReadableByteChannel rbc;
	private FileOutputStream fos;
	private static final long CHUNK_SIZE = 1024 * 1024 * 5;
	
	public FileDownloader(ReadableByteChannel rbc, FileOutputStream fos) {
		this.rbc = rbc;
		this.fos = fos;
	}
	
	/**
	public void call() throws IOException  {
		fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
	}
	*/
	public void call(Thread parentThread) throws IOException {
        long position = 0;
        while (true) {
            long transferred = fos.getChannel().transferFrom(rbc, position, CHUNK_SIZE);
            if (transferred == 0) {
                break;
            }

            position += transferred;

            if (parentThread.isInterrupted()) {
                // Close resources if needed and exit
                closeResources();
                throw new IOException("File download was interrupted");
            }
        }
    }

    private void closeResources() {
        try {
            if (rbc != null) rbc.close();
            if (fos != null) fos.close();
        } catch (IOException e) {
            // Handle exception during close
        }
    }
}
