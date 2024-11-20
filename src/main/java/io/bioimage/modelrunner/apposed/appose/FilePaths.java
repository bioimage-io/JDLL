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

package io.bioimage.modelrunner.apposed.appose;

import java.io.File;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.List;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
/**
 * Utility methods for working with file paths.
 */
public final class FilePaths {

	private FilePaths() {
		// Prevent instantiation of utility class.
	}

	/**
	 * Gets the path to the JAR file containing the given class. Technically
	 * speaking, it might not actually be a JAR file, it might be a raw class
	 * file, or even something weirder... But for our purposes, we'll just
	 * assume it's going to be something you can put onto a classpath.
	 *
	 * @param c The class whose file path should be discerned.
	 * @return File path of the JAR file containing the given class.
	 */
	public static File location(Class<?> c) {
		try {
			return new File(c.getProtectionDomain().getCodeSource().getLocation().toURI());
		}
		catch (URISyntaxException exc) {
			return null;
		}
	}

	public static File findExe(List<String> dirs, List<String> exes) {
		for (String exe : exes) {
			File exeFile = new File(exe);
			if (exeFile.isAbsolute()) {
				// Candidate is an absolute path; check it directly.
				if (exeFile.canExecute()) return exeFile;
			}
			else {
				// Candidate is a relative path; check beneath each given directory.
				for (String dir : dirs) {
					File f = Paths.get(dir, exe).toFile();
					if (f.canExecute()) return f;
				}
			}
		}
		return null;
	}
}
