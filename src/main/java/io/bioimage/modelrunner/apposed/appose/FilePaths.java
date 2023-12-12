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
