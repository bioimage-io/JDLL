/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
import java.io.IOException;
import java.net.URISyntaxException;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
public class Builder {
	

	/**
	 * Executes build.
	 *
	 * @return the resulting value.
	 */
	public Environment build() {
		String base = baseDir.getPath();
		boolean useSystemPath = systemPath;

		// TODO Build the thing!~
		// Hash the state to make a base directory name.
		// - Construct conda environment from condaEnvironmentYaml.
		// - Download and unpack JVM of the given vendor+version.
		// - Populate ${baseDirectory}/jars with Maven artifacts?

		try {
			Mamba conda = new Mamba(Mamba.BASE_PATH);
			conda.installMicromamba();
			String envName = "appose";
			if (conda.getEnvironmentNames().contains( envName )) {
				// TODO: Should we update it? For now, we just use it.
			}
			else {
				conda.createWithYaml(envName, condaEnvironmentYaml.getAbsolutePath());
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		} catch (URISyntaxException e) {
			throw new RuntimeException(e);
		} catch (MambaInstallException e) {
			throw new RuntimeException(e);
		}

		return new Environment() {
			@Override public String base() { return base; }
			@Override public boolean useSystemPath() { return useSystemPath; }
		};
	}

	// -- Configuration --

	private boolean systemPath;

	/**
	 * Executes use system path.
	 *
	 * @return the resulting value.
	 */
	public Builder useSystemPath() {
		systemPath = true;
		return this;
	}

	private File baseDir;

	/**
	 * Executes base.
	 *
	 * @param directory the directory parameter.
	 * @return the resulting value.
	 */
	public Builder base(File directory) {
		baseDir = directory;
		return this;
	}

	// -- Conda --

	private File condaEnvironmentYaml;

	/**
	 * Executes conda.
	 *
	 * @param environmentYaml the environmentYaml parameter.
	 * @return the resulting value.
	 */
	public Builder conda(File environmentYaml) {
		this.condaEnvironmentYaml = environmentYaml;
		return this;
	}

	// -- Java --

	private String javaVendor;
	private String javaVersion;

	/**
	 * Executes java.
	 *
	 * @param vendor the vendor parameter.
	 * @param version the version parameter.
	 * @return the resulting value.
	 */
	public Builder java(String vendor, String version) {
		this.javaVendor = vendor;
		this.javaVersion = version;
		return this;
	}
}
