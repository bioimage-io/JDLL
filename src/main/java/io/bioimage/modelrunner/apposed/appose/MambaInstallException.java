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

/**
 * Exception to be thrown when Micromamba is not found in the wanted directory
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 */
public class MambaInstallException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new exception with the default detail message
     */
	public MambaInstallException() {
        super("Micromamba installation not found in the provided directory.");
    }

	/**
	 * Constructs a new exception with the specified detail message
	 * @param message
	 *  the detail message.
	 */
    public MambaInstallException(String message) {
        super(message);
    }

}
