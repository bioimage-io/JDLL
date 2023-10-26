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
package io.bioimage.modelrunner.bioimageio.description.exceptions;

/**
 * A exception to be launched when there is any error reading the Bioimage.io rdf.yaml file
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ModelSpecsException extends Exception
{
	private static final long serialVersionUID = 1L;
	
	/**
	 * Constructor for every exception related to reading the Bioimage.io rdf.yaml specs file
	 * @param message
	 * 	the message that wants to be passed as the exception info.
	 */
	public ModelSpecsException(String message) {
        super(message);
    }
}
