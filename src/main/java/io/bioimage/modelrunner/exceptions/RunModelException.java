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
package io.bioimage.modelrunner.exceptions;

/**
 * Exception thrown when there have been problems running a Deep Learning model
 * 
 * @author Carlos Garcia Lopez de Haro
 * 
 */
public class RunModelException extends Exception
{

	private static final long serialVersionUID = 1L;

	/**
	 * Constructor that transports exceptions that happened in the Deep Learning
	 * engine interface into the Deep LEarning manager inside which is being run
	 * inside the main program
	 * 
	 * @param msg
	 *            the message of the original exception
	 */
	public RunModelException( String msg )
	{
		super( msg );
	}

	/**
	 * Exception when the number of tensors expected is not the same as the
	 * number of tensors outputted by the model
	 * 
	 * @param nOutputTensors
	 *            number of tensors outputted by the model
	 * @param nExpectedTensors
	 *            number of tensors expected
	 */
	public RunModelException( int nOutputTensors, int nExpectedTensors )
	{
		super( "The Deep Learning model outputted " + nOutputTensors
				+ " tensors but the specifications of the model indicated that" + "there were only " + nExpectedTensors
				+ " output tensors." );
	}

}
