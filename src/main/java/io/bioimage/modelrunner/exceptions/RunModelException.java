/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
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
