/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.exceptions;

/**
 * Exception thrown when there have been problems loading a Deep Learning model
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class LoadModelException extends Exception
{

	private static final long serialVersionUID = 1L;

	/**
	 * The original exception that caused the error
	 */
	private String ex;

	/**
	 * Message given by the Deep Learning engine interface
	 */
	private static String defaultMsg = "Error loading a Deep Learning model.";

	public LoadModelException( String ex )
	{
		super( defaultMsg + System.lineSeparator() + ex );
		this.ex = defaultMsg + System.lineSeparator() + ex;
	}

	public LoadModelException( String msg, String ex )
	{
		super( msg + System.lineSeparator() + ex );
		this.ex = msg + System.lineSeparator() + ex;
	}
	
	public LoadModelException()
	{
		super( defaultMsg );
		this.ex = defaultMsg;
	}

	public String toString()
	{
		return this.ex;
	}

}
