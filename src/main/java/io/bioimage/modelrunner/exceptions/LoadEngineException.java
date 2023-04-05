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

import java.io.File;

/**
 * Exception thrown when there have been problems loading a Deep Learning model
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class LoadEngineException extends Exception
{

	private static final long serialVersionUID = 1L;

	/**
	 * Message given by the Deep Learning engine interface
	 */
	private static String msg = "Error loading a Deep Learning engine";

	private String nonStaticMsg;

	public LoadEngineException( String info )
	{
		super( msg + ".\n" + info );
		this.nonStaticMsg = msg + "\n" + info;
	}

	public LoadEngineException( File dir, String info )
	{
		super( msg + " located at " + dir.getName() + ".\n" + info );
		this.nonStaticMsg = msg + " located at " + dir.getName() + ".\n" + info;
	}

	public LoadEngineException( File dir )
	{
		super( msg + " located at " + dir.getName() );
		this.nonStaticMsg = msg + " located at " + dir.getName();
	}

	public LoadEngineException()
	{
		super( msg );
		this.nonStaticMsg = msg;
	}

	public String toString()
	{
		return this.nonStaticMsg;
	}

}
