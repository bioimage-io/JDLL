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
