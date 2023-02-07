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
