package org.bioimageanalysis.icy.deeplearning.exceptions;

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
	private String msg;

	public LoadModelException( String ex, String msg )
	{
		super( msg + "\n" + ex );
		this.ex = ex;
		this.msg = msg;
	}

	public LoadModelException()
	{
		super( "Error loading a Deep Learning model." );
		this.ex = "";
		this.msg = "";
	}

	public String toString()
	{
		return this.msg + "\n" + this.ex;
	}

}
