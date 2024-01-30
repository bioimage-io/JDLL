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
