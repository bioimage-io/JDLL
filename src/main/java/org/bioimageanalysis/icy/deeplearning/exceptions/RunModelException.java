package org.bioimageanalysis.icy.deeplearning.exceptions;

/**
 * Exception thrown when there have been problems running a Deep Learning model
 * @author Carlos Garcia Lopez de Haro
 * 
 */
public class RunModelException extends Exception {
	
		private static final long serialVersionUID = 1L;
		
		/**
		 * Constructor that transports exceptions that happened in the Deep Learning
		 * engine interface into the Deep LEarning manager inside which is being run
		 * inside the main program
		 * @param msg
		 * 	the message of the original exception
		 */
		public RunModelException(String msg) {
			super(msg);
		}
		
		/** Exception when the number of tensors expected is
		 * not the same as the number of tensors outputted by the
		 * model
		 * @param nOutputTensors
		 * 	number of tensors outputted by the model
		 * @param nExpectedTensors
		 * 	number of tensors expected
		 */
		public RunModelException(int nOutputTensors, int nExpectedTensors) {
			super("The Deep Learning model outputted " + nOutputTensors + 
					" tensors but the specifications of the model indicated that"
					+ "there were only " + nExpectedTensors + " output tensors.");
		}

}
