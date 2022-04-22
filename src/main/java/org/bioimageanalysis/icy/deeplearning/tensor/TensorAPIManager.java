package org.bioimageanalysis.icy.deeplearning.tensor;

import java.util.List;

/**
 * Class used to convert tensors that are defined in a particular API version into the API
 * version needed to execute the model. 
 * This is necessary because for Deep Java Library, each API is only compatible with few 
 * engine versions, thus to avoid conflicts, the tensor should always use the NDArray
 * from the API version compatible with the engine version.
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class TensorAPIManager {
	
	
	/**
	 * Method that changes the backend of all the tensors to buffer, so they can be transferred from
	 * one API to another one
	 * @param inputs
	 * 	list of tensor inputs
	 */
	public static void tensorsAsBuffers(List<Tensor> inputs) {
		for (Tensor tt : inputs) {
			try {
				tt.array2buffer();
			} catch (IllegalArgumentException ex) {
				tt.getDataAsBuffer();
			}
		}
	}
	/**
	 * Method that changes the backend of all the tensors to NDArrays, so they can be managed
	 * easily by the user to do operations in a Numpy-like manner
	 * @param inputs
	 * 	list of tensor inputs
	 */
	public static void tensorsAsNDArrays(List<Tensor> inputs) {
		for (Tensor tt : inputs) {
			try {
				tt.buffer2array();;
			} catch (IllegalArgumentException ex) {
				tt.getDataAsNDArray();
			}
		}
	}
	
}
