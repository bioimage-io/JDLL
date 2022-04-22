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
	 * Method that changes the backend of all the tensors to buffer, so they can be transerred from
	 * one API to another one
	 * @param inputs
	 * 	list of tensor inputs
	 */
	public static void prepareInputTensors(List<Tensor> inputs) {
		for (Tensor tt : inputs) {
			try {
				tt.array2buffer();
			} catch (IllegalArgumentException ex) {
				tt.getDataAsBuffer();
			}
		}
	}

}
