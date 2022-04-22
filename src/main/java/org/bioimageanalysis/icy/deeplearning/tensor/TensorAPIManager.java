package org.bioimageanalysis.icy.deeplearning.tensor;

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
	
	
	
	public static void convertListOfTensors

}
