package org.bioimageanalysis.icy.deeplearning.tensor;

import java.util.ArrayList;
import java.util.List;

import ai.djl.ndarray.NDArray;

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
				if (tt.isEmpty())
					continue;
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
				if (tt.isEmpty())
					continue;
				tt.buffer2array();;
			} catch (IllegalArgumentException ex) {
				tt.getDataAsNDArray();
			}
		}
	}
	
	/**
	 * Create a copy of the original list of tensors from the Deep Learning Manager DJL API
	 * into the DJL API of the DL engine that is going to be used
	 * @param ogTensors
	 * 	tensors from original DJL API version
	 * @return tensors in the new DJL API version
	 */
	public static List<Tensor> createTensorsCopyAPI(List<Tensor> ogTensors) {
		TensorManager manager = TensorManager.build();
		List<Tensor> newTensors = new ArrayList<Tensor>();
		for (Tensor tt : ogTensors) {
			if (tt.isEmpty()) {
				newTensors.add(Tensor.buildEmptyTensor(tt.getName(), tt.getAxesString(), manager));
				continue;
			}
			NDArray backendNDArr = manager.getManager().create(tt.getDataAsBuffer(), null);
			// Empty the input tensor from the Deep Learning MAnager API version
			tt.setBufferData(null);
			Tensor nTensor = manager.createTensor(tt.getName(), tt.getAxesString(), backendNDArr);
			newTensors.add(nTensor);
		}
		return newTensors;
	}
	
}
