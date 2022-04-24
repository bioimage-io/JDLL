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
	 * @param ogInTensors
	 * 	tensors from original DJL API version 
	 * @return tensors in the new DJL API version
	 */
	public static List<Tensor> createTensorsCopyIntoAPI(List<Tensor> ogInTensors) {
		TensorManager manager = TensorManager.build();
		return createTensorsCopyIntoAPI(ogInTensors, manager);
	}
	
	/**
	 * Create a copy of the original list of tensors from the Deep Learning Manager DJL API
	 * into the DJL API of the DL engine that is going to be used
	 * @param ogInTensors
	 * 	tensors from original DJL API version 
	 * @param manager
	 * 	TensorManager used to create the new tensors
	 * @return tensors in the new DJL API version
	 */
	public static List<Tensor> createTensorsCopyIntoAPI(List<Tensor> ogInTensors, TensorManager manager) {
		List<Tensor> newTensors = new ArrayList<Tensor>();
		for (Tensor tt : ogInTensors) {
			if (tt.isEmpty()) {
				newTensors.add(Tensor.buildEmptyTensor(tt.getName(), tt.getAxesString(), manager));
				continue;
			}
			NDArray backendNDArr = manager.getManager().create(tt.getDataAsBuffer(), 
										Tensor.ndarrayShapeFromIntArr(tt.getShape()));
			// Empty the input tensor from the Deep Learning MAnager API version
			tt.setBufferData(null);
			Tensor nTensor = manager.createTensor(tt.getName(), tt.getAxesString(), backendNDArr);
			newTensors.add(nTensor);
		}
		return newTensors;
	}
	
	/** TODO remove?
	 * Copy the backend of the source list of tensors into the target tensors.
	 * Both list of tensors must contain tensors called with the same exact names.
	 * @param source
	 * 	source tensors created in some DJL API version
	 * @param target
	 * 	target tensors created in another DJL API version
	 */
	public static void copyTensorsIntoAPIAsNDArrays(List<Tensor> source, List<Tensor> target) {
		if (source.size() != target.size())
			throw new IllegalArgumentException("Source and target tensors list must contain the same "
					+ "number of tensors.");
		for (Tensor ss : source) {
			Tensor tt = Tensor.getTensorByNameFromList(target, ss.getName());
			if (tt == null) {
				throw new IllegalArgumentException("Source list contains a tensor called "
						+ "'" + ss.getName() + "'. The target list must contain a tensor with the "
								+ "same name.");
			}
			if (ss.isEmpty())
				continue;
			if (ss.getDataAsBuffer() == null)
				throw new IllegalArgumentException("The backend of every source tensor must be defined "
						+ "with a data Buffer.");
			tt.copyBufferTensorBackend(ss);
			tt.buffer2array();
		}
	}
	
	/**
	 * Copy the backend of the source list of tensors into the target tensors.
	 * Both list of tensors must contain tensors called with the same exact names.
	 * @param source
	 * 	source tensors created in some DJL API version
	 * @param target
	 * 	target tensors created in another DJL API version
	 */
	public static void copyTensorsIntoAPIAsBuffers(List<Tensor> source, List<Tensor> target) {
		if (source.size() != target.size())
			throw new IllegalArgumentException("Source and target tensors list must contain the same "
					+ "number of tensors.");
		for (Tensor ss : source) {
			Tensor tt = Tensor.getTensorByNameFromList(target, ss.getName());
			if (tt == null) {
				throw new IllegalArgumentException("Source list contains a tensor called "
						+ "'" + ss.getName() + "'. The target list must contain a tensor with the "
								+ "same name.");
			}
			if (ss.isEmpty())
				continue;
			tt.copyBufferTensorBackend(ss);
		}
	}
	
}
