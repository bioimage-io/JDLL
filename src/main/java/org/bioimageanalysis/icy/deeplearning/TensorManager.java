package org.bioimageanalysis.icy.deeplearning;

import java.nio.FloatBuffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Class created to mimic the Deep Java Library NDManager. that is needed 
 * to create NDArrays, which are the backend of Icy Tensors.
 * Icy Tensors need to be created using this class. Only one manager can be created
 * per session.
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class TensorManager implements AutoCloseable{
	/**
	 * NDManager used to create and manage NDArrays
	 */
	private NDManager manager;
	/**
	 * Unique identifier of the manager
	 */
	private String identifier;
	
	/**
	 * Manager that is needed to create Icy Tensors
	 */
	private TensorManager() {
		manager = NDManager.newBaseManager();
		identifier = manager.getName();
	}

	/**
	 * Manager that is needed to create Icy Tensors
	 * @param manager
	 * 	the NDManager used to create NDArrays
	 */
	private TensorManager(NDManager manager) {
		this.manager = manager;
		identifier = manager.getName();
	}
	
	/**
	 * Build the {@link #TensorManager()}
	 * @return an instance of TensorManager
	 */
	public static TensorManager build() {
		return new TensorManager();
	}
	
	/**
	 * Creates the actual Icy tensor from an NDArray
	 * @param tensorName
	 * 	the name of the tensor
	 * @param axes
	 * 	the axes order of the tensor in String mode
	 * @param data
	 * 	the actual number of the tensor in an array form
	 * @return
	 */
	public Tensor createTensor(String tensorName, String axes, NDArray data) {
		manager = data.getManager();
		return Tensor.build(tensorName, axes, data, this);
	}
	
	public Tensor createTensor(String tensorName, String originalAxes, String targetAxes, FloatBuffer dataBuff) {
		return Tensor.build(tensorName, originalAxes, data, this);
	}
	
	public Tensor createTensor(String tensorName, String originalAxes, String targetAxes, DoubleBuffer dataBuff) {
		return Tensor.build(tensorName, originalAxes, data, this);
	}
	
	public Tensor createTensor(String tensorName, String originalAxes, String targetAxes, IntBuffer dataBuff) {
		return Tensor.build(tensorName, originalAxes, data, this);
	}
	
	/**
	 * Retrieves the NDManager used to create tensors
	 * @return the NDManager
	 */
	public NDManager getManager() {
		return manager;
	}
	
	/**
	 * Retrieves identifier of the TEnsorMAnager
	 * @return the name or identifier of the instance
	 */
	public String getIdentifier() {
		return identifier;
	}

	@Override
	/**
	 * Close all the tensors associated to the manager and the manager
	 */
	public void close() throws Exception {
		manager.close();		
	}
}
