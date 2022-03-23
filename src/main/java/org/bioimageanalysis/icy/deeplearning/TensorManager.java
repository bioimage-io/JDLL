package org.bioimageanalysis.icy.deeplearning;

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
	
	private TensorManager() {
		manager = NDManager.newBaseManager();
	}
	
	public static TensorManager build() {
		return new TensorManager();
	}
	
	public Tensor createTensor(String tensorName, String axes, NDArray data) {
		return Tensor.build(tensorName, axes, data, this);
	}
	
	public NDManager getManager() {
		return manager;
	}

	@Override
	public void close() throws Exception {
		manager.close();		
	}
}
