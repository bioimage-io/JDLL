package org.bioimageanalysis.icy.deeplearning.tensor;


import java.util.ArrayList;
import java.util.List;

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
public class TensorManager implements AutoCloseable {
	/**
	 * NDManager used to create and manage NDArrays
	 */
	private NDManager manager;
	/**
	 * Unique identifier of the manager
	 */
	private String identifier;
	/**
	 * List of tensors associated with this TensorManager
	 */
	private List<Tensor> tensors = new ArrayList<Tensor>();

	/**
	 * Manager that is needed to create Icy Tensors
	 * @param startManager
	 * 	whether the manager should be initialized or not
	 */
	private TensorManager(boolean startManager) {
		if (startManager) {
			manager = NDManager.newBaseManager();
			identifier = manager.getName();
		}
	}
	/**
	 * 
	 * Manager that is needed to create Icy Tensors
	 * @param manager
	 * 	the {@link NDManager} that acts as the backend of the tensor manager
	 */
	private TensorManager(NDManager manager) {
		this.manager = manager;
		identifier = manager.getName();
	}
	
	/**
	 * Build the {@link #TensorManager()}, by default it crates an {@link NDManager}
	 * @return an instance of TensorManager
	 */
	public static TensorManager build() {
		return new TensorManager(true);
	}
	
	/**
	 * Build the {@link #TensorManager()}
	 * @param manager
	 * 	manager that will be in the backend of the tensor manager
	 * @return an instance of TensorManager
	 */
	public static TensorManager build(NDManager manager) {
		return new TensorManager(manager);
	}
	
	/**
	 * Build the {@link #TensorManager()}
	 * @param startManager
	 * 	whether the manager should be initialized or not
	 * @return an instance of TensorManager
	 */
	public static TensorManager build(boolean startManager) {
		return new TensorManager(startManager);
	}
	
	/**
	 * Creates the actual Icy tensor from an NDArray
	 * @param tensorName
	 * 	the name of the tensor
	 * @param axes
	 * 	the axes order of the tensor in String mode
	 * @param data
	 * 	the actual number of the tensor in an array form
	 * @return an Icy Tensor
	 * @throws IllegalArgumentException if the NDArray provided comes from a different NDManager
	 */
	public Tensor createTensor(String tensorName, String axes, NDArray data) throws IllegalArgumentException {
		if (manager == null && data != null) {
			manager = data.getManager();
			identifier = manager.getName();
		} else if (!manager.getName().equals(identifier)) {
			throw new IllegalArgumentException("All the NDArrays associated to the same TensorManager"
					+ " need to have been created with the same NDManager. In addition to this, "
					+ "running models with DJL Pytorch will only work if all the NDArrays come"
					+ " from the same NDManager.");
		}
		return Tensor.build(tensorName, axes, data, this);
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
	
	/**
	 * Get all the tensors created by the TensorManager
	 * @return list of tensors associated to the TensorManager
	 */
	public List<Tensor> getTensorList(){
		return this.tensors;
	}
	
	/**
	 * Add tensor to list of tensors owned by this tensor manager. REgard that the parent
	 * TensorManager of the added tensor must be the same as the tensorManager instance.
	 * @param tt
	 * 	tensor to be added
	 */
	public void addTensorToList(Tensor tt) {
		if (tt.getManager() != this) {
			throw new IllegalArgumentException("The input tensor parent Tensormanager must "
					+ "be the same as the TensorManager owning the list of tensors.");
		}
		tensors.add(tt);
	}

	@Override
	/**
	 * Close all the tensors associated to the manager and the manager
	 */
	public void close() throws Exception {
		manager.close();		
	}
}
