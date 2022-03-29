package org.bioimageanalysis.icy.deeplearning;

import java.util.List;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Tensors created to interact with a Deep Learning engine while
 * being agnostic to it. This class just contains the information to create
 * a tensor while maintaining flexibility to interact with any wanted
 * Deep Learning framework.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class Tensor
{
	/**
	 * Name given to the tensor in the model.
	 */
	private String tensorName;
	/**
	 * Axes order in int array form.
	 */
	private int[] axesArray;
	/**
	 * Axes order in String form.
	 */
	private String axesString;
	/** 
	 * Software agnostic representation of the tensor data
	 */
	private NDArray data;
	/**
	 * Whether the tensor represents an image or not
	 */
	private boolean isImage = false;
	/**
	 * TODO make more complex and complete
	 * The data type of the tensor
	 */
	private String dType = "float32";
	/**
	 * TODO make a more robust case for shape
	 * Shape of the tensor
	 */
	private int[] shape;
	
	/**
	 * Create the tensor object. 
	 * 
	 * @param tensorName
	 * 	name of the tensor as defined by the model
	 * @param axes
	 * 	String containing the axes order of the tensor. For example: "bcyx"
	 * @param data
	 * 	data structure similar to a Numpy array that contains all tensor numbers
	 */
    private Tensor(String tensorName, String axes, NDArray data)
    {
    	this.tensorName = tensorName;
    	this.axesString = axes;
    	this.axesArray = convertToTensorDimOrder(axes);
    	this.data = data;
    }
    
    /**
     * Return a tensor object
	 * 
	 * @param tensorName
	 * 	name of the tensor as defined by the model
	 * @param axes
	 * 	String containing the axes order of the tensor. For example: "bcyx"
	 * @param data
	 * 	data structure similar to a Numpy array that contains all tensor numbers
     * @return the tensor
     */
    public static Tensor build(String tensorName, String axes, NDArray data)
    {
    	return new Tensor(tensorName, axes, data);
    }
    
    /**
     * Creates a tensor without data. The idea is to fill the data later.
	 * 
	 * @param tensorName
	 * 	name of the tensor as defined by the model
	 * @param axes
	 * 	String containing the axes order of the tensor. For example: "bcyx"
     * @return the tensor
     */
    public static Tensor buildEmptyTensor(String tensorName, String axes)
    {
    	return new Tensor(tensorName, axes, null);
    }

    /**
     * Convert the String representation of the axes order into an int array 
     * representation, easier to handle by the program
     * 
     * @param dimOrder
     * 	String representation of the axes
     * @return the int[] representation of the axes
     * @throws IllegalArgumentException if the String representation contains
     * repeated axes
     */
    public static int[] convertToTensorDimOrder(String dimOrder) throws IllegalArgumentException
    {
    	dimOrder = dimOrder.toLowerCase();
        int[] tensorDimOrder = new int[dimOrder.length()];
        int hasB = 0, hasI = 0, hasT = 0, hasX = 0, hasY = 0, hasZ = 0, hasC = 0;
        for (int i = 0; i < dimOrder.length(); i++)
        {
            switch (dimOrder.charAt(i))
            {
                case 'b':
                    tensorDimOrder[i] = 4;
                    hasB = 1;
                    break;
                case 'i':
                    tensorDimOrder[i] = 4;
                    hasI = 1;
                    break;
                case 't':
                    tensorDimOrder[i] = 4;
                    hasT = 1;
                    break;
                case 'z':
                    tensorDimOrder[i] = 3;
                    hasZ += 1;
                    break;
                case 'c':
                    tensorDimOrder[i] = 2;
                    hasC += 1;
                    break;
                case 'y':
                    tensorDimOrder[i] = 1;
                    hasY += 1;
                    break;
                case 'x':
                    tensorDimOrder[i] = 0;
                    hasX += 1;
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Illegal axis for tensor dim order " + dimOrder + " (" + dimOrder.charAt(i)
                                    + ")");
            }
        }
        if (hasB + hasI + hasT > 1)
            throw new IllegalArgumentException("Has at least two of b, i or t at the same time.");
        else if (hasY > 1 || hasX > 1 || hasC > 1 || hasZ > 1)
            throw new IllegalArgumentException("There cannot be repeated dimensions in the axes "
            		+ "order as it is specified for this tensor (" + dimOrder + ").");
        return tensorDimOrder;
    }
    
    /**
     * Get the name of the tensor
     * @return the name of the tensor
     */
    public String getName() {
    	return this.tensorName;
    }
    
    /**
     * Return the array containing the int representation of the axes order
     * @return the axes order in int[] representation
     */
    public int[] getAxesOrder() {
    	return this.axesArray;
    }
    
    /**
     * Set whether the tensor represents an image or not
     * @param isImage
     * 	if the tensor is an image or not
     */
    public void setIsImage(boolean isImage) {
    	this.isImage = isImage;
    }
    
    /**
     * Whether the tensor represents an image or not
     * @return true if the tensor represents an image, false otherwise
     */
    public boolean isImage() {
    	return isImage;
    }
    
    /**
     * Set the data type of the tensor
     * @param dtype
     * 	the data type of the tensor
     */
    public void dType(String dType) {
    	this.dType = dType;
    }
    
    /**
     * GEt the data type of the tensor
     * @return the data type of the tensor
     */
    public String getDType() {
    	return dType;
    }
    
    /**
     * Set the data structure of the tensor that contains the numbers
     * @param data
     * 	the numbers of the tensor in a Numpy array like structure
     */
    public void setData(NDArray data) {
    	this.data = data;
    }
    
    /**
     * REturn the data in a software agnostic way using DJL NDArrays
     * @return the data of the tensor as a buffer
     */
    public NDArray getData() {
    	return this.data;
    }
    
    /**
     * Set the shape of the tensor
     * @param shape
     * 	shape of the tensor
     */
    public void setShape(int[] shape) {
    	this.shape = shape;
    }
    
    /**
     * REturns the shape of the tensor
     * @return the shape of the tensor
     */
    public int[] getShape() {
    	return shape;
    }
    
    /**
     * REtrieve the axes order in String form
	 * @return the axesString
	 */
	public String getAxesString() {
		return axesString;
	}

	/**
     * Empty the tensor information
     */
    public void close() {
	   	tensorName = null;
	   	axesArray = null;
	   	data.close();
    }
    
    /**
     * Retrieve tensor with the wanted name from a list of tensors
     * @param lTensors
     * 	list of tensors
     * @param name
     * 	name of the tensor of interest
     * @return the tensor of interest
     */
    public static Tensor getTensorByNameFromList(List<Tensor> lTensors, String name) {
    	return lTensors.stream().filter(pp -> pp.getName().equals(name)).findAny().orElse(null);
    }
    
    public static void main(String[] args) {
    	NDManager a = NDManager.newBaseManager();
    	NDManager b = NDManager.newBaseManager();
    	NDArray tensora = a.create(false);
    	tensora.detach();
    	String nameb = b.getName();
    	tensora.attach(b);
    }
}
