package org.bioimageanalysis.icy.deeplearning.tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;




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
	private INDArray data;
	/**
	 * Whether the tensor represents an image or not
	 */
	private boolean isImage = true;
	/**
	 * Whether the tensor has been created without an NDarray or not. Once
	 * the NDarray is added, the tensor cannot be empty anymore
	 */
	private boolean emptyTensor;
	/**TODO develop a DAtaType class for this Tensor class?
	 * The data type of the tensor
	 */
	private DataType dType;
	/**
	 * Shape of the tensor
	 */
	private int[] shape;
	/**
	 * Whether the tensor is closed or not. If it is, nothing can be done on the Tensor
	 */
	private boolean closed = false;
	
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
    private Tensor(String tensorName, String axes, INDArray data)
    {
    	Objects.requireNonNull(tensorName, "'tensorName' field should not be empty");
    	Objects.requireNonNull(axes, "'axes' field should not be empty");
    	chechDims(data, axes);
    	this.tensorName = tensorName;
    	this.axesString = axes;
    	this.axesArray = convertToTensorDimOrder(axes);
    	this.data = data;
    	if (data != null) {
    		setShape();
        	dType = data.dataType();
    		emptyTensor = false;
    	} else {
    		emptyTensor = true;
    	}
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
    public static Tensor build(String tensorName, String axes, INDArray data)
    {
    	if (data == null)
    		throw new IllegalArgumentException("Trying to create tensor from an empty NDArray");
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
     * Set the data structure of the tensor that contains the numbers
     * @param data
     * 	the numbers of the tensor in a Numpy array like structure
     */
    public void setNDArrayData(INDArray data) {
    	throwExceptionIfClosed();
    	if (data == null && this.data != null) {
    		this.data.data().flush();
    		this.data.data().destroy();
    		this.data.close();
    		this.data = null;
    		return;
    	} else if (this.data != null) {
    		throw new IllegalArgumentException("Tensor '" + tensorName + "' has already "
    				+ "been defined. Cannot redefine the backend data of a tensor once it has"
    				+ " been set. In order to modify the tensor, please modify the NDArray "
    				+ "used as backend for the tensor.");
    	}
    	if (!emptyTensor && !equalShape(data.shape())) {
    		throw new IllegalArgumentException("Trying to set an INDArray as the backend of the Tensor "
    				+ "with a different shape than the Tensor. Tensor shape is: " + Arrays.toString(shape)
    				+ " and INDArray shape is: " + Arrays.toString(data.shape()));
    	}
    	if (!emptyTensor && this.data != null && this.data.dataType() != data.dataType()) {
    		throw new IllegalArgumentException("Trying to set an INDArray as the backend of the Tensor "
    				+ "with a different data type than the Tensor. Tensor data type is: " + dType.toString()
    				+ " and INDArray data type is: " + data.dataType().toString());
    	}
    	dType = data.dataType();
    	this.data = data;
    	if (emptyTensor) {
    		setShape();
        	dType = data.dataType();
        	emptyTensor = false;
    	}
    }
    
    /**
     * REturn the data in a software agnostic way using DJL NDArrays
     * @return the data of the tensor as a NDArray
     */
    public INDArray getDataAsNDArray() {
    	throwExceptionIfClosed();
    	if (data == null && isEmpty())
    		throw new IllegalArgumentException("Tensor '" + this.tensorName + "' is empty.");
    	else if (data == null)
    		throw new IllegalArgumentException("If you want to retrieve the tensor data as an NDArray,"
    				+ " please first transform the tensor data into an NDArray using: "
    				+ "TensorManager.buffer2array(tensor)");
    	return this.data;
    }
    
    /**
     * Copy the backend of a tensor (data either as an NDArray or Buffer)
     * @param tt
     * 	the tensor whose backedn is going to be copied
     */
    public void copyTensorBackend(Tensor tt) {
    	throwExceptionIfClosed();
    	if (tt.getDataAsNDArray() != null) {
    		copyNDArrayTensorBackend(tt);
    	}
    }
    
    /**
     * Copy the NDArray backend of a tensor 
     * @param tt
     * 	the tensor whose backedn is going to be copied
     */
    public void copyNDArrayTensorBackend(Tensor tt) {
    	throwExceptionIfClosed();
		setNDArrayData(tt.getDataAsNDArray());
    }
    
    /**
     * Method to convert the Tensor from its data type to another
     * @param dt
     * 	the data type into which the tensor is going to be converted
     */
    public void convertToDataType(DataType dt) {
    	data = data.castTo(dt);
    	this.dType = dt;
    }
    
    /**
     * Throw {@link IllegalStateException} if the tensor has been closed
     */
    private void throwExceptionIfClosed() {
    	if (!closed)
    		return;
    	throw new IllegalStateException("The tensor that is trying to be modified has already been "
    			+ "closed.");
    }

	/**
     * Empty the tensor information
     */
    public void close() {
    	if (closed)
    		return;
    	try {
		   	closed = true;
		   	axesArray = null;
		   	if (data != null) {
		   		data.data().setData(new boolean[] {false});
		   		data.data().destroy();
		   		data.close();
		   	}
		   	this.data = null;
		   	this.axesString = null;
		   	this.dType = null;
		   	this.shape = null;
		   	tensorName = null;
    	} catch(Exception ex) {
	   		closed = false;
	   		String msg = "Error trying to close tensor: " + tensorName + ". ";
	   		msg += ex.toString();
	   		throw new IllegalStateException(msg);
	   	}
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
    	return lTensors.stream().filter(pp -> !pp.isClosed() && pp.getName() != null && pp.getName().equals(name)).findAny().orElse(null);
    }
    
    /**
     * If the shape of a tensor is the same as the same  as the shape of this tensor
     * @param shape
     * 	the shape of the other tensor as a long arr
     * @return whether the tensor has the same shape to this tensor
     */
    private boolean equalShape(long[] longShape) {
    	if (longShape.length != this.shape.length)
    		return false;
    	for (int i = 0; i < longShape.length; i ++) {
    		if (((int) longShape[i]) != this.shape[i])
    			return false;
    	}
    	return true;
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
     * Set the shape of the tensor from the NDArray shape
     */
    private void setShape() {
    	if (data == null)
    		throw new IllegalArgumentException("Trying to create tensor from an empty NDArray");
    	long[] longShape = data.shape();
    	shape = new int[longShape.length];
    	for (int i = 0; i < shape.length; i ++)
    		shape[i] = (int) longShape[i];
    }
    
    /**
     * Get the name of the tensor
     * @return the name of the tensor
     */
    public String getName() {
    	return this.tensorName;
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
	public String getAxesOrderString() {
		return axesString;
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
     * Whether the tensor has already been filled with an NDArray or not
     * @return true if the tensor already has data or false otherwise
     */
    public boolean isEmpty() {
    	return emptyTensor;
    }
    
    /**
     * GEt the data type of the tensor
     * @return the data type of the tensor
     */
    public DataType getDataType() {
    	return data.dataType();
    }
    
    /**
     * Whether the tensor is closed or not
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
    	return closed;
    }
    
    /**
     * Method to check if the number of dimensions of the {@link INDArray} corresponds
     * to the number of dimensions specified by the {@link #axesString}
     * @param data
     * 	the array backend of the tensor
     * @param axesOrder
     * 	the axes order of the tensor
     */
    private void chechDims(INDArray data, String axesOrder) {
    	if (data.shape().length != axesOrder.length())
    		throw new IllegalArgumentException("The axes order introduced has to correspond "
    				+ "to the same number of dimenensions that the INDArray has. In this case"
    				+ " the axes order is specfied for " + axesOrder.length() + " dimensions "
						+ "while the array has " + data.shape().length + " dimensions.");
    }
    
    public static void main(String[] args) {
    	
    }
}
