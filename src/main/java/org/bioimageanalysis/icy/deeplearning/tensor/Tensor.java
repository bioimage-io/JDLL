package org.bioimageanalysis.icy.deeplearning.tensor;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;




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
	 * Data of the tensor stored in a buffer 
	 */
	private Buffer dataBuffer;
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
     * {@link TensorManager} that needs to be associated with each tensor
	 */
	private TensorManager manager;
	/**
	 * TODO make a more robust case for shape
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
     * @param manager
     * 	{@link TensorManager} that needs to be associated with each tensor
	 */
    private Tensor(String tensorName, String axes, INDArray data, TensorManager manager)
    {
    	Objects.requireNonNull(tensorName, "'tensorName' field should not be empty");
    	Objects.requireNonNull(axes, "'axes' field should not be empty");
    	Objects.requireNonNull(manager, "'manager' field should not be empty");
    	this.tensorName = tensorName;
    	this.axesString = axes;
    	this.axesArray = convertToTensorDimOrder(axes);
    	this.data = data;
    	this.manager = manager;
    	addToList();
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
     * @param manager
     * 	{@link TensorManager} that needs to be associated with each tensor
     * @return the tensor
     */
    public static Tensor build(String tensorName, String axes, INDArray data, TensorManager manager)
    {
    	if (data == null)
    		throw new IllegalArgumentException("Trying to create tensor from an empty NDArray");
    	return new Tensor(tensorName, axes, data, manager);
    }
    
    /**
     * Creates a tensor without data. The idea is to fill the data later.
	 * 
	 * @param tensorName
	 * 	name of the tensor as defined by the model
	 * @param axes
	 * 	String containing the axes order of the tensor. For example: "bcyx"
     * @param manager
     * 	{@link TensorManager} that needs to be associated with each tensor
     * @return the tensor
     */
    public static Tensor buildEmptyTensor(String tensorName, String axes, TensorManager manager)
    {
    	return new Tensor(tensorName, axes, null, manager);
    }
    
    /**
     * Method that converts the backend data of the tensor from an {@link NDArray}
     * object to a {@link Buffer} object. Regard that after the data is converted
     * into a Buffer, the NDArray is closed and set to null.
     */
    public void array2buffer() {
    	throwExceptionIfClosed();
    	Buffer dataBuffer;
    	if (getDataType() == DataType.INT8 || getDataType() == DataType.INT8)
    		dataBuffer = getDataAsNDArray().data().asNio();
    	else if (getDataType() == DataType.DOUBLE)
	    	dataBuffer = getDataAsNDArray().data().asNioDouble();
    	else if (getDataType() == DataType.FLOAT || getDataType() == DataType.FLOAT16)
	    	dataBuffer = getDataAsNDArray().data().asNioFloat();
    	else if (getDataType() == DataType.INT32)
	    	dataBuffer = getDataAsNDArray().data().asNioInt();
    	else if (getDataType() == DataType.INT64)
	    	dataBuffer = getDataAsNDArray().data().asNioLong();
    	else 
    		throw new IllegalArgumentException("Not supported data type: " + getDataType().toString());
    	getDataAsNDArray().close();
    	setNDArrayData(null);
    	setBufferData(dataBuffer);
    }
    
    /**
     * Method that converts the backend data of the tensor from an {@link Buffer} 
     * object to a {@link NDArray} object. Regard that after the data is converted
     * into a NDArray, the buffer is closed and set to null.
     */
    public void buffer2array () {
    	throwExceptionIfClosed();
    	NDArray ndarray = manager.getManager().create(getDataAsBuffer(),
    			Tensor.ndarrayShapeFromIntArr(getShape()), getDataType());
    	Nd4j.create(DataBuffer. getDataAsBuffer(),getShape(),'c');
    	setBufferData(null);
    	setNDArrayData(ndarray);
    }
    
    /**
     * Set the data structure of the tensor that contains the numbers
     * @param data
     * 	the numbers of the tensor in a Numpy array like structure
     */
    public void setNDArrayData(INDArray data) {
    	throwExceptionIfClosed();
    	if (data == null && this.data != null) {
    		this.data.close();
    		this.data = null;
    		return;
    	} else if (dataBuffer != null || this.data != null) {
    		throw new IllegalArgumentException("Tensor '" + tensorName + "' has already "
    				+ "been defined. Cannot redefine the backend data of a tensor once it has"
    				+ " been set. In order to modify the tensor, please modify the NDArray "
    				+ "used as backend for the tensor.");
    	}else if (data.getManager() != manager.getManager()) {
    		throw new IllegalArgumentException("The NDManager of the NDArray must be the same"
    				+ " as the NDManager of the tensor TensoManager (tensorManager.getManager()).");
    	}
    	this.data = data;
    	if (emptyTensor) {
    		setShape();
        	dType = data.getDataType();
        	emptyTensor = false;
    	}
    	if (!equalShape(data.getShape())) {
    		throw new IllegalArgumentException("Trying to set an NDArray as the backend of the Tensor "
    				+ "with a different shape than the Tensor. Tensor shape is: " + Arrays.toString(shape)
    				+ " and NDArray shape is: " + Arrays.toString(data.getShape().getShape()));
    	}
    	if (dType != data.getDataType()) {
    		throw new IllegalArgumentException("Trying to set an NDArray as the backend of the Tensor "
    				+ "with a different data type than the Tensor. Tensor data type is: " + dType.toString()
    				+ " and NDArray data type is: " + data.getDataType().toString());
    	}
    }
    
    /**
     * Set the data structure of the tensor that contains the numbers.
     * @param data
     * 	the numbers of the tensor in a buffer structure
     */
    public void setBufferData(Buffer bufferData) {
    	throwExceptionIfClosed();
    	// The tensor has to be first initialized with a NDArray as the backend.
    	// They cannot be initialized with Buffer
    	if (shape == null)
    		throw new IllegalArgumentException("The tensor '" + this.tensorName + "' has to be initialized with an NDArray"
    				+ " first, using: tensor.setNDArrayData(ndarray)");
    	// If there already exist some information as the backend, throw an exception
    	if (bufferData != null && data != null) {
    		throw new IllegalArgumentException("This tensor  already contains data. The data"
    				+ " of a tensor cannot be changed by setting another object as the backend."
    				+ " In order to modify a tensor, the backend NDArray has to be modified.");
    	}
    	this.dataBuffer = bufferData;
    }
    
    /**
     * REturn the data in a software agnostic way using DJL NDArrays
     * @return the data of the tensor as a NDArray
     */
    public INDArray getDataAsNDArray() {
    	throwExceptionIfClosed();
    	if (data == null && dataBuffer == null)
    		throw new IllegalArgumentException("Tensor '" + this.tensorName + "' is empty.");
    	else if (data == null)
    		throw new IllegalArgumentException("If you want to retrieve the tensor data as an NDArray,"
    				+ " please first transform the tensor data into an NDArray using: "
    				+ "TensorManager.buffer2array(tensor)");
    	return this.data;
    }
    
    /**
     * REturn the data in a software agnostic way using Buffers
     * @return the data of the tensor as a buffer
     */
    public Buffer getDataAsBuffer() {
    	throwExceptionIfClosed();
    	if (data == null && dataBuffer == null)
    		throw new IllegalArgumentException("Tensor '" + this.tensorName + "' is empty.");
    	else if (dataBuffer == null)
    		throw new IllegalArgumentException("If you want to retrieve the tensor data as a Buffer,"
    				+ " please first transform the tensor data into a Buffer using: "
    				+ "TensorManager.buffer2array(tensor)");
    	return this.dataBuffer;
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
    	} else {
    		copyBufferTensorBackend(tt);
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
     * Copy the Buffer backend of a tensor 
     * @param tt
     * 	the tensor whose backedn is going to be copied
     */
    public void copyBufferTensorBackend(Tensor tt) {
    	throwExceptionIfClosed();
    	if (tt.getDataAsBuffer() == null)
			throw new IllegalArgumentException("The source tensor to be copied from does not have a backend, it is empty.");
		if (emptyTensor) {
			emptyTensor = false;
	    	dataBuffer = tt.getDataAsBuffer();
	    	shape = tt.getShape();
	    	dType = tt.getDataType();
		} else if (dType != tt.getDataType()) {
			throw new IllegalArgumentException("The tensor to be copied from has a different data type."
					+ " Data types must be the same."
					+ " This data type of the tensor to be copied in is: " + dType.toString()
					+ " and the tensor to be copied from data type is: " + tt.getDataType().toString());
		} else if (tt.getShape().equals(shape)) {
			throw new IllegalArgumentException("The tensor to be copied from has a different shape."
					+ " Shapes must be the same."
					+ " The shape of the tensor to be copied in is: " + Arrays.toString(shape)
					+ " and the tensor to be copied from data type is: " + Arrays.toString(tt.getShape()));
		} else {
	    	dataBuffer = tt.getDataAsBuffer();
		}
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
		   	if (data != null)
		   		data.close();
		   	this.data = null;
		   	this.dataBuffer = null;
		   	this.axesString = null;
		   	this.dType = null;
		   	this.shape = null;
		   	manager = null;
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
     * Creates a tensor shape from an int array
     * 
     * @param shapeArr
     * 	int array with the size of each dimension
     * @return Shape with the image dimensions in the desired order.
     */
    public static Shape ndarrayShapeFromIntArr(int[] shapeArr)
    {
        long[] dimensionSizes = new long[shapeArr.length];
        for (int i = 0; i < dimensionSizes.length; i++)
        {
        	dimensionSizes[i] = (long) shapeArr[i];
        }
        return new Shape(dimensionSizes);
    }
    
    /**
     * If the shape of a tensor is the same as the same  as the shape of this tensor
     * @param shape
     * 	the shape of the other tensor
     * @return whether the tensor has the same shape to this tensor
     */
    private boolean equalShape(Shape shape) {
    	long[] longShape = shape.getShape();
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
     * Add tensor to list of tensors owned by the parent TensorManager
     */
    private void addToList() {
    	List<Tensor> list = manager.getTensorList();
    	Tensor coincidence = getTensorByNameFromList(list, tensorName);
    	if (coincidence != null) {
    		throw new IllegalArgumentException("There already exists a Tensor called '" + tensorName
    				+ "' in the list of tensors associated to the parent TensorManager. Tensor names"
    				+ " for the same TEnsorManager should be unique.");
    	}
    	manager.addTensor(this);
    }
    
    /**
     * Set the shape of the tensor from the NDArray shape
     */
    private void setShape() {
    	if (data == null)
    		throw new IllegalArgumentException("Trying to create tensor from an empty NDArray");
    	long[] longShape = data.getShape().getShape();
    	shape = new int[longShape.length];
    	for (int i = 0; i < shape.length; i ++)
    		shape[i] = (int) longShape[i];
    }
    
    /**
     * Get the parent TensorManager of this tensor
     * @return the parent TensorManager
     */
    public TensorManager getManager() {
    	return manager;
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
    	return dType;
    }
    
    /**
     * Whether the tensor is closed or not
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
    	return closed;
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
