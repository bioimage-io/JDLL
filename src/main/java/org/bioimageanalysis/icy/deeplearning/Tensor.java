package org.bioimageanalysis.icy.deeplearning;

import icy.sequence.Sequence;

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
	 * Name given to the tensor in the model.
	 */
	private int[] axesArray;
	/**
	 * Name given to the tensor in the model. Cannot be modified
	 * after it has been set
	 */
	private Sequence sequence;
	
	/**
	 * Create the tensor object. 
	 * 
	 * @param tensorName
	 * 	name of the tensor as defined by the model
	 * @param axes
	 * 	String containing the axes order of the tensor. For example: "bcyx"
	 * @param sequence
	 * 	the sequence that contains the information that will be transformed into
	 * 	tensor
	 */
    private Tensor(String tensorName, String axes, Sequence sequence)
    {
    	this.tensorName = tensorName;
    	this.axesArray = convertToTensorDimOrder(axes);
    	this.sequence = sequence;
    }
    
    /**
     * Return a tensor object
	 * 
	 * @param tensorName
	 * 	name of the tensor as defined by the model
	 * @param axes
	 * 	String containing the axes order of the tensor. For example: "bcyx"
	 * @param sequence
	 * 	the sequence that contains the information that will be transformed into
	 * 	tensor
     * @return the tensor
     */
    public static Tensor build(String tensorName, String axes, Sequence sequence)
    {
    	return new Tensor(tensorName, axes, sequence);
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
    private int[] convertToTensorDimOrder(String dimOrder) throws IllegalArgumentException
    {
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
     * Set the data of the Tensor 
     * @param seq
     * 	data of the tensor in the for of an Icy Sequence
     */
    public void setData(Sequence seq) {
    	this.sequence = seq;
    }
    
    /**
     * Returns the data of the tensor. This data is contained in a sequence
     * @return the sequence containing the data of the tensor
     */
    public Sequence getData() {
    	return this.sequence;
    }
    
    /**
     * Empty the tensor information
     */
    public void close() {
	   	tensorName = null;
	   	axesArray = null;
	   	sequence.close();;
   	
    }
}
