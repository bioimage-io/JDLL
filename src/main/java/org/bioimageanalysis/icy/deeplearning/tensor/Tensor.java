package org.bioimageanalysis.icy.deeplearning.tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;

/**
 * Tensors created to interact with a Deep Learning engine while being agnostic
 * to it. This class just contains the information to create a tensor while
 * maintaining flexibility to interact with any wanted Deep Learning framework.
 *
 * @author Carlos Garcia Lopez de Haro
 */
public final class Tensor< T extends RealType< T > & NativeType< T > >
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
	private RandomAccessibleInterval< T > data;

	/**
	 * Whether the tensor represents an image or not
	 */
	private boolean isImage = true;

	/**
	 * Whether the tensor has been created without an NDarray or not. Once the
	 * NDarray is added, the tensor cannot be empty anymore
	 */
	private boolean emptyTensor;

	/**
	 * TODO develop a DAtaType class for this Tensor class? 
	 * The data type of the tensor
	 */
	private Type< T > dType;

	/**
	 * Shape of the tensor
	 */
	private int[] shape;

	/**
	 * Whether the tensor is closed or not. If it is, nothing can be done on the
	 * Tensor
	 */
	private boolean closed = false;

	/**
	 * Create the tensor object.
	 *
	 * @param tensorName
	 *            name of the tensor as defined by the model
	 * @param axes
	 *            String containing the axes order of the tensor. For example:
	 *            "bcyx"
	 * @param data
	 *            data structure similar to a Numpy array that contains all
	 *            tensor numbers
	 */
	private Tensor( final String tensorName, final String axes, final RandomAccessibleInterval< T > data )
	{
		Objects.requireNonNull( tensorName, "'tensorName' field should not be empty" );
		Objects.requireNonNull( axes, "'axes' field should not be empty" );
		if ( data != null )
			checkDims( data, axes );
		this.tensorName = tensorName;
		this.axesString = axes;
		this.axesArray = convertToTensorDimOrder( axes );
		this.data = data;
		if ( data != null )
		{
			setShape();
			dType = Util.getTypeFromInterval( data );
			emptyTensor = false;
		}
		else
		{
			emptyTensor = true;
		}
	}

	/**
	 * Return a tensor object
	 *
	 * @param tensorName
	 *            name of the tensor as defined by the model
	 * @param axes
	 *            String containing the axes order of the tensor. For example:
	 *            "bcyx"
	 * @param data
	 *            data structure similar to a Numpy array that contains all
	 *            tensor numbers
	 * @return the tensor
	 */
	public static < T extends RealType< T > & NativeType< T > > Tensor< T > build( final String tensorName, final String axes, final RandomAccessibleInterval< T > data )
	{
		if ( data == null )
			throw new IllegalArgumentException( "Trying to create tensor from an empty NDArray" );
		return new Tensor<>( tensorName, axes, data );
	}

	/**
	 * Creates a tensor without data. The idea is to fill the data later.
	 *
	 * @param tensorName
	 *            name of the tensor as defined by the model
	 * @param axes
	 *            String containing the axes order of the tensor. For example:
	 *            "bcyx"
	 * @return the tensor
	 */
	public static < T extends RealType< T > & NativeType< T > > Tensor< T > buildEmptyTensor( final String tensorName, final String axes )
	{
		return new Tensor<>( tensorName, axes, null );
	}

	/**
	 * Creates a tensor without data. However, the memory that this tensor will consume is already
	 * allocated during its creation
	 *
	 * @param tensorName
	 *            name of the tensor as defined by the model
	 * @param axes
	 *            String containing the axes order of the tensor. For example:
	 *            "bcyx"
	 * @param shape
	 * 			  Shape of the tensor
	 * @param dtype
	 * 			  data type of the tensor
	 * @return the tensor
	 */
	public static < T extends RealType< T > & NativeType< T > , R extends RealType< R > & NativeType< R > > 
				Tensor< R > buildEmptyTensorAndAllocateMemory( final String tensorName, 
																final String axes, final long[] shape,
																final R dtype)
	{
		final ImgFactory< R > imgFactory = new CellImgFactory<>( dtype, 5 );
		final Img<R> backendData = imgFactory.create(shape);
		return new Tensor<>( tensorName, axes, backendData );
	}

	/**
	 * Set the data structure of the tensor that contains the numbers. In order
	 * to change the data of the tensor, first do 'tensor.setData(null)'. Once
	 * the tensor data is null, set the wanted {@link Random
	 * AccessibleInterval}. The {@link RandomAccessibleInerval} needs to be of
	 * the same data type, shape and size as the tensor data.
	 *
	 * @param data
	 *            the numbers of the tensor in a Numpy array like structure
	 */
	/** TODO remove once it is clear that tensors can be rewritten
	public void setData( final RandomAccessibleInterval< T > data )
	{
		throwExceptionIfClosed();
		if ( data == null && this.data != null )
		{
			this.data = null;
			return;
		}
		else if ( this.data != null )
		{ throw new IllegalArgumentException( "Tensor '" + tensorName + "' has already "
				+ "been defined. Cannot redefine the backend data of a tensor once it has"
				+ " been set. In order to modify the tensor, please modify the NDArray "
				+ "used as backend for the tensor." ); }
		if ( !emptyTensor && !equalShape( data.dimensionsAsLongArray() ) )
		{ throw new IllegalArgumentException( "Trying to set an NDArray as the backend of the Tensor "
				+ "with a different shape than the Tensor. Tensor shape is: " + Arrays.toString( shape )
				+ " and NDArray shape is: " + Arrays.toString( data.dimensionsAsLongArray() ) ); }
		if ( !emptyTensor && this.data != null
				&& Util.getTypeFromInterval( this.data ) != Util.getTypeFromInterval( data ) )
		{ throw new IllegalArgumentException( "Trying to set an NDArray as the backend of the Tensor "
				+ "with a different data type than the Tensor. Tensor data type is: " + dType.toString()
				+ " and NDArray data type is: " + Util.getTypeFromInterval( data ).toString() ); }
		if ( !emptyTensor )
			checkDims( data, axesString );

		dType = Util.getTypeFromInterval( data );
		this.data = data;
		if ( emptyTensor )
		{
			setShape();
			dType = Util.getTypeFromInterval( data );
			emptyTensor = false;
		}
	}
	*/
	public void setData( final RandomAccessibleInterval< T > data )
	{
		throwExceptionIfClosed();
		if ( data == null && this.data != null ) {
			this.data = null;
			return;
		}
		
		if ( !emptyTensor )
			checkDims( data, axesString );
		
		if ( !emptyTensor && !equalShape( data.dimensionsAsLongArray() ) ) { 
			throw new IllegalArgumentException( "Trying to set an array as the backend of the Tensor "
				+ "with a different shape than the Tensor. Tensor shape is: " + Arrays.toString( shape )
				+ " and array shape is: " + Arrays.toString( data.dimensionsAsLongArray() ) ); 
		}
		if ( !emptyTensor && this.data != null
				&& this.getDataType().getClass() != Util.getTypeFromInterval( data ).getClass() ) { 
			throw new IllegalArgumentException( "Trying to set an array as the backend of the Tensor "
				+ "with a different data type than the Tensor. Tensor data type is: " + dType.toString()
				+ " and array data type is: " + Util.getTypeFromInterval( data ).toString() ); 
		}
		/**
		 * TODO
		 * Copy or reference?
		if (this.data == null) {
			final ImgFactory< T > factory = Util.getArrayOrCellImgFactory( data, Util.getTypeFromInterval(data) );
			this.data = factory.create( data );
		}
		 */
		
		if (this.data == null) {
			this.data = data;
		} else {
			LoopBuilder.setImages( this.data, data )
				.multiThreaded().forEachPixel( ( i, o ) -> o.set( i ) );
		}
		
		if ( emptyTensor ) {
			setShape();
			dType = Util.getTypeFromInterval( data );
			emptyTensor = false;
		}
	}

	/**
	 *
	 * @return the data of the tensor as a RandomAccessible interval
	 */
	public RandomAccessibleInterval< T > getData()
	{
		throwExceptionIfClosed();
		if ( data == null && isEmpty() )
			throw new IllegalArgumentException( "Tensor '" + this.tensorName + "' is empty." );
		else if ( data == null )
			throw new IllegalArgumentException( "If you want to retrieve the tensor data as an NDArray,"
					+ " please first transform the tensor data into an NDArray using: "
					+ "TensorManager.buffer2array(tensor)" );
		return this.data;
	}

	/**
	 * Copy from the backend of a tensor 
	 *
	 * @param source
	 *            the tensor whose backend is going to be copied. Must be of the
	 *            same type than this tensor.
	 */
	public void copyTensorBackend( final Tensor< T > source )
	{
		throwExceptionIfClosed();
		if ( source.getData() != null )
			copyRAITensorBackend( source );
	}

	/**
	 * Copy the NDArray backend of a tensor
	 *
	 * @param source
	 *            the tensor whose backend is going to be copied. Must be of the
	 *            same type than this tensor.
	 */
	public void copyRAITensorBackend( final Tensor< T > source )
	{
		throwExceptionIfClosed();
		setData( source.getData() );
	}

	/**
	 * Method that creates a copy of the tensor in the wanted data type.
	 * Everything is the same or the new tensor (including the name), except the
	 * data type of the data
	 *
	 * @param tt
	 *            tensor where the copy is created from
	 * @param type
	 *            data type of the wanted tensor
	 */
	public static < T extends RealType< T > & NativeType< T >, R extends RealType< R > & NativeType< R > > Tensor< R > createCopyOfTensorInWantedDataType( final Tensor< T > tt, final R type )
	{
		tt.throwExceptionIfClosed();
		final RandomAccessibleInterval< T > input = tt.getData();

		final ImgFactory< R > factory = Util.getArrayOrCellImgFactory( input, type );
		final Img< R > output = factory.create( input );
		RealTypeConverters.copyFromTo( input, output );
		return Tensor.build( tt.getName(), tt.getAxesOrderString(), output );
	}

	/**
	 * Method that creates a copy of the tensor in the wanted data type.
	 * Everything is the same or the new tensor (including the name), except the
	 * data type of the data
	 *
	 * @param tt
	 *            tensor where the copy is created from
	 * @param type
	 *            data type of the wanted tensor
	 */
	public static < T extends RealType< T > & NativeType< T >, R extends RealType< R > & NativeType< R > > RandomAccessibleInterval< R > createCopyOfRaiInWantedDataType( final RandomAccessibleInterval< T > input, final R type )
	{
		final ImgFactory< R > factory = Util.getArrayOrCellImgFactory( input, type );
		final Img< R > output = factory.create( input );
		RealTypeConverters.copyFromTo( input, output );
		return output;
	}

	/**
	 * Throw {@link IllegalStateException} if the tensor has been closed
	 */
	private void throwExceptionIfClosed()
	{
		if ( !closed )
			return;
		throw new IllegalStateException( "The tensor that is trying to be modified has already been " + "closed." );
	}

	/**
	 * Empty the tensor information
	 */
	public void close()
	{
		if ( closed )
			return;
		try
		{
			closed = true;
			axesArray = null;
			if ( data != null )
			{
				data = null;
			}
			this.data = null;
			this.axesString = null;
			this.dType = null;
			this.shape = null;
			tensorName = null;
		}
		catch ( final Exception ex )
		{
			closed = false;
			String msg = "Error trying to close tensor: " + tensorName + ". ";
			msg += ex.toString();
			throw new IllegalStateException( msg );
		}
	}

	/**
	 * Retrieve tensor with the wanted name from a list of tensors
	 *
	 * @param lTensors
	 *            list of tensors
	 * @param name
	 *            name of the tensor of interest
	 * @return the tensor of interest
	 */
	public static Tensor< ? > getTensorByNameFromList( final List< Tensor< ? > > lTensors, final String name )
	{
		return lTensors.stream().filter( pp -> !pp.isClosed() && pp.getName() != null && pp.getName().equals( name ) )
				.findAny().orElse( null );
	}

	/**
	 * If the shape of a tensor is the same as the same as the shape of this
	 * tensor
	 *
	 * @param shape
	 *            the shape of the other tensor as a long arr
	 * @return whether the tensor has the same shape to this tensor
	 */
	private boolean equalShape( final long[] longShape )
	{
		if ( longShape.length != this.shape.length )
			return false;
		for ( int i = 0; i < longShape.length; i++ )
		{
			if ( ( ( int ) longShape[ i ] ) != this.shape[ i ] )
				return false;
		}
		return true;
	}

	/**
	 * Convert the String representation of the axes order into an int array
	 * representation, easier to handle by the program
	 *
	 * @param dimOrder
	 *            String representation of the axes
	 * @return the int[] representation of the axes
	 * @throws IllegalArgumentException
	 *             if the String representation contains repeated axes
	 */
	public static int[] convertToTensorDimOrder( String dimOrder ) throws IllegalArgumentException
	{
		dimOrder = dimOrder.toLowerCase();
		final int[] tensorDimOrder = new int[ dimOrder.length() ];
		int hasB = 0, hasI = 0, hasT = 0, hasX = 0, hasY = 0, hasZ = 0, hasC = 0;
		final int hasR = 0;
		for ( int i = 0; i < dimOrder.length(); i++ )
		{
			switch ( dimOrder.charAt( i ) )
			{
			case 'b':
				tensorDimOrder[ i ] = 4;
				hasB = 1;
				break;
			case 'i':
				tensorDimOrder[ i ] = 3;
				hasI = 1;
				break;
			case 'r':
				tensorDimOrder[ i ] = 3;
				hasI = 1;
				break;
			case 't':
				tensorDimOrder[ i ] = 4;
				hasT = 1;
				break;
			case 'z':
				tensorDimOrder[ i ] = 3;
				hasZ += 1;
				break;
			case 'c':
				tensorDimOrder[ i ] = 2;
				hasC += 1;
				break;
			case 'y':
				tensorDimOrder[ i ] = 1;
				hasY += 1;
				break;
			case 'x':
				tensorDimOrder[ i ] = 0;
				hasX += 1;
				break;
			default:
				throw new IllegalArgumentException(
						"Illegal axis for tensor dim order " + dimOrder + " (" + dimOrder.charAt( i ) + ")" );
			}
		}
		if ( hasB + hasT > 1 )
			throw new IllegalArgumentException(
					"Tensor axes order can only have either one 'b' or " + "one 't'. These axes are exclusive ." );
		else if ( hasZ + hasR + hasI > 1 )
			throw new IllegalArgumentException(
					"Tensor axes order can only have either one 'i', one 'z' or " + "one 'r'." );
		else if ( hasY > 1 || hasX > 1 || hasC > 1 || hasZ > 1 || hasR > 1 || hasT > 1 || hasI > 1 || hasB > 1 )
			throw new IllegalArgumentException( "There cannot be repeated dimensions in the axes "
					+ "order as this tensor has (" + dimOrder + ")." );
		return tensorDimOrder;
	}

	/**
	 * Set the shape of the tensor from the NDArray shape
	 */
	private void setShape()
	{
		if ( data == null )
			throw new IllegalArgumentException( "Trying to create tensor from an empty NDArray" );
		final long[] longShape = data.dimensionsAsLongArray();
		shape = new int[ longShape.length ];
		for ( int i = 0; i < shape.length; i++ )
			shape[ i ] = ( int ) longShape[ i ];
	}

	/**
	 * Get the name of the tensor
	 *
	 * @return the name of the tensor
	 */
	public String getName()
	{
		throwExceptionIfClosed();
		return this.tensorName;
	}

	/**
	 * REturns the shape of the tensor
	 *
	 * @return the shape of the tensor
	 */
	public int[] getShape()
	{
		throwExceptionIfClosed();
		return shape;
	}

	/**
	 * REtrieve the axes order in String form
	 *
	 * @return the axesString
	 */
	public String getAxesOrderString()
	{
		throwExceptionIfClosed();
		return axesString;
	}

	/**
	 * Return the array containing the int representation of the axes order
	 *
	 * @return the axes order in int[] representation
	 */
	public int[] getAxesOrder()
	{
		throwExceptionIfClosed();
		return this.axesArray;
	}

	/**
	 * Set whether the tensor represents an image or not
	 *
	 * @param isImage
	 *            if the tensor is an image or not
	 */
	public void setIsImage( final boolean isImage )
	{
		throwExceptionIfClosed();
		if ( !isImage )
			assertIsList();
		this.isImage = isImage;
	}

	/**
	 * Whether the tensor represents an image or not
	 *
	 * @return true if the tensor represents an image, false otherwise
	 */
	public boolean isImage()
	{
		throwExceptionIfClosed();
		return isImage;
	}

	/**
	 * Whether the tensor has already been filled with an NDArray or not
	 *
	 * @return true if the tensor already has data or false otherwise
	 */
	public boolean isEmpty()
	{
		throwExceptionIfClosed();
		return emptyTensor;
	}

	/**
	 * GEt the data type of the tensor
	 *
	 * @return the data type of the tensor
	 */
	public Type< T > getDataType()
	{
		throwExceptionIfClosed();
		return Util.getTypeFromInterval( data );
	}

	/**
	 * Whether the tensor is closed or not
	 *
	 * @return true if closed, false otherwise
	 */
	public boolean isClosed()
	{
		return closed;
	}

	/**
	 * Method to check if the number of dimensions of the
	 * {@link RandomAccessibleInterval} corresponds to the number of dimensions
	 * specified by the {@link #axesString}
	 *
	 * @param data
	 *            the array backend of the tensor
	 * @param axesOrder
	 *            the axes order of the tensor
	 */
	private void checkDims( final RandomAccessibleInterval< T > data, final String axesOrder )
	{
		if ( data.dimensionsAsLongArray().length != axesOrder.length() )
			throw new IllegalArgumentException( "The axes order introduced has to correspond "
					+ "to the same number of dimenensions that the NDArray has. In this case"
					+ " the axes order is specfied for " + axesOrder.length() + " dimensions " + "while the array has "
					+ data.dimensionsAsLongArray().length + " dimensions." );
	}

	private void assertIsList()
	{
		final boolean x = axesString.toLowerCase().indexOf( "x" ) != -1;
		final boolean y = axesString.toLowerCase().indexOf( "y" ) != -1;
		final boolean t = axesString.toLowerCase().indexOf( "t" ) != -1;
		final boolean z = axesString.toLowerCase().indexOf( "z" ) != -1;
		if ( x || y || t || z )
		{ throw new IllegalArgumentException( "Tensor '" + this.tensorName + "' cannot be represented as "
				+ "a ist because lists can only have the axes: 'b', 'i', 'c' and 'r'. The axes for this "
				+ "tensor are :" + axesString + "." ); }
	}

	public static void main( final String[] args )
	{

	}
}
