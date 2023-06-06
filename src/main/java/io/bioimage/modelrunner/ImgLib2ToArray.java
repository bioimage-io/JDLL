/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

/**
 * Class that maps {@link Tensor} objects to {@link ByteBuffer} objects.
 * This is done to modify the files that are used to communicate between process
 * to avoid the TF2-TF1/Pytorch incompatibility that happens in these systems
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ImgLib2ToArray
{
	/**
	 * Header used to identify files for interprocessing communication
	 */
	final public static byte[] MODEL_RUNNER_HEADER = 
		{(byte) 0x93, 'M', 'O', 'D', 'E', 'L', '-', 'R', 'U', 'N', 'N', 'E', 'R'};

    /**
     * Not used (Utility class).
     */
    private ImgLib2ToArray()
    {
    }

    /**
     * Maps a {@link Tensor} to the provided {@link ByteBuffer} with all the information
     * needed to reconstruct the tensor again
     * 
     * @param <T> 
     * 	the type of the tensor
     * @param tensor 
     * 	tensor to be mapped into byte buffer
     * @param byteBuffer 
     * 	target byte bufer
     * @throws IllegalArgumentException
     *         If the {@link Tensor} ImgLib2 type is not supported.
     */
    public static < T extends RealType< T > & NativeType< T > > void build(Tensor<T> tensor, ByteBuffer byteBuffer)
    {
		byteBuffer.put(ImgLib2ToArray.createFileHeader(tensor));
		if (tensor.isEmpty())
			return;
    	build(tensor.getData(), byteBuffer);
    }

    /**
     * Adds the {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    private static <T extends Type<T>> void build(RandomAccessibleInterval<T> rai, ByteBuffer byteBuffer)
    {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		buildByte((RandomAccessibleInterval<ByteType>) rai, byteBuffer);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		buildInt((RandomAccessibleInterval<IntType>) rai, byteBuffer);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		//buildFloat((RandomAccessibleInterval<FloatType>) rai, byteBuffer);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		buildDouble((RandomAccessibleInterval<DoubleType>) rai, byteBuffer);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
    }

    /**
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static void buildByte(RandomAccessibleInterval<ByteType> imgTensor, ByteBuffer byteBuffer)
    {
    	Cursor<ByteType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<ByteType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteBuffer.put(tensorCursor.get().getByte());
		}
    }

    /**
     * Adds the IntType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static void buildInt(RandomAccessibleInterval<IntType> imgTensor, ByteBuffer byteBuffer)
    {
    	Cursor<IntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<IntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteBuffer.putInt(tensorCursor.get().getInt());
		}
    }

    /**
     * Adds the FloatType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    public static float[] buildFloat(RandomAccessibleInterval<FloatType> imgTensor)
    {
    	Cursor<FloatType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<FloatType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		int size = 1; 
		for (long i : imgTensor.dimensionsAsLongArray()) {size *= i;}
		float[] floatArr = new float[size];
		int c = 0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			floatArr[c ++] = tensorCursor.get().getRealFloat();
		}
		return floatArr;
    }

    /**
     * Adds the DoubleType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static void buildDouble(RandomAccessibleInterval<DoubleType> imgTensor, ByteBuffer byteBuffer)
    {
    	Cursor<DoubleType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
        	byteBuffer.putDouble(tensorCursor.get().getRealDouble());
		}
    }
    
    /**
     * Create header for the temp file that is used for interprocess communication.
     * The header should contain the first key word as an array of bytes (MODEl-RUNNER)
     * @param <T> 
     * 	type of the tensor
     * @param tensor
     * 	tensor whose info is recorded
     * @return byte array containing the header info for the file
     */
    public static < T extends RealType< T > & NativeType< T > > byte[] 
    		createFileHeader(io.bioimage.modelrunner.tensor.Tensor<T> tensor) {
    	String dimsStr = 
    			!tensor.isEmpty() ? Arrays.toString(tensor.getData().dimensionsAsLongArray()) : "[]";
    	T dtype = !tensor.isEmpty() ? Util.getTypeFromInterval(tensor.getData()): (T) new FloatType();
    	String descriptionStr = "{'dtype':'" 
    			+ getDataTypeString(dtype) + "','axes':'" 
    			+ tensor.getAxesOrderString() + "','name':'" + tensor.getName() +  "','shape':'" 
    			+ dimsStr + "'}";
    	
    	byte[] descriptionBytes = descriptionStr.getBytes(StandardCharsets.UTF_8);
    	int lenDescriptionBytes = descriptionBytes.length;
    	byte[] intAsBytes = ByteBuffer.allocate(4).putInt(lenDescriptionBytes).array();
    	int totalHeaderLen = MODEL_RUNNER_HEADER.length + intAsBytes.length + lenDescriptionBytes;
    	byte[] byteHeader = new byte[totalHeaderLen];
    	for (int i = 0; i < MODEL_RUNNER_HEADER.length; i ++)
    		byteHeader[i] = MODEL_RUNNER_HEADER[i];
    	for (int i = MODEL_RUNNER_HEADER.length; i < MODEL_RUNNER_HEADER.length + intAsBytes.length; i ++)
    		byteHeader[i] = intAsBytes[i - MODEL_RUNNER_HEADER.length];
    	for (int i = MODEL_RUNNER_HEADER.length + intAsBytes.length; i < totalHeaderLen; i ++)
    		byteHeader[i] = descriptionBytes[i - MODEL_RUNNER_HEADER.length - intAsBytes.length];
    	
    	return byteHeader;
    }
    
    /**
     * Method that returns a Sting representing the datatype of T
     * @param <T>
     * 	type of the tensor
     * @param type
     * 	pixel of an imglib2 object to get the info of teh data type
     * @return String representation of the datatype
     */
    public static< T extends RealType< T > & NativeType< T > > String getDataTypeString(T type) {
    	if (type instanceof ByteType) {
    		return "byte";
    	} else if (type instanceof IntType) {
    		return "int32";
    	} else if (type instanceof FloatType) {
    		return "float32";
    	} else if (type instanceof DoubleType) {
    		return "float64";
    	} else if (type instanceof LongType) {
    		return "int64";
    	} else if (type instanceof UnsignedByteType) {
    		return "ubyte";
    	} else {
            throw new IllegalArgumentException("Unsupported data type. At the moment the only "
            		+ "supported dtypes are: " + IntType.class + ", " + FloatType.class + ", "
            		 + DoubleType.class + ", " + LongType.class + " and " + UnsignedByteType.class);
    	}
    }

    /**
     * Get the total byte size of the temp file that is going to be created to be
     * able to reconstruct a {@link Tensor} to in the separate process in MacOS Intel
     * systems
     * 
     * @param <T>
     * 	type of the imglib2 object
     * @param tensor
     * 	tensor of interest
     * @return number of bytes needed to create a file with the info of the tensor
     */
    public static  < T extends RealType< T > & NativeType< T > > long 
    		findTotalLengthFile(io.bioimage.modelrunner.tensor.Tensor<T> tensor) {
    	long startLen = createFileHeader(tensor).length;
    	long[] dimsArr = !tensor.isEmpty() ? tensor.getData().dimensionsAsLongArray() : null;
    	if (dimsArr == null)
    		return startLen;
    	long totSizeFlat = 1;
    	for (long i : dimsArr) {totSizeFlat *= i;}
    	long nBytesDt = 1;
    	Type<T> dtype = !tensor.isEmpty() ? 
    			Util.getTypeFromInterval(tensor.getData()) : (Type<T>) new FloatType();
    	if (dtype instanceof IntType) {
    		nBytesDt = 4;
    	} else if (dtype instanceof ByteType) {
    		nBytesDt = 1;
        } else if (dtype instanceof FloatType) {
        	nBytesDt = 4;
        } else if (dtype instanceof DoubleType) {
        	nBytesDt = 8;
        } else {
            throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
    	return startLen + nBytesDt * totSizeFlat;
    }
    
    public T[] createArray(int length) {
        @SuppressWarnings("unchecked")
        T[] array = (T[]) Array.newInstance(Number.class, length);
        return array;
    }
}
