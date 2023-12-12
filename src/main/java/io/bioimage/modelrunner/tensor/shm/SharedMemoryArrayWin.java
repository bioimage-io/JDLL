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
package io.bioimage.modelrunner.tensor.shm;

import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.UUID;

import com.sun.jna.Pointer;
import com.sun.jna.platform.win32.Kernel32;
import com.sun.jna.platform.win32.WinBase;
import com.sun.jna.platform.win32.WinNT;
import com.sun.jna.platform.win32.WinNT.HANDLE;
import com.sun.jna.platform.win32.BaseTSD;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Class that maps {@link Tensor} objects to the shared memory for interprocessing communication
 * in Windows
 * @author Carlos Garcia Lopez de Haro
 */
public class SharedMemoryArrayWin implements SharedMemoryArray
{
	/**
	 * file mapping for shared memory
	 */
	private WinNT.HANDLE hMapFile;
	/**
	 * 
	 */
	private Pointer pSharedMemory;
	/**
	 * Name defining the location of the shared memory block
	 */
	private final String memoryName;
	/**
	 * Size of the shared memory block
	 */
	private int size;
	/**
	 * Datatype of the shm array
	 */
	private final String originalDataType;
	/**
	 * Original dimensions of the shm array
	 */
	private final long[] originalDims;
	/**
	 * Whether the shared memory segment has been already closed and unlinked or not
	 */
	private boolean unlinked = false;
	/**
	 * Whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
	 */
	private boolean isNumpyFormat = false;
	
    private SharedMemoryArrayWin(int size, String dtype, long[] shape)
    {
    	this("Local\\" + UUID.randomUUID().toString(), size, dtype, shape);
    }
	
    private SharedMemoryArrayWin(String name, int size, String dtype, long[] shape)
    {
    	memoryName = name;
    	this.originalDataType = dtype;
    	this.originalDims = shape;
    	this.size = size;
        hMapFile = Kernel32.INSTANCE.CreateFileMapping(
                WinBase.INVALID_HANDLE_VALUE,
                null,
                WinNT.PAGE_READWRITE,
                0,
                size,
                memoryName
        );
        
        if (hMapFile == null) {
            throw new RuntimeException("Error creating shared memory array. CreateFileMapping failed: "
            		+ "" + Kernel32.INSTANCE.GetLastError());
        }
        
        // Map the shared memory
        pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(
                hMapFile,
                WinNT.FILE_MAP_WRITE,
                0,
                0,
                size
        );
        
        if (pSharedMemory == null) {
            Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("Error creating shared memory array. MapViewOfFile failed: "
            		+ "" + Kernel32.INSTANCE.GetLastError());
        }
    }
    
    public String getName() {
    	return this.memoryName;
    }
    
    public String getNameForPython() {
    	return this.memoryName.substring("Local\\".length());
    }
    
    public Pointer getPointer() {
    	return this.pSharedMemory;
    }
    
    public HANDLE getSharedMemoryBlock() {
    	return this.hMapFile;
    }
    
    public int getSize() {
    	return this.size;
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
     * @return an instance of {@link SharedMemoryArrayWin} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayWin build(RandomAccessibleInterval<T> rai)
    {
    	String name = "Local\\" + UUID.randomUUID().toString();
    	return build(name, rai);
    }

    /**
     * Adds the {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param name
     * 	name of the memory location where the shm segment is going to be created, cannot contain any special character
     * and should start by "Local\\" in windows. The shm name is generated automatically with the the method {@link #build(String, RandomAccessibleInterval)}
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     * @return an instance of {@link SharedMemoryArrayWin} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported or if the name does
     * 	not fulfil the requirements.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayWin build(String name, RandomAccessibleInterval<T> rai)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("Local\\"))
    		name = "Local\\" + name;
		SharedMemoryArrayWin shma = null;
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt64(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildFloat32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildFloat64(Cast.unchecked(rai));
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
		return shma;
    }

    /**
     * Creates a shared memory segment containing the information of the {@link RandomAccessibleInterval}.
     * The shared memory segment is created with numpy format, which means that it will have a header of bytes indicating
     * the data type, shape and byte ordering
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param name
     * 	name of the memory location where the shm segment is going to be created, cannot contain any special character
     * and should should start by "Local\\" in windows. The shm name is generated automatically with the the method {@link #build(String, RandomAccessibleInterval)}
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @return an instance of {@link SharedMemoryArrayLinux} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayWin buildNumpyFormat(RandomAccessibleInterval<T> rai)
    {
    	return buildNumpyFormat("Local\\" + UUID.randomUUID().toString(), rai);
    }

    /**
     * Creates a shared memory segment containing the information of the {@link RandomAccessibleInterval} in the 
     * location specified by the 'name' argument. 
     * The shared memory segment is created with numpy format, which means that it will have a header of bytes indicating
     * the data type, shape and byte ordering
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param name
     * 	name of the memory location where the shm segment is going to be created, cannot contain any special character
     *  and should start by "Local\\" in windows. The shm name is generated 
     *  automatically with the the method {@link #build(String, RandomAccessibleInterval)}. In Macos, names should not be 
     *  longer than 30 characters
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @return an instance of {@link SharedMemoryArrayLinux} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayWin buildNumpyFormat(String name, RandomAccessibleInterval<T> rai)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayWin shma = null;
    	byte[] total = DecodeNumpy.createNumpyStyleByteArray(rai);
    	shma = new SharedMemoryArrayWin(name, total.length, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
    	shma.addByteArray(total);
    	shma.isNumpyFormat = true;
		return shma;
    }
    
    /**
     * Add a byte array to the shm segment
     * @param arr
     * 	the byte array that is going to be added
     */
    private void addByteArray(byte[] arr) {
    	for (int i = 0; i < arr.length; i ++) {
    		this.pSharedMemory.setByte(i, arr[i]);
    	}
    }

    /**
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildInt8(RandomAccessibleInterval<ByteType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<ByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setByte(i ++, cursor.get().get());
		}
    }

    /**
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildUint8(RandomAccessibleInterval<UnsignedByteType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<UnsignedByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setByte(i ++, cursor.get().getByte());
		}
    }

    /**
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildInt16(RandomAccessibleInterval<ShortType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<ShortType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setShort((i * Short.BYTES), cursor.get().get());
			i ++;
		}
    }

    /**
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildUint16(RandomAccessibleInterval<UnsignedShortType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<UnsignedShortType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setShort((i * Short.BYTES), cursor.get().getShort());
			i ++;
		}
    }

    /**
     * Adds the IntType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildInt32(RandomAccessibleInterval<IntType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<IntType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setInt((i * Integer.BYTES), cursor.get().get());
			i ++;
		}
    }

    /**
     * Adds the IntType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildUint32(RandomAccessibleInterval<UnsignedIntType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<UnsignedIntType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setInt((i * Integer.BYTES), cursor.get().getInt());
			i ++;
		}
    }

    /**
     * Adds the IntType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildInt64(RandomAccessibleInterval<LongType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<LongType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setLong((i * Long.BYTES), cursor.get().get());
			i ++;
		}
    }

    /**
     * Adds the FloatType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildFloat32(RandomAccessibleInterval<FloatType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<FloatType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setFloat((i * Float.BYTES), cursor.get().get());
			i ++;
		}
    }

    /**
     * Adds the DoubleType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private void buildFloat64(RandomAccessibleInterval<DoubleType> tensor)
    {
		tensor = Utils.transpose(tensor);
		Cursor<DoubleType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setDouble((i * Double.BYTES), cursor.get().get());
			i ++;
		}
    }

	@Override
	/**
	 * Unmap and close the shared memory. Necessary to eliminate the shared memory block
	 */
	public void close() {
		if (unlinked) return;
        Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
        Kernel32.INSTANCE.CloseHandle(hMapFile);
        unlinked = true;
	}
	
	public static void main(String[] args) {
	    String memoryName = "Local\\wnsm_52f561c9";
	    WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping(
	            WinNT.FILE_MAP_READ, false, memoryName
	    );
	    if (hMapFile == null) {
	        throw new RuntimeException("OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
	    }

	    // Map the shared memory object into the current process's address space
	    Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(
	            hMapFile, WinNT.FILE_MAP_READ, 0, 0, 0
	    );
	    if (pSharedMemory == null) {
	        Kernel32.INSTANCE.CloseHandle(hMapFile);
	        throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
	    }

	    Kernel32.MEMORY_BASIC_INFORMATION mbi = new Kernel32.MEMORY_BASIC_INFORMATION();
	    if (Kernel32.INSTANCE.VirtualQueryEx(Kernel32.INSTANCE.GetCurrentProcess(), pSharedMemory, mbi, new BaseTSD.SIZE_T((long) mbi.size())).intValue() != 0) {
	        System.out.println("Shared Memory Size: " + mbi.regionSize + " bytes");
	    } else {
	        System.err.println("Unable to query memory region.");
	    }

	    Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
	    Kernel32.INSTANCE.CloseHandle(hMapFile);
	}

	
	// TODO support boolean
	protected static HashMap<String, Object> buildMapFromNumpyLikeSHMA(String memoryName) {
		if (!memoryName.startsWith("Local\\"))
			memoryName = "Local\\" + memoryName;
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping( WinNT.FILE_MAP_READ, false, memoryName);
        if (hMapFile == null) {
            throw new RuntimeException("OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        // Map the shared memory object into the current process's address space
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(hMapFile, WinNT.FILE_MAP_READ, 0, 0, 0);
        if (pSharedMemory == null) {
        	Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        Kernel32.MEMORY_BASIC_INFORMATION mbi = new Kernel32.MEMORY_BASIC_INFORMATION();
        
        if (Kernel32.INSTANCE.VirtualQueryEx(
        		Kernel32.INSTANCE.GetCurrentProcess(), pSharedMemory, mbi, new BaseTSD.SIZE_T((long) mbi.size())
        		).intValue() == 0) {
            throw new RuntimeException("Unable to retrieve the size of the shm segment located at '" 
        		+ memoryName + "'. Errno: " + Kernel32.INSTANCE.GetLastError());
        }
        int size = mbi.regionSize.intValue();
        byte[] flat = new byte[(int) size];
		for (int i = 0; i < size; i++)
			flat[i] = pSharedMemory.getByte((long) i);
        try (ByteArrayInputStream bis = new ByteArrayInputStream(flat)) {
			HashMap<String, Object> map = DecodeNumpy.decodeNumpyFromByteArrayStreamToRawMap(bis);
        	Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
        	return map;
        } catch (Exception ex) {
            Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
        	throw new RuntimeException(ex);
        }
	}

	
	// TODO support boolean
	protected static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromNumpyLikeSHMA(String memoryName) {
		if (!memoryName.startsWith("Local\\"))
			memoryName = "Local\\" + memoryName;
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping( WinNT.FILE_MAP_READ, false, memoryName);
        if (hMapFile == null) {
            throw new RuntimeException("OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        // Map the shared memory object into the current process's address space
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(hMapFile, WinNT.FILE_MAP_READ, 0, 0, 0);
        if (pSharedMemory == null) {
        	Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        Kernel32.MEMORY_BASIC_INFORMATION mbi = new Kernel32.MEMORY_BASIC_INFORMATION();
        
        if (Kernel32.INSTANCE.VirtualQueryEx(
        		Kernel32.INSTANCE.GetCurrentProcess(), pSharedMemory, mbi, new BaseTSD.SIZE_T((long) mbi.size())
        		).intValue() == 0) {
            throw new RuntimeException("Unable to retrieve the size of the shm segment located at '" 
        		+ memoryName + "'. Errno: " + Kernel32.INSTANCE.GetLastError());
        }
        int size = mbi.regionSize.intValue();
        byte[] flat = new byte[(int) size];
		for (int i = 0; i < size; i++)
			flat[i] = pSharedMemory.getByte((long) i);
        try (ByteArrayInputStream bis = new ByteArrayInputStream(flat)) {
        	RandomAccessibleInterval<T> rai = DecodeNumpy.decodeNumpyFromByteArrayStream(bis);
        	Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
        	return rai;
        } catch (Exception ex) {
            Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
        	throw new RuntimeException(ex);
        }
	}
	
	// TODO support boolean
	protected static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> createImgLib2RaiFromSharedMemoryBlock(String memoryName, long[] shape, boolean isFortran, String dataType) {
		int size = getArrayByteSize(shape, Cast.unchecked(CommonUtils.getImgLib2DataType(dataType)));
		if (!memoryName.startsWith("Local\\"))
			memoryName = "Local\\" + memoryName;
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping(
                WinNT.FILE_MAP_READ | WinNT.FILE_MAP_WRITE,
                false,
                memoryName
        );
        if (hMapFile == null) {
            throw new RuntimeException("OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
        }

        // Map the shared memory object into the current process's address space
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(
                hMapFile,
                WinNT.FILE_MAP_READ | WinNT.FILE_MAP_WRITE,
                0,
                0,
                size
        );
        if (pSharedMemory == null) {
        	Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        try {
        	RandomAccessibleInterval<T> rai = buildFromSharedMemoryBlock(pSharedMemory, shape, isFortran, dataType);
        	Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
        	return rai;
        } catch (Exception ex) {
            Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
        	throw new RuntimeException(ex);
        }
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildFromSharedMemoryBlock(Pointer pSharedMemory, long[] shape, boolean isFortran, String type) {
		T dataType = CommonUtils.getImgLib2DataType(type);
		long[] transposedShape = new long[shape.length];
		for (int i = 0; i < shape.length; i ++) {transposedShape[i] = shape[shape.length - i - 1];}
		if (dataType instanceof ByteType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			byte[] flat = new byte[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getByte((long) i);
			return Cast.unchecked(Utils.transpose(ArrayImgs.bytes(flat, transposedShape)));
		} else if (dataType instanceof ByteType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			byte[] flat = new byte[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getByte((long) i);
			return Cast.unchecked(ArrayImgs.bytes(flat, shape));
		} else if (dataType instanceof UnsignedByteType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			byte[] flat = new byte[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getByte((long) i);
			return Cast.unchecked(ArrayImgs.unsignedBytes(flat, shape));
		} else if (dataType instanceof UnsignedByteType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			byte[] flat = new byte[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getByte((long) i);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedBytes(flat, transposedShape)));
		} else if (dataType instanceof ShortType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			short[] flat = new short[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getShort((long) i * Short.BYTES);
			return Cast.unchecked(ArrayImgs.shorts(flat, shape));
		} else if (dataType instanceof ShortType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			short[] flat = new short[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getShort((long) i * Short.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.shorts(flat, transposedShape)));
		} else if (dataType instanceof UnsignedShortType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			short[] flat = new short[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getShort((long) i * Short.BYTES);
			return Cast.unchecked(ArrayImgs.unsignedShorts(flat, shape));
			
		} else if (dataType instanceof UnsignedShortType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			short[] flat = new short[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getShort((long) i * Short.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedShorts(flat, transposedShape)));
			
		} else if (dataType instanceof IntType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			int[] flat = new int[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getInt((long) i * Integer.BYTES);
			return Cast.unchecked(ArrayImgs.ints(flat, shape));
		} else if (dataType instanceof IntType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			int[] flat = new int[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getInt((long) i * Integer.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.ints(flat, transposedShape)));
		} else if (dataType instanceof UnsignedIntType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			int[] flat = new int[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getInt((long) i * Integer.BYTES);
			return Cast.unchecked(ArrayImgs.unsignedInts(flat, shape));
		} else if (dataType instanceof UnsignedIntType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			int[] flat = new int[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getInt((long) i * Integer.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedInts(flat, transposedShape)));
		} else if (dataType instanceof LongType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			long[] flat = new long[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getLong((long) i * Long.BYTES);
			return Cast.unchecked(ArrayImgs.longs(flat, shape));
		} else if (dataType instanceof LongType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			long[] flat = new long[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getLong((long) i * Long.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.longs(flat, transposedShape)));
		} else if (dataType instanceof FloatType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			float[] flat = new float[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getFloat((long) i * Float.BYTES);
			return Cast.unchecked(ArrayImgs.floats(flat, shape));
		} else if (dataType instanceof FloatType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			float[] flat = new float[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getFloat((long) i * Float.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.floats(flat, transposedShape)));
		} else if (dataType instanceof DoubleType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			double[] flat = new double[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getDouble((long) i * Double.BYTES);
			return Cast.unchecked(ArrayImgs.doubles(flat, shape));
		} else if (dataType instanceof DoubleType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
			double[] flat = new double[arrSize]; 
			for (int i = 0; i < arrSize; i++)
				flat[i] = pSharedMemory.getDouble((long) i * Double.BYTES);
			return Cast.unchecked(Utils.transpose(ArrayImgs.doubles(flat, transposedShape)));
		} else {
    		throw new IllegalArgumentException("Type not supported: " + dataType.getClass().toString());
		}
	}
        
    public static <T extends RealType<T> & NativeType<T>> int getArrayByteSize(long[] shape, T type) {
    	int noByteSize = 1;
    	for (long l : shape) {noByteSize *= l;}
    	if (type instanceof ByteType || type instanceof UnsignedByteType) {
    		return noByteSize * 1;
    	} else if (type instanceof ShortType || type instanceof UnsignedShortType) {
    		return noByteSize * 2;
    	} else if (type instanceof IntType || type instanceof UnsignedIntType
    			|| type instanceof FloatType) {
    		return noByteSize * 4;
    	} else if (type instanceof LongType || type instanceof DoubleType) {
    		return noByteSize * 8;
    	} else {
    		throw new IllegalArgumentException("Type not supported: " + type.getClass().toString());
    	}
	}

	@Override
	public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI() {
		return buildFromSharedMemoryBlock(pSharedMemory, this.originalDims, false, this.originalDataType);
	}

	@Override
	public String getOriginalDataType() {
		return this.originalDataType;
	}

	@Override
	public long[] getOriginalShape() {
		return this.originalDims;
	}
	
	@Override
	public boolean isNumpyFormat() {
		return this.isNumpyFormat;
	}
}
