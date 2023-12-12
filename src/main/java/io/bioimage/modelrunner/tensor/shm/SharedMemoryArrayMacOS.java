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

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;

import com.sun.jna.Native;

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
 * in POSIX based systems (LINUX, MACOS...)
 * @author Carlos Garcia Lopez de Haro
 */
public class SharedMemoryArrayMacOS implements SharedMemoryArray
{
	private static final CLibrary INSTANCE = CLibrary.INSTANCE;
	private static final MacosHelpers macosInstance = MacosHelpers.INSTANCE;
	/**
	 * Shared memory location
	 */
	private final int shmFd;
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
	 * Whether the memory block has been closed and unlinked
	 */
	private boolean unlinked = false;
	/**
	 * Whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
	 */
	private boolean isNumpyFormat = false;

    public static final int O_RDONLY = 0;
    private static final int O_RDWR = 2;    // Read-write mode
    private static final int PROT_READ = 0x1;  // Page can be read
    private static final int PROT_WRITE = 0x2; // Page can be written
    private static final int MAP_SHARED = 0x01; // Share changes
    
    protected static final int MACOS_MAX_LENGTH = 30;
    
    private SharedMemoryArrayMacOS(int size, String dtype, long[] shape)
    {
    	this(("/shm-" + UUID.randomUUID()).substring(0, MACOS_MAX_LENGTH), size, dtype, shape);
    }
    
    private SharedMemoryArrayMacOS(String name, int size, String dtype, long[] shape)
    {
    	this.originalDataType = dtype;
    	this.originalDims = shape;
    	this.size = size;
    	this.memoryName = name;

        int openFd = macosInstance.create_shared_memory(memoryName, size);
        if (openFd < 0) {
            throw new RuntimeException("shm_open failed, errno: " + Native.getLastError());
        }
        shmFd = INSTANCE.shm_open(this.memoryName, O_RDWR, 0666);
        if (shmFd < 0) {
            throw new RuntimeException("shm_open failed, errno: " + Native.getLastError());
        }

        pSharedMemory = INSTANCE.mmap(Pointer.NULL, this.size, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        if (pSharedMemory == Pointer.NULL) {
        	INSTANCE.close(shmFd);
            throw new RuntimeException("mmap failed, errno: " + Native.getLastError());
        }
    }
    
    public String getName() {
    	return this.memoryName;
    }
    
    public String getNameForPython() {
    	return this.memoryName.substring("/".length());
    }
    
    public Pointer getPointer() {
    	return this.pSharedMemory;
    }
    
    public int getSharedMemoryBlock() {
    	return this.shmFd;
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
     * @return an instance of {@link SharedMemoryArrayMacOS} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayMacOS build(RandomAccessibleInterval<T> rai)
    {
    	return build(("/shm-" + UUID.randomUUID()).substring(0, MACOS_MAX_LENGTH), rai);
    }

    /**
     * Adds the {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param name
     * 	name of the memory location where the shm segment is going to be created, cannot contain any special character
     *  and should start by "/" in Unix based systems. The shm name is generated 
     *  automatically with the the method {@link #build(String, RandomAccessibleInterval)}. In Macos, names should not be 
     *  longer than 30 characters
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer

     * @return an instance of {@link SharedMemoryArrayMacOS} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayMacOS build(String name, RandomAccessibleInterval<T> rai)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayMacOS shma = null;
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt64(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildFloat32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
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
     * and should start by "/" in Unix based systems. The shm name is generated automatically with the the method {@link #build(String, RandomAccessibleInterval)}
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @return an instance of {@link SharedMemoryArrayLinux} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayMacOS buildNumpyFormat(RandomAccessibleInterval<T> rai)
    {
    	return buildNumpyFormat(("/shm-" + UUID.randomUUID()).substring(0, MACOS_MAX_LENGTH), rai);
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
     *  and should start by "/" in Unix based systems. The shm name is generated 
     *  automatically with the the method {@link #build(String, RandomAccessibleInterval)}. In Macos, names should not be 
     *  longer than 30 characters
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @return an instance of {@link SharedMemoryArrayLinux} containing the pointer to the 
     * 	shared memory where the array is, the hMapFile, the size of the object in bytes, and 
     * 	name of the memory location
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayMacOS buildNumpyFormat(String name, RandomAccessibleInterval<T> rai)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayMacOS shma = null;
    	byte[] total = DecodeNumpy.createNumpyStyleByteArray(rai);
    	shma = new SharedMemoryArrayMacOS(name, total.length, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
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
		if (this.unlinked) return;

        // Unmap the shared memory
        if (this.pSharedMemory != Pointer.NULL && INSTANCE.munmap(this.pSharedMemory, size) == -1) {
            throw new RuntimeException("munmap failed. Errno: " + Native.getLastError());
        }

        // Close the file descriptor
        if (INSTANCE.close(this.shmFd) == -1) {
            throw new RuntimeException("close failed. Errno: " + Native.getLastError());
        }

        // Unlink the shared memory object
        INSTANCE.shm_unlink(this.memoryName);
        unlinked = true;
	}
	
	// TODO support boolean
	public static HashMap<String, Object> buildMapFromNumpyLikeSHMA(String memoryName) {
		if (!memoryName.startsWith("/")) memoryName = "/" + memoryName;
	    int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
        if (shmFd < 0) 
            throw new RuntimeException("Failed to open shared memory. Errno: " + Native.getLastError());

        long size = INSTANCE.lseek(shmFd, 0, CLibrary.SEEK_END);
	    if (size == -1) {
            CLibrary.INSTANCE.close(shmFd);
	    	throw new RuntimeException("Failed to get shared memory segment size. Errno: " + Native.getLastError());
	    }	    

        // Map the shared memory into the process's address space
        Pointer pSharedMemory = INSTANCE.mmap(null, (int) size, PROT_READ, MAP_SHARED, shmFd, 0);
        if (pSharedMemory == Pointer.NULL) {
            CLibrary.INSTANCE.close(shmFd);
            throw new RuntimeException("Failed to map shared memory. Errmo: " + Native.getLastError());
        }
        byte[] flat = new byte[(int) size];
		for (int i = 0; i < size; i++)
			flat[i] = pSharedMemory.getByte((long) i);
		try (ByteArrayInputStream bis = new ByteArrayInputStream(flat)){
			HashMap<String, Object> map = DecodeNumpy.decodeNumpyFromByteArrayStreamToRawMap(bis);
        	if (pSharedMemory != Pointer.NULL) {
                INSTANCE.munmap(pSharedMemory, (int) size);
            }
            if (shmFd >= 0) {
            	INSTANCE.close(shmFd);
            }
            INSTANCE.shm_unlink(memoryName);
        	return map;
        } catch (Exception ex) {
            if (pSharedMemory != Pointer.NULL) {
                INSTANCE.munmap(pSharedMemory, (int) size);
            }
            if (shmFd >= 0) {
            	INSTANCE.close(shmFd);
            }
            INSTANCE.shm_unlink(memoryName);
        	throw new RuntimeException(ex);
        }
	}
	
	// TODO support boolean
	public static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromNumpyLikeSHMA(String memoryName) {
		if (!memoryName.startsWith("/")) memoryName = "/" + memoryName;
	    int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
        if (shmFd < 0) 
            throw new RuntimeException("Failed to open shared memory. Errno: " + Native.getLastError());

        long size = INSTANCE.lseek(shmFd, 0, CLibrary.SEEK_END);
	    if (size == -1) {
            CLibrary.INSTANCE.close(shmFd);
	    	throw new RuntimeException("Failed to get shared memory segment size. Errno: " + Native.getLastError());
	    }	    

        // Map the shared memory into the process's address space
        Pointer pSharedMemory = INSTANCE.mmap(null, (int) size, PROT_READ, MAP_SHARED, shmFd, 0);
        if (pSharedMemory == Pointer.NULL) {
            CLibrary.INSTANCE.close(shmFd);
            throw new RuntimeException("Failed to map shared memory. Errmo: " + Native.getLastError());
        }
        byte[] flat = new byte[(int) size];
		for (int i = 0; i < size; i++)
			flat[i] = pSharedMemory.getByte((long) i);
		try (ByteArrayInputStream bis = new ByteArrayInputStream(flat)){
			RandomAccessibleInterval<T> rai = DecodeNumpy.decodeNumpyFromByteArrayStream(bis);
        	if (pSharedMemory != Pointer.NULL) {
                INSTANCE.munmap(pSharedMemory, (int) size);
            }
            if (shmFd >= 0) {
            	INSTANCE.close(shmFd);
            }
            INSTANCE.shm_unlink(memoryName);
        	return rai;
        } catch (Exception ex) {
            if (pSharedMemory != Pointer.NULL) {
                INSTANCE.munmap(pSharedMemory, (int) size);
            }
            if (shmFd >= 0) {
            	INSTANCE.close(shmFd);
            }
            INSTANCE.shm_unlink(memoryName);
        	throw new RuntimeException(ex);
        }
	}
	
	// TODO support boolean
	public static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> createImgLib2RaiFromSharedMemoryBlock(String memoryName, long[] shape, boolean isFortran, String dataType) {
		int size = getArrayByteSize(shape, Cast.unchecked(CommonUtils.getImgLib2DataType(dataType)));
		if (!memoryName.startsWith("/")) memoryName = "/" + memoryName;
		int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0);
        if (shmFd < 0) {
            throw new RuntimeException("Failed to open shared memory. Errno: " + Native.getLastError());
        }

        // Map the shared memory into the process's address space
        Pointer pSharedMemory = INSTANCE.mmap(null, size, PROT_READ, MAP_SHARED, shmFd, 0);
        if (pSharedMemory == Pointer.NULL) {
            CLibrary.INSTANCE.close(shmFd);
            throw new RuntimeException("Failed to map shared memory. Errmo: " + Native.getLastError());
        }
		try {
        	RandomAccessibleInterval<T> rai = buildFromSharedMemoryBlock(pSharedMemory, shape, isFortran, dataType);
        	if (pSharedMemory != Pointer.NULL) {
                INSTANCE.munmap(pSharedMemory, size);
            }
            if (shmFd >= 0) {
            	INSTANCE.close(shmFd);
            }
            INSTANCE.shm_unlink(memoryName);
        	return rai;
        } catch (Exception ex) {
            if (pSharedMemory != Pointer.NULL) {
                INSTANCE.munmap(pSharedMemory, size);
            }
            if (shmFd >= 0) {
            	INSTANCE.close(shmFd);
            }
            INSTANCE.shm_unlink(memoryName);
        	throw ex;
        }
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildFromSharedMemoryBlock(Pointer pSharedMemory, long[] shape, boolean isFortran, String type) {
		long[] transposedShape = new long[shape.length];
		T dataType = CommonUtils.getImgLib2DataType(type);
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
    
    public static void main(String[] args){
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
