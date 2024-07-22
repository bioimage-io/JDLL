/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.util.Arrays;
import java.util.regex.Matcher;

import com.sun.jna.Pointer;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;

import com.sun.jna.Native;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.ByteAccess;
import net.imglib2.img.basictypeaccess.DoubleAccess;
import net.imglib2.img.basictypeaccess.FloatAccess;
import net.imglib2.img.basictypeaccess.IntAccess;
import net.imglib2.img.basictypeaccess.LongAccess;
import net.imglib2.img.basictypeaccess.ShortAccess;
import net.imglib2.img.basictypeaccess.nio.ByteBufferAccess;
import net.imglib2.img.basictypeaccess.nio.DoubleBufferAccess;
import net.imglib2.img.basictypeaccess.nio.FloatBufferAccess;
import net.imglib2.img.basictypeaccess.nio.IntBufferAccess;
import net.imglib2.img.basictypeaccess.nio.LongBufferAccess;
import net.imglib2.img.basictypeaccess.nio.ShortBufferAccess;
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
 * Class that maps {@link RandomAccessibleInterval} objects to the shared memory for interprocessing communication
 * in MacOS based systems
 * @author Carlos Garcia Lopez de Haro
 */
public class SharedMemoryArrayMacOS implements SharedMemoryArray
{
	/**
	 * Instance of the CLibrary JNI containing the methods to interact with the Shared memory segments
	 */
	private static final CLibrary INSTANCE = CLibrary.INSTANCE;
	/**
	 * Instance of the  JNI containing the methods that help ineracting with shared memory segments
	 * and are not contained in the {@link CLibrary} {@value #INSTANCE}
	 */
	private static final MacosHelpers MACOS_INSTANCE = MacosHelpers.INSTANCE;
	/**
	 * File descriptor value of the shared memory segment
	 */
	private int shmFd;
	/**
	 * Pointer referencing the shared memory byte array
	 */
	private Pointer pSharedMemory;
	/**
	 * Name of the file containing the shared memory segment. In Unix based systems consits of "/" + file_name.
	 * In Linux the shared memory segments can be inspected at /dev/shm.
	 * For MacOS the name can only have a certain length, {@value #MACOS_MAX_LENGTH}
	 */
	private final String memoryName;
	/**
	 * Size of the shared memory block
	 */
	private int size;
	/**
	 * Shared memory segments store bytes. This field represents the original data type of the array that was written
	 * into the bytes of the shared memory segment. It is helful to retrieve the object later.
	 */
	private String originalDataType;
	/**
	 * Shared memory segments are flat arrays, only one dimension. This field keeps the dimensions of the array before
	 * flattening it and copying it to the shared memory.
	 */
	private long[] originalDims;
	/**
	 * Whether the memory block has been closed and unlinked
	 */
	private boolean unlinked = false;
	/**
	 * Whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
	 */
	private Boolean isNumpyFormat = null;
	/**
	 * This parameter makes sense for nd-arrays. Whether the n-dimensional array is flattened followin
	 * fortran order or not (c order)
	 */
	private boolean isFortran = false;
	/**
	 * Maximum length of the name that can be given to a shared memory region
	 */
    protected static final int MACOS_MAX_LENGTH = 30;
	
	/**
	 * This method creates a shared memory segment with the wanted name. The byte size is defined by the
	 * 'size' argument, but it has to be coherent with 'shape', 'dtype' and 'isNumpy' arguments.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but the size required by the
	 * shape and data type is not the same as the size of the existing shared memory segment, an exception will
	 * be thrown.
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 		
	 * 		SharedMemoryArrayMacOS("shm_example", 4096, "float32", new long[]{1024}, false, false);
	 * 
	 * An exception will be thrown because the required number of bytes is 1024 * 4 (4 bytes per float) = 4096 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param size
	 * 	the byte size of the shared memory segment
	 * @param dtype
	 * 	the data type of the nd array that is written from the shared memory segment
	 * @param shape
	 * 	the dimensions of the nd array that is written from the shared memory segment
	 * @param isNumpy
	 * 	whether an nd array is saved to the shared memory segment in Numpy npy format, that is with a header at the 
	 * 	beginning that increases the byte size
	 * @param isFortran
	 * 	whether nd arrays are stored with fortran order or not (c-order)
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the specified shape and datatype
	 */
    protected SharedMemoryArrayMacOS(int size, String dtype, long[] shape, Boolean isNumpy, boolean isFortran) throws FileAlreadyExistsException
    {
    	this(SharedMemoryArray.createShmName(), size, dtype, shape, isNumpy, isFortran);
    }
    
	/**
	 * This method creates (or retrieves if it already exists) a shared memory segment with the wanted name. The byte size is defined by the
	 * 'size' argument, but it has to be coherent with 'shape', 'dtype' and 'isNumpy' arguments.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but the size required by the
	 * shape and data type is not the same as the size of the existing shared memory segment, an exception will
	 * be thrown.
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 		
	 * 		SharedMemoryArrayMacOS("shm_example", 4096, "float32", new long[]{1024}, false, false);
	 * 
	 * An exception will be thrown because the required number of bytes is 1024 * 4 (4 bytes per float) = 4096 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param name
	 * 	name of the file name that is going to be used to identify the shared memory segment
	 * @param size
	 * 	the byte size of the shared memory segment
	 * @param dtype
	 * 	the data type of the nd array that is written from the shared memory segment
	 * @param shape
	 * 	the dimensions of the nd array that is written from the shared memory segment
	 * @param isNumpy
	 * 	whether an nd array is saved to the shared memory segment in Numpy npy format, that is with a header at the 
	 * 	beginning that increases the byte size
	 * @param isFortran
	 * 	whether nd arrays are stored with fortran order or not (c-order)
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the specified shape and datatype
	 */
    protected SharedMemoryArrayMacOS(String name, int size, String dtype, long[] shape, Boolean isNumpy, boolean isFortran) throws FileAlreadyExistsException
    {
    	this.originalDataType = dtype;
    	this.originalDims = shape;
    	this.size = size;
    	this.memoryName = name;
    	this.isNumpyFormat = isNumpy;
    	this.isFortran = isFortran;

    	boolean alreadyExists = false;
    	shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
    	long prevSize = 0;
    	if (shmFd != -1) {
        	prevSize = getSHMSize(shmFd);
    		alreadyExists = true;
    	}

		if (alreadyExists && prevSize < size) {
    		throw new FileAlreadyExistsException("Shared memory segment already exists with smaller dimensions, data type or format. "
    				+ "Size of the existing shared memory segment cannot be smaller than the size of the proposed object. "
    				+ "Size of existing shared memory segment: " + prevSize + ", size of proposed object: " + size);
    	}
        shmFd = MACOS_INSTANCE.create_shared_memory(memoryName, size);
        if (shmFd < 0) {
            throw new RuntimeException("shm_open failed, errno: " + Native.getLastError());
        }
        /*
        shmFd = INSTANCE.shm_open(this.memoryName, O_RDWR, 0666);
        if (shmFd < 0) {
            throw new RuntimeException("shm_open failed, errno: " + Native.getLastError());
        }*/

        pSharedMemory = INSTANCE.mmap(Pointer.NULL, this.size, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        if (pSharedMemory == Pointer.NULL) {
        	INSTANCE.close(shmFd);
            throw new RuntimeException("mmap failed, errno: " + Native.getLastError());
        }
    }
    
    /**
     * MEthod to find the size of an already created shared memory segment
     * @param memoryName
     * 	the name of the shared memory segment
     * @return the size in bytes of the shared memory segment
     */
    protected static long getSHMSize(String memoryName) {
    	SharedMemoryArray.checkMemorySegmentName(memoryName);
    	if (!memoryName.startsWith("/"))
    		memoryName = "/" + memoryName;
    	int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
        if (shmFd < 0) throw new RuntimeException("Failed to open shared memory, it might not exist. Errno: " + Native.getLastError());

	    return getSHMSize(shmFd);
    }
    
    /**
     * MEthod to find the size of an already created shared memory segment
     * @param shmFd
     * 	the shared memory segment identifier
     * @return the size in bytes of the shared memory segment
     */
    protected static long getSHMSize(int shmFd) {
        if (shmFd < 0) throw new RuntimeException("Invalid shmFd. It should be bigger than 0.");


        long size = MACOS_INSTANCE.get_shared_memory_size(shmFd);
	    if (size == -1) {
	    	// TODO remove macosInstance.unlink_shared_memory(null);;
	    	throw new RuntimeException("Failed to get shared memory segment size. Errno: " + Native.getLastError());
	    }
	    return size;
    }

    /**
     * Private constructor to create an instance for the specific case when it is wrapping an ImgLib2
     * {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory segment
     */
	private SharedMemoryArrayMacOS(String name) {
		this.memoryName = name;
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayMacOS readOrCreate(String name, int size, long[] shape, String strDType, Boolean isNumpy, boolean isFortran) throws FileAlreadyExistsException {
		return new SharedMemoryArrayMacOS(name, size, strDType, shape, isNumpy, isFortran);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayMacOS readOrCreate(String name, int size) throws FileAlreadyExistsException {
		return new SharedMemoryArrayMacOS(name, size, null, null, null, false);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayMacOS create(int size, long[] shape, String strDType, Boolean isNumpy, boolean isFortran) {
		try {
			return new SharedMemoryArrayMacOS(size, strDType, shape, isNumpy, isFortran);
		} catch (FileAlreadyExistsException e) {
			throw new RuntimeException("Unexpected error.", e);
		}
	}

	protected static SharedMemoryArrayMacOS create(int size) {
		try {
			return new SharedMemoryArrayMacOS(size, null, null, null, false);
		} catch (FileAlreadyExistsException e) {
			throw new RuntimeException("Unexpected error.", e);
		}
	}
	
    protected static <T extends RealType<T> & NativeType<T>> 
    SharedMemoryArrayMacOS createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) throws FileAlreadyExistsException
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayMacOS shma = null;
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt8(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildUint8(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt16(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildUint16(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildUint32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt64(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildFloat32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayMacOS(name, size, CommonUtils.getDataTypeFromRAI(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildFloat64(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
		return shma;
    }
    
    /**
     * Retrive an existing Shared memory segment and wrap it into a {@link SharedMemoryArray} 
     * @param memoryName
     * 	the name of the segment
     * @return the {@link SharedMemoryArray}  pointing to the shared memory segment
     */
    protected static SharedMemoryArrayMacOS read(String memoryName) {
    	if (!memoryName.startsWith("/")) memoryName = "/" + memoryName;
	    int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
        if (shmFd < 0) 
            throw new RuntimeException("Shared memory segmentmight not exist: "
					+ memoryName + ". Failed to open shared memory. Errno: " + Native.getLastError());

        long size = MACOS_INSTANCE.get_shared_memory_size(shmFd);;
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
        //pSharedMemory.read(0, flat, 0, flat.length);
        SharedMemoryArrayMacOS shm = new SharedMemoryArrayMacOS(memoryName);
        shm.shmFd = shmFd;
        shm.pSharedMemory = pSharedMemory;
        shm.size = (int) size;
        shm.findNumpyFormat();
        return shm;
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

    private void buildInt8(RandomAccessibleInterval<ByteType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<ByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setByte(offset + (i ++), cursor.get().get());
		}
    }

    private void buildUint8(RandomAccessibleInterval<UnsignedByteType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<UnsignedByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setByte(offset + (i ++), cursor.get().getByte());
		}
    }

    private void buildInt16(RandomAccessibleInterval<ShortType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<ShortType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setShort(offset + (i * Short.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildUint16(RandomAccessibleInterval<UnsignedShortType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<UnsignedShortType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setShort(offset + (i * Short.BYTES), cursor.get().getShort());
			i ++;
		}
    }

    private void buildInt32(RandomAccessibleInterval<IntType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<IntType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setInt(offset + (i * Integer.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildUint32(RandomAccessibleInterval<UnsignedIntType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<UnsignedIntType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setInt(offset + (i * Integer.BYTES), cursor.get().getInt());
			i ++;
		}
    }

    private void buildInt64(RandomAccessibleInterval<LongType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<LongType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setLong(offset + (i * Long.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildFloat32(RandomAccessibleInterval<FloatType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<FloatType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setFloat(offset + (i * Float.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildFloat64(RandomAccessibleInterval<DoubleType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<DoubleType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setDouble(offset + (i * Double.BYTES), cursor.get().get());
			i ++;
		}
    }
    
    private static <T extends RealType<T> & NativeType<T>>
    byte[] getNpyHeader(RandomAccessibleInterval<T> tensor) {
    	String strHeader = "{'descr': '<";
    	strHeader += DecodeNumpy.getDataType(tensor.getAt(tensor.minAsLongArray()));
    	strHeader += "', 'fortran_order': False, 'shape': (";
    	for (long ll : tensor.dimensionsAsLongArray()) strHeader += ll + ", ";
    	strHeader = strHeader.substring(0, strHeader.length() - 2);
    	strHeader += "), }" + System.lineSeparator();
    	byte[] bufInverse = strHeader.getBytes(StandardCharsets.UTF_8);
    	byte[] major = {1};
        byte[] minor = {0};
        byte[] len = new byte[2];
        len[0] = (byte) (short) strHeader.length();
        len[1] = (byte) (((short) strHeader.length()) >> 8);
        int totalLen = DecodeNumpy.NUMPY_PREFIX.length + 2 + 2 + bufInverse.length;
        byte[] total = new byte[totalLen];
        int c = 0;
        for (int i = 0; i < DecodeNumpy.NUMPY_PREFIX.length; i ++)
        	total[c ++] = DecodeNumpy.NUMPY_PREFIX[i];
        total[c ++] = major[0];
        total[c ++] = minor[0];
        total[c ++] = len[0];
        total[c ++] = len[1];
        for (int i = 0; i < bufInverse.length; i ++)
        	total[c ++] = bufInverse[i];
        return total;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public String getName() {
    	return this.memoryName;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public String getNameForPython() {
    	return this.memoryName.substring("/".length());
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public Pointer getPointer() {
    	return this.pSharedMemory;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public Object getSharedMemoryBlockID() {
    	return this.shmFd;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public int getSize() {
    	return this.size;
    }

	@Override
    /**
     * {@inheritDoc}
     */
	public String getOriginalDataType() {
		return this.originalDataType;
	}

	@Override
    /**
     * {@inheritDoc}
     */
	public long[] getOriginalShape() {
		return this.originalDims;
	}
	
	@Override
    /**
     * {@inheritDoc}
     */
	public boolean isNumpyFormat() {
		if (this.isNumpyFormat == null) {
			findNumpyFormat();
		}
		return this.isNumpyFormat;
	}
	
	/**
	 * Find whether the shared memory segment wrapped by the {@link SharedMemoryArray} is in Numpy npy format of not.
	 */
	private void findNumpyFormat() {
		this.isNumpyFormat = true;
		try {
			int offset = 0;
			byte[] buf = new byte[DecodeNumpy.NUMPY_PREFIX.length];
			pSharedMemory.getByteBuffer(offset, DecodeNumpy.NUMPY_PREFIX.length).get(buf, 0, DecodeNumpy.NUMPY_PREFIX.length);
	        if (!Arrays.equals(buf, DecodeNumpy.NUMPY_PREFIX)) {
	            throw new IllegalArgumentException("Malformed  or unsopported Numpy array");
	        }
	        offset = DecodeNumpy.NUMPY_PREFIX.length;
	        byte major = pSharedMemory.getByteBuffer(offset, 1).get();
	        offset ++;
	        byte minor = pSharedMemory.getByteBuffer(offset, 1).get();
	        offset ++;
	        if (major < 1 || major > 3 || minor != 0) {
	            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
	        }
	        int len = major == 1 ? 2 : 4;
	        ByteBuffer bb = pSharedMemory.getByteBuffer(offset, len);
	        offset += len;
	        bb.order(ByteOrder.LITTLE_ENDIAN);
	        if (major == 1) {
	            len = bb.getShort();
	        } else {
	            len = bb.getInt();
	        }
	        buf = new byte[len];
	        pSharedMemory.getByteBuffer(offset, len).get(buf, 0, len);
	        offset += len;
	        String header = new String(buf, StandardCharsets.UTF_8);
	        Matcher m = DecodeNumpy.HEADER_PATTERN.matcher(header);
	        if (!m.find()) {
	            throw new IllegalArgumentException("Invalid numpy header: " + header);
	        }
	        String typeStr = m.group(1);
	        String fortranOrder = m.group(2).trim();
	        String shapeStr = m.group(3);
	        long[] shape = new long[0];
	        if (!shapeStr.isEmpty()) {
	            String[] tokens = shapeStr.split(", ?");
	            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
	        }
	        char order = typeStr.charAt(0);
	    	if (order != '>' && order != '<' && order != '|') {
	        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
	        }
	        String dtype = DecodeNumpy.getDataType(typeStr.substring(1));
	        this.originalDims = shape;
	        this.originalDataType = dtype;
	        this.isFortran = fortranOrder.equals("True");
		} catch (Exception ex) {
			this.isNumpyFormat = false;
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
    
    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI() {
    	if ((this.originalDims == null || this.originalDataType == null) && !this.isNumpyFormat())
    		throw new IllegalArgumentException("The shared memory segment is not stored in Numpy format and the shape and/or "
    				+ "data type are not known. Please provide information about them and use the method "
    				+ "'getSharedRAI(long[] shape, boolean isFortran, T dataType)'.");
    	if (this.isNumpyFormat()) {
    		return buildImgLib2FromNumpyLikeSHMA();
    	} else {
    		return buildFromSharedMemoryBlock(pSharedMemory, this.originalDims, this.originalDataType, this.isFortran);
    	}
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType) {
		return buildFromSharedMemoryBlock(pSharedMemory, shape, dataType, isFortran, 0, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType, boolean isFortran) {
		return buildFromSharedMemoryBlock(pSharedMemory, shape, dataType, isFortran, 0, ByteOrder.LITTLE_ENDIAN);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void setBuffer(ByteBuffer buffer) {
    	if (buffer.capacity() > this.size) {
    		throw new IllegalArgumentException("The buffer capacity has to be smaller or equal "
    				+ "than the size of the shared memory segment.");
    	}
    	byte[] array;
    	if (buffer.hasArray()) {
    	    array = buffer.array();
    	} else {
    	    array = new byte[buffer.remaining()];
    	    buffer.get(array);
    	}
    	this.pSharedMemory.write(0, buffer.array(), 0, buffer.capacity());
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public ByteBuffer getDataBuffer() {
    	return pSharedMemory.getByteBuffer(0, this.size);
    }

	@Override
	/**
	 * {@inheritDoc}
	 */
	public ByteBuffer getDataBufferNoHeader() {
    	int offset = 0;
    	if (this.isNumpyFormat()) {
    		long flatSize = 1;
    		for (long l : this.originalDims) flatSize *= l;
    		offset =  (int) (this.size - DecodeNumpy.DATA_TYPES_MAP.get(this.originalDataType) * flatSize);
    	}
    	return pSharedMemory.getByteBuffer(offset, this.size - offset);
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildFromSharedMemoryBlock(Pointer pSharedMemory, long[] shape, String type, boolean isFortran) {
		T dataType = CommonUtils.getImgLib2DataType(type);
		return buildFromSharedMemoryBlock(pSharedMemory, shape, dataType, isFortran, 0, ByteOrder.LITTLE_ENDIAN);
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildFromSharedMemoryBlock(Pointer pSharedMemory, long[] shape, T dataType, 
			boolean isFortran, int offset, ByteOrder order) {
		long[] transposedShape = new long[shape.length];
		for (int i = 0; i < shape.length; i ++) {transposedShape[i] = shape[shape.length - i - 1];}
		if (dataType instanceof ByteType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.bytes(access, transposedShape)));
		} else if (dataType instanceof ByteType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.bytes(access, shape));
		} else if (dataType instanceof UnsignedByteType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.unsignedBytes(access, shape));
		} else if (dataType instanceof UnsignedByteType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedBytes(access, transposedShape)));
		} else if (dataType instanceof ShortType && isFortran) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.shorts(access, shape));
		} else if (dataType instanceof ShortType) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.shorts(access, transposedShape)));
		} else if (dataType instanceof UnsignedShortType && isFortran) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.unsignedShorts(access, shape));
			
		} else if (dataType instanceof UnsignedShortType) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedShorts(access, transposedShape)));
			
		} else if (dataType instanceof IntType && isFortran) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.ints(access, shape));
		} else if (dataType instanceof IntType) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.ints(access, transposedShape)));
		} else if (dataType instanceof UnsignedIntType && isFortran) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.unsignedInts(access, shape));
		} else if (dataType instanceof UnsignedIntType) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedInts(access, transposedShape)));
		} else if (dataType instanceof LongType && isFortran) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			LongAccess access = new LongBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.longs(access, shape));
		} else if (dataType instanceof LongType) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			LongAccess access = new LongBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.longs(access, transposedShape)));
		} else if (dataType instanceof FloatType && isFortran) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			FloatAccess access = new FloatBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.floats(access, shape));
		} else if (dataType instanceof FloatType) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			FloatAccess access = new FloatBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.floats(access, transposedShape)));
		} else if (dataType instanceof DoubleType && isFortran) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			DoubleAccess access = new DoubleBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.doubles(access, shape));
		} else if (dataType instanceof DoubleType) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			DoubleAccess access = new DoubleBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.doubles(access, transposedShape)));
		} else {
    		throw new IllegalArgumentException("Type not supported: " + dataType.getClass().toString());
		}
	}
	
	private <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromNumpyLikeSHMA() {
		int offset = 0;
		byte[] buf = new byte[DecodeNumpy.NUMPY_PREFIX.length];
		pSharedMemory.getByteBuffer(offset, DecodeNumpy.NUMPY_PREFIX.length).get(buf, 0, DecodeNumpy.NUMPY_PREFIX.length);
        if (!Arrays.equals(buf, DecodeNumpy.NUMPY_PREFIX)) {
            throw new IllegalArgumentException("Malformed  or unsopported Numpy array");
        }
        offset = DecodeNumpy.NUMPY_PREFIX.length;
        byte major = pSharedMemory.getByteBuffer(offset, 1).get();
        offset ++;
        byte minor = pSharedMemory.getByteBuffer(offset, 1).get();
        offset ++;
        if (major < 1 || major > 3 || minor != 0) {
            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
        }
        int len = major == 1 ? 2 : 4;
        ByteBuffer bb = pSharedMemory.getByteBuffer(offset, len);
        offset += len;
        bb.order(ByteOrder.LITTLE_ENDIAN);
        if (major == 1) {
            len = bb.getShort();
        } else {
            len = bb.getInt();
        }
        buf = new byte[len];
        pSharedMemory.getByteBuffer(offset, len).get(buf, 0, len);
        offset += len;
        String header = new String(buf, StandardCharsets.UTF_8);
        Matcher m = DecodeNumpy.HEADER_PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        String fortranOrder = m.group(2).trim();
        String shapeStr = m.group(3);
        long[] shape = new long[0];
        if (!shapeStr.isEmpty()) {
            String[] tokens = shapeStr.split(", ?");
            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
        }
        char order = typeStr.charAt(0);
        ByteOrder byteOrder = null;
        if (order == '>') {
        	byteOrder = ByteOrder.BIG_ENDIAN;
        } else if (order == '<') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        } else if (order == '|') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        	new IOException("Numpy .npy file did not specify the byte order of the array."
        			+ " It was automatically opened as little endian but this does not guarantee"
        			+ " the that the file is open correctly. Caution is advised.").printStackTrace();
    	} else {
        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
        }
        String dtype = DecodeNumpy.getDataType(typeStr.substring(1));
        long numBytes = DecodeNumpy.DATA_TYPES_MAP.get(dtype);
    	long count;
    	if (shape.length == 0)
    		count = 1;
		else
			count = Arrays.stream(shape).reduce(Math::multiplyExact).getAsLong();
        len = Math.toIntExact(count * numBytes);
        
        return buildFromSharedMemoryBlock(this.pSharedMemory, shape, 
        		Cast.unchecked(CommonUtils.getImgLib2DataType(dtype)), fortranOrder.equals("True"), offset, byteOrder);
	}
}
