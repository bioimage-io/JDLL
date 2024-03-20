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

import java.io.Closeable;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
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
 * Interface to interact with shared memory segments retrieving the underlying information 
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class SharedMemoryArrayLiunxTem implements SharedMemoryArray {
		/**
		 * Instance of the LibRT JNI containing the methods to interact with the Shared memory segments
		 */
		private static final LibRt INSTANCE_RT = LibRt.INSTANCE;
		/**
		 * Instance of the CLibrary JNI containing the methods to interact with the Shared memory segments
		 */
		private static final CLibrary INSTANCE_C = CLibrary.INSTANCE;
		/**
		 * Depending on the computer, some might work with LibRT or LibC to create SHM segments.
		 * Thus if true use librt if false, use libc instance
		 */
		private boolean useLibRT = true;
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
		 * In Linux the shared memory segments can be inspected at /dev/shm
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
		private final String originalDataType;
		/**
		 * Shared memory segments are flat arrays, only one dimension. This field keeps the dimensions of the array before
		 * flattening it and copying it to the shared memory.
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
		/**
		 * This parameter makes sense for nd-arrays. Whether the n-dimensional array is flattened followin
		 * fortran order or not (c order)
		 */
		private boolean isFortran = false;
	

	/**
	 * Create a shared memory segment with the wanted size, where an object of a certain datatype and
	 * share is going to be stored.
	 * Unless the array of bytes that is going to be written into the shared memory segment has numpy format,
	 * the size parameter should only depend on the shape and the data type.
	 * The name of the file containing the shared memory segment is assigned automatically.
	 * @param size
	 * 	number of bytes that are going to be written into the shared memory
	 * @param dtype
	 * 	data type of the object that is going to be written into the shared memory
	 * @param shape
	 * 	shape (array dimensions) of the array that is going to be  flattened and written into the shared memory segment
	 */
    protected SharedMemoryArrayLiunxTem(int size, String dtype, long[] shape, boolean isNumpy, boolean isFortran)
    {
    	this(SharedMemoryArray.createShmName(), size, dtype, shape, isNumpy, isFortran);
    }

	/**
	 * Create a shared memory segment with the wanted size, where an object of a certain datatype and
	 * share is going to be stored. The shared memory name is created in the location of the name provided
	 * Unless the array of bytes that is going to be written into the shared memory segment has numpy format,
	 * the size parameter should only depend on the shape and the data type.
	 * @param name
	 * 	name of the file name that is going to be used to identify the shared memory segment
	 * @param size
	 * 	number of bytes that are going to be written into the shared memory
	 * @param dtype
	 * 	data type of the object that is going to be written into the shared memory
	 * @param shape
	 * 	shape (array dimensions) of the array that is going to be  flattened and written into the shared memory segment
	 */
    protected SharedMemoryArrayLiunxTem(String name, int size, String dtype, long[] shape, boolean isNumpy, boolean isFortran)
    {
    	this.originalDataType = dtype;
    	this.originalDims = shape;
    	this.size = size;
    	this.memoryName = name;
    	this.isNumpyFormat = isNumpy;
    	this.isFortran = isFortran;
    	try {
            shmFd = INSTANCE_RT.shm_open(this.memoryName, O_RDWR | O_CREAT, 0700);
    	} catch (Exception ex) {
    		this.useLibRT = false;
            shmFd = INSTANCE_C.shm_open(this.memoryName, O_RDWR | O_CREAT, 0700);
    	}
        if (shmFd < 0) {
            throw new RuntimeException("shm_open failed, errno: " + Native.getLastError());
        }

        if (this.useLibRT && INSTANCE_RT.ftruncate(shmFd, this.size) == -1) {
        	INSTANCE_RT.close(shmFd);
            throw new RuntimeException("ftruncate failed, errno: " + Native.getLastError());
        } else if (!this.useLibRT && INSTANCE_C.ftruncate(shmFd, this.size) == -1) {
        	INSTANCE_C.close(shmFd);
            throw new RuntimeException("ftruncate failed, errno: " + Native.getLastError());
        }
        if (this.useLibRT)
        	pSharedMemory = INSTANCE_RT.mmap(Pointer.NULL, this.size, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        else
        	pSharedMemory = INSTANCE_C.mmap(Pointer.NULL, this.size, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        
        if (this.useLibRT && pSharedMemory == Pointer.NULL) {
        	INSTANCE_RT.close(shmFd);
            throw new RuntimeException("mmap failed, errno: " + Native.getLastError());
        } else if (!this.useLibRT && pSharedMemory == Pointer.NULL) {
        	INSTANCE_C.close(shmFd);
            throw new RuntimeException("mmap failed, errno: " + Native.getLastError());
        }
    }

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem readOrCreate(String name, int size, long[] shape, String strDType, boolean isNumpy, boolean isFortran) {
		return new SharedMemoryArrayLiunxTem(name, size, strDType, shape, isNumpy, isFortran);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem create(int size, long[] shape, String strDType, boolean isNumpy, boolean isFortran) {
		return new SharedMemoryArrayLiunxTem(size, strDType, shape, isNumpy, isFortran);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem readOrCreate(String name, int size) {
		return new SharedMemoryArrayLiunxTem(name, size, null, null, false, false);
	}

	protected static SharedMemoryArrayLiunxTem create(int size) {
		return new SharedMemoryArrayLiunxTem(size, null, null, false, false);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai) {
		return createSHMAFromRAI(name, rai, false, true);
    }

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) {
		return build(name, rai, isFortranOrder, isNumpy);
    }

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem createSHMAFromRAI(RandomAccessibleInterval<T> rai) {
		return createSHMAFromRAI(rai, false, true);
    }

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayLiunxTem createSHMAFromRAI(RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) {
		return createSHMAFromRAI(SharedMemoryArray.createShmName(), rai, isFortranOrder, isNumpy);
    }

	/**
	 * This method copies the data from a {@link RandomAccessibleInterval} into a shared memory region
	 * to be able to shared it with other processes.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region where the {@link RandomAccessibleInterval} data has been copied.
     * 	The name should consist of "/" + file_name, where file_name should not contain any special character
	 * @param rai
	 * 	the {@link RandomAccessibleInterval} that is going to be written into a shared memory region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
    protected static <T extends RealType<T> & NativeType<T>> 
    SharedMemoryArrayLiunxTem build(String name, RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayLiunxTem shma = null;
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildInt8(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildUint8(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildInt16(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildUint16(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildInt32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildUint32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildInt64(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildFloat32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayLiunxTem(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isFortranOrder, isNumpy);
        	shma.buildFloat64(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
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

    private void buildInt8(RandomAccessibleInterval<ByteType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
		Cursor<ByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setByte(offset + (i ++), cursor.get().get());
		}
    }

    private void buildUint8(RandomAccessibleInterval<UnsignedByteType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
		Cursor<UnsignedByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.pSharedMemory.setByte(offset + (i ++), cursor.get().getByte());
		}
    }

    private void buildInt16(RandomAccessibleInterval<ShortType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.pSharedMemory.setByte(offset, b);
    		offset ++;
    	}
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
    public String getName() {
    	return this.memoryName;
    }
    
    /**
     * {@inheritDoc}
     */
    public String getNameForPython() {
    	return this.memoryName.substring("/".length());
    }
    
    /**
     * {@inheritDoc}
     */
    public Pointer getPointer() {
    	return this.pSharedMemory;
    }
    
    /**
     * {@inheritDoc}
     */
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
		return this.isNumpyFormat;
	}

	@Override
	/** TODO add close and unlink separated
	 * Unmap and close the shared memory. Necessary to eliminate the shared memory block
	 */
	public void close() {
		if (this.unlinked) return;
		int checkhmFd;
		if (this.useLibRT) checkhmFd = INSTANCE_RT.shm_open(this.memoryName, O_RDONLY, 0700);
		else checkhmFd = INSTANCE_C.shm_open(this.memoryName, O_RDONLY, 0700);
		
        if (checkhmFd < 0) {
            unlinked = true;
            return;
        }

        // Unmap the shared memory
        if (this.pSharedMemory != Pointer.NULL && this.useLibRT && INSTANCE_RT.munmap(this.pSharedMemory, size) == -1) {
            throw new RuntimeException("munmap failed. Errno: " + Native.getLastError());
        } else if (this.pSharedMemory != Pointer.NULL && !this.useLibRT && INSTANCE_C.munmap(this.pSharedMemory, size) == -1) {
            throw new RuntimeException("munmap failed. Errno: " + Native.getLastError());
        }

        // Close the file descriptor
        if (this.useLibRT && INSTANCE_RT.close(this.shmFd) == -1) {
            throw new RuntimeException("close failed. Errno: " + Native.getLastError());
        } else if (!this.useLibRT && INSTANCE_C.close(this.shmFd) == -1) {
            throw new RuntimeException("close failed. Errno: " + Native.getLastError());
        }

        // Unlink the shared memory object
        if (this.useLibRT) INSTANCE_RT.shm_unlink(memoryName);
        else INSTANCE_C.shm_unlink(memoryName);
        unlinked = true;
	}
    
    /**
     * Retrieve the {@link RandomAccessibleInterval} defined in the shared memory segment
     * 
     * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
     * @return the randomAccessible interval that is defined in the shared memory segment
     */
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI();

    /**
     * Retrieve the {@link RandomAccessibleInterval} defined in the shared memory segment
     * 
     * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape (array dimensions) into which the flat array of the shared memory segment will be reconstructed
	 * @param isFortran
	 * 	whether converting the falt array into a ndarray is done using Fortran ordering or not (C-ordering)
	 * @param dataType
	 * 	the data type into which the bytes in the shared memory region will be converted
     * @return the randomAccessible interval that is defined in the shared memory segment
     */
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, boolean isFortran, T dataType);
    
    /**
     * Copy the data from the {@link RandomAccessibleInterval} to the Shared memory segment.
     * TODO decide whether the copy is in fortran or c order
     * 
     * Note that if the dimensions of the array are not valid for the shared memory array, it will throw an exception
     * 
     * @param <T>
     * 	the possible ImgLib2 data types of the {@link RandomAccessibleInterval}
     * @param rai
     * 	the data array that is going to be copied into the shared memory array
     */
    public <T extends RealType<T> & NativeType<T>> void setRAI(RandomAccessibleInterval<T> rai);
    
    /**
     * Copy the ByteBuffer to the shared memory array.
     * @param buffer
     * 	the ByteBuffer to be copied to the shared memory region
     */
    public void setBuffer(ByteBuffer buffer);
    
    /**
     * 
     * @return the {@link ByteBuffer} with all the bytes of the Shared memory segment
     */
    public ByteBuffer getDataBuffer();
}
