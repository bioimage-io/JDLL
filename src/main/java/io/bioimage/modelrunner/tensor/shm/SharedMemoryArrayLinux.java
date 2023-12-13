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

import java.io.ByteArrayInputStream;
import java.util.HashMap;

import com.sun.jna.Pointer;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.tensor.Tensor;
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
 * Class that maps {@link Tensor} objects to the shared memory for inter-processing communication
 * in LINUX based systems
 * @author Carlos Garcia Lopez de Haro
 */
public class SharedMemoryArrayLinux implements SharedMemoryArray
{
	/**
	 * Instance of the CLibrary JNI containing the methods to interact with the Shared memory segments
	 */
	private static final CLibrary INSTANCE = CLibrary.INSTANCE;
	/**
	 * File descriptor value of the shared memory segment
	 */
	private final int shmFd;
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
    private SharedMemoryArrayLinux(int size, String dtype, long[] shape)
    {
    	this(SharedMemoryArray.createShmName(), size, dtype, shape);
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
    private SharedMemoryArrayLinux(String name, int size, String dtype, long[] shape)
    {
    	this.originalDataType = dtype;
    	this.originalDims = shape;
    	this.size = size;
    	this.memoryName = name;

        shmFd = INSTANCE.shm_open(this.memoryName, O_RDWR | O_CREAT, 0700);
        if (shmFd < 0) {
            throw new RuntimeException("shm_open failed, errno: " + Native.getLastError());
        }

        if (INSTANCE.ftruncate(shmFd, this.size) == -1) {
        	INSTANCE.close(shmFd);
            throw new RuntimeException("ftruncate failed, errno: " + Native.getLastError());
        }

        pSharedMemory = INSTANCE.mmap(Pointer.NULL, this.size, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        if (pSharedMemory == Pointer.NULL) {
        	INSTANCE.close(shmFd);
            throw new RuntimeException("mmap failed, errno: " + Native.getLastError());
        }
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

	/**
	 * This method copies the data from a {@link RandomAccessibleInterval} into a shared memory region
	 * to be able to shared it with other processes.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * The name is assigned automatically.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
	 * @param rai
	 * 	the {@link RandomAccessibleInterval} that is going to be written into a shared memory region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayLinux build(RandomAccessibleInterval<T> rai)
    {
    	return build(SharedMemoryArray.createShmName(), rai);
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
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayLinux build(String name, RandomAccessibleInterval<T> rai)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayLinux shma = null;
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildUint32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildInt64(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildFloat32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	shma = new SharedMemoryArrayLinux(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
        	shma.buildFloat64(Cast.unchecked(rai));
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
		return shma;
    }

	/**
	 * This method copies the data from a {@link RandomAccessibleInterval} into a shared memory region
	 * to be able to shared it with other processes.
	 * This method copies the data into the shared memory region following the Numpy .npy format. This means
	 * that the header of the region will contain info about the shape, the byte order, the column order (whether
	 * is fortran or not) and the data type.
	 * This way, the underlying nd array can be reconstructed just with the shared memory region name.
	 * 
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * The name is assigned automatically.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
	 * @param rai
	 * 	the {@link RandomAccessibleInterval} that is going to be written into a shared memory region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayLinux buildNumpyFormat(RandomAccessibleInterval<T> rai)
    {
    	return buildNumpyFormat(SharedMemoryArray.createShmName(), rai);
    }

	/**
	 * This method copies the data from a {@link RandomAccessibleInterval} into a shared memory region
	 * to be able to shared it with other processes.
	 * This method copies the data into the shared memory region following the Numpy .npy format. This means
	 * that the header of the region will contain info about the shape, the byte order, the column order (whether
	 * is fortran or not) and the data type.
	 * This way, the underlying nd array can be reconstructed just with the shared memory region name.
	 * 
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region where the {@link RandomAccessibleInterval} data has been copied
     * 	The name should consist of "/" + file_name, where file_name should not contain any special character
	 * @param rai
	 * 	the {@link RandomAccessibleInterval} that is going to be written into a shared memory region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
    protected static <T extends RealType<T> & NativeType<T>> SharedMemoryArrayLinux buildNumpyFormat(String name, RandomAccessibleInterval<T> rai)
    {
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("/"))
    		name = "/" + name;
    	SharedMemoryArrayLinux shma = null;
    	byte[] total = DecodeNumpy.createNumpyStyleByteArray(rai);
    	shma = new SharedMemoryArrayLinux(name, total.length, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray());
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
	/** TODO add close and unlink separated
	 * Unmap and close the shared memory. Necessary to eliminate the shared memory block
	 */
	public void close() {
		if (this.unlinked) return;
		
		int checkhmFd = INSTANCE.shm_open(this.memoryName, O_RDONLY, 0700);
        if (checkhmFd < 0) {
            unlinked = true;
            return;
        }

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
	/**
	 * Build a {@link HashMap} from the data stored in an existing shared memory segment.
	 * The returned {@link HashMap} contains one entry for the data type, another for the shape (array dimensions),
	 * byte ordering, column order (whether it is Fortran ordering or C ordering) and another for the actual byte
	 * data (a flat array with the byte values of the array).
	 * 
	 * The shared memory segment should contain an array of bytes that can be read using the .npy format.
	 * That is an array of bytes which specifies the characteristics of the nd array (shape, data type, byte order...)
	 * followed by the flattened data converted into bytes.
	 * If the shared memory region follows that convention, only the name of the shared memory region is needed to 
	 * reconstruct the underlying nd array.
	 * 
	 * @param memoryName
	 * 	name of the region where the shared memory segment is located
	 * @return the {@link RandomAccessibleInterval} defined exclusively by the shared memory region following the .npy format
	 */
	public static HashMap<String, Object> buildMapFromNumpyLikeSHMA(String memoryName) {
		if (!memoryName.startsWith("/")) memoryName = "/" + memoryName;
	    int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
        if (shmFd < 0 )
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
        pSharedMemory.read(0, flat, 0, flat.length);
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
        	throw new RuntimeException(ex.toString());
        }
	}
	
	// TODO support boolean
	/**
	 * Build a {@link RandomAccessibleInterval} from the data stored in an existing shared memory segment.
	 * The shared memory segment should contain an array of bytes that can be read using the .npy format.
	 * That is an array of bytes which specifies the characteristics of the nd array (shape, data type, byte order...)
	 * followed by the flattened data converted into bytes.
	 * If the shared memory region follows that convention, only the name of the shared memory region is needed to 
	 * reconstruct the underlying nd array
	 * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
	 * @param memoryName
	 * 	name of the region where the shared memory segment is located
	 * @return the {@link RandomAccessibleInterval} defined exclusively by the shared memory region following the .npy format
	 */
	public static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromNumpyLikeSHMA(String memoryName) {
		if (!memoryName.startsWith("/")) memoryName = "/" + memoryName;
	    int shmFd = INSTANCE.shm_open(memoryName, O_RDONLY, 0700);
        if (shmFd < 0) throw new RuntimeException("Failed to open shared memory. Errno: " + Native.getLastError());

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
        pSharedMemory.read(0, flat, 0, flat.length);
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
	/**
	 * Build a {@link RandomAccessibleInterval} from the data stored in an existing shared memory segment.
	 * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
	 * @param memoryName
	 * 	name of the region where the shared memory segment is located
	 * @param shape
	 * 	shape (array dimensions) into which the flat array of the shared memory segment will be reconstructed
	 * @param isFortran
	 * 	whether converting the falt array into a ndarray is done using Fortran ordering or not (C-ordering)
	 * @param dataType
	 * 	the data type into which the bytes in the shared memory region will be converted
	 * @return the {@link RandomAccessibleInterval} defined by the arguments and the shared memory segment
	 */
	public static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> createImgLib2RaiFromSharedMemoryBlock(String memoryName, long[] shape, boolean isFortran, String dataType) {
		int size = SharedMemoryArray.getArrayByteSize(shape, Cast.unchecked(CommonUtils.getImgLib2DataType(dataType)));
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
    
    public static void main(String[] args) {
    }

	@Override
    /**
     * {@inheritDoc}
     */
	public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI() {
		return buildFromSharedMemoryBlock(pSharedMemory, this.originalDims, false, this.originalDataType);
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
}
