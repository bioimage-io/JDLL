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
import java.nio.file.FileAlreadyExistsException;
import java.util.UUID;

import com.sun.jna.Pointer;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.utils.CommonUtils;
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

/**
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * 
 * Interface to interact with shared memory segments retrieving the underlying information 
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public interface SharedMemoryArray extends Closeable {

	/**
	 * Constant to specify that the shared memory segment that is going to be open is only for 
	 * reading
	 */
    public static final int O_RDONLY = 0;
	/**
	 * Constant to specify that the shared memory segment that is going to be open is for 
	 * reading and/or writing
	 */
    public static final int O_RDWR = 2;
	/**
	 * Constant to specify that the shared memory segment that is going to be open will
	 * be created if it does not exist
	 */
    public static final int O_CREAT = 64;
	/**
	 * Constant to specify that the shared memory regions mapped can be read but not written
	 */
    public static final int PROT_READ = 0x1;
	/**
	 * Constant to specify that the shared memory regions mapped can be written
	 */
    public static final int PROT_WRITE = 0x2;
	/**
	 * Constant to specify that the shared memory regions mapped can be shared with other processes
	 */
    public static final int MAP_SHARED = 0x01;
	/**
	 * List of special characters that should not be used to name shared memory segments
	 */
	final static String[] SPECIAL_CHARS_LIST = new String[] {"/", "\\", "#", "·", "!", "¡", "¿", "?", "@", "|", "$", ">", "<", ";"};


	/**
	 * This method creates a segment on the Shared Memory region of the computer with the size
	 * needed to store an image of the wanted characteristics.
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * The amount of space reserved will depend on the shape provided and the datatype.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region that has been created
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 * @throws FileAlreadyExistsException 
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray readOrCreate(String name, long[] shape, T datatype) throws FileAlreadyExistsException {
		String strDType = DecodeNumpy.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, null, false);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, null, false);
    	else 
    		return SharedMemoryArrayMacOS.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, null, false);
	}

	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray readOrCreate(String name, long[] shape, T datatype, boolean isFortran, boolean isNpy) throws FileAlreadyExistsException {
		String strDType = DecodeNumpy.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, isFortran, isNpy);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, isFortran, isNpy);
    	else 
    		return SharedMemoryArrayMacOS.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, isFortran, isNpy);
	}

	/**
	 * This method creates a segment on the Shared Memory region of the computer with the size
	 * needed to store an image of the wanted characteristics.
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * The amount of space reserved will depend on the shape provided and the datatype.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region that has been created
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 * @throws FileAlreadyExistsException 
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray readOrCreate(String name, int size) throws FileAlreadyExistsException {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.readOrCreate(name, size);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.readOrCreate(name, size);
    	else 
    		return SharedMemoryArrayMacOS.readOrCreate(name, size);
	}

	/**
	 * This method creates a segment on the Shared Memory region of the computer with the size
	 * needed to store an image of the wanted characteristics.
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * The amount of space reserved will depend on the shape provided and the datatype.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray create(long[] shape, T datatype) {
		String strDType = CommonUtils.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.create(size, shape, strDType, true, false);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.create(size, shape, strDType, true, false);
    	else 
    		return SharedMemoryArrayMacOS.create(size, shape, strDType, true, false);
	}

	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray create(long[] shape, T datatype, boolean isFortran, boolean isNpy) {
		String strDType = DecodeNumpy.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.create(size, shape, strDType, isNpy, isFortran);
    	else if (PlatformDetection.isLinux())
    		return SharedMemoryArrayLinux.create(size, shape, strDType, isNpy, isFortran);
    	else 
    		return SharedMemoryArrayMacOS.create(size, shape, strDType, isNpy, isFortran);
	}

	/**
	 * This method creates a segment on the Shared Memory region of the computer with the size
	 * needed to store an image of the wanted characteristics.
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * The amount of space reserved will depend on the shape provided and the datatype.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static SharedMemoryArray create(int size) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.create(size);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.create(size);
    	else 
    		return SharedMemoryArrayMacOS.create(size);
	}

	static SharedMemoryArray read(String name) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.read(name);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.read(name);
    	else 
    		return SharedMemoryArrayMacOS.read(name);
	}

	static long getSize(String name) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.getSHMSize(name);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.getSHMSize(name);
    	else 
    		return SharedMemoryArrayMacOS.getSHMSize(name);
	}

	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai) throws FileAlreadyExistsException {
		return createSHMAFromRAI(name, rai, false, true);
    }

	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(RandomAccessibleInterval<T> rai) throws FileAlreadyExistsException {
		return createSHMAFromRAI(rai, false, true);
    }

	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) throws FileAlreadyExistsException {
		return createSHMAFromRAI(SharedMemoryArray.createShmName(), rai, isFortranOrder, isNumpy);
    }

	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) throws FileAlreadyExistsException {
        if (PlatformDetection.isWindows()) 
        	return null;//SharedMemoryArrayWin.createSHMAFromRAI(SharedMemoryArray.createShmName(), rai, isFortranOrder, isNumpy);;
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.createSHMAFromRAI(SharedMemoryArray.createShmName(), rai, isFortranOrder, isNumpy);
    	else 
    		return null;//SharedMemoryArrayMacOS.createSHMAFromRAI(SharedMemoryArray.createShmName(), rai, isFortranOrder, isNumpy);
    }
	
	/**
	 * Checks whether the String provided  can be used as the name given to a shared memory segment
	 * @param name
	 * 	the string that wants to be used as a name to a shared memory segment
	 * @throws IllegalArgumentException if the name does not fulfill the required conditions
	 */
	static void checkMemorySegmentName(String name) throws IllegalArgumentException {
		String auxName;
		if (PlatformDetection.isWindows() && name.startsWith("Local\\"))
			auxName = name.substring("Local\\".length());
		else if (name.startsWith("/"))
				auxName = name.substring(1);
		else 
			auxName = name;
		for (String specialChar : SPECIAL_CHARS_LIST) {
			if (auxName.contains(specialChar))
				throw new IllegalArgumentException("Argument 'name' should not contain the special character '" + specialChar + "'.");
		}
		if (PlatformDetection.isMacOS() && auxName.length() > SharedMemoryArrayMacOS.MACOS_MAX_LENGTH - 1)
			throw new IllegalArgumentException("Parameter 'name' cannot have more than " 
									+ (SharedMemoryArrayMacOS.MACOS_MAX_LENGTH - 1) + " characters. Shared memory segments "
									+ "cannot have names with more than " + (SharedMemoryArrayMacOS.MACOS_MAX_LENGTH - 1) + " characters.");
	}
	
	/**
	 * Create a random unique name for a shared memory segment
	 * @return a random unique name for a shared memory segment
	 */
	static String createShmName() {
        if (PlatformDetection.isWindows()) return "Local" + File.separator + UUID.randomUUID().toString();
    	else if (PlatformDetection.isLinux()) return "/shm-" + UUID.randomUUID();
    	else return ("/shm-" + UUID.randomUUID()).substring(0, SharedMemoryArrayMacOS.MACOS_MAX_LENGTH);
	}
    
	/**
	 * Get the number of bytes that is required to store the data in an nd array of a certain data type
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of the array
	 * @param type
	 * 	ImgLib2 data type of the array
	 * @return the number of bytes needed to store the nd array
	 */
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
    
	/**
	 * 
	 * @return the unique name for the shared memory, specified as a string. When creating a new shared memory bloc.k instance
	 * 	{@link SharedMemoryArray} a name can be supploed, and if not it will be generated automatically.
	 * 	Two shared memory blocks existing at the same time cannot share the name.
	 * 	In Unix based systems, Shared memory segment names start with "/", for example "/shm_block"
	 * 	In Windows shared memory block names start either with "Global\\" or "Local\\". Example: "Local\\shm_block" 
	 */
    public String getName();
    
    /**
     * 
	 * @return the unique name for the shared memory, specified as a string and as 
	 * 	the Python package multiprocessing.shared_memory returns it. For Unix based systems it removes the 
	 * 	initial "/", for example: "/shm_block" -&gt; "shm_block".
	 * 	In Windows shared memory block names start either with "Global\\" or "Local\\", this is also removed when 
	 * 	providing a shared memory name to Python. Example: "Local\\shm_block" -&gt; "shm_block"
     */
    public String getNameForPython();
    
    /**
     * 
     * @return the pointer to the shared memory segment
     */
    public Pointer getPointer();
    
    /**
     * 
     * @return get number of bytes in the shared memory segment
     */
    public int getSize();
    
    /**
     * 
     * @return the data type of the array that was flattened and copied into the shared memory segment
     */
    public String getOriginalDataType();
    
    /**
     * 
     * @return the shape (array dimensions) of the array that was flattened and copied into the shared memory segment
     */
    public long[] getOriginalShape();
    
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
    // TODO public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, boolean isFortran, T dataType);
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType);
    
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
    // TODO is it necessary? public <T extends RealType<T> & NativeType<T>> void setRAI(RandomAccessibleInterval<T> rai);
    
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
    
    /**
     * 
     * @return whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
     */
    public boolean isNumpyFormat();
}
