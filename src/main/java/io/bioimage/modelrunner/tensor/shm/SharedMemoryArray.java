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

import java.io.Closeable;
import java.util.HashMap;
import java.util.UUID;

import com.sun.jna.Pointer;

import io.bioimage.modelrunner.system.PlatformDetection;
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
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildSHMA(RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.build(rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.build(rai);
    	else return SharedMemoryArrayMacOS.build(rai);
    }

	/**
	 * This method copies the data from a {@link RandomAccessibleInterval} into a shared memory region
	 * to be able to shared it with other processes.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region where the {@link RandomAccessibleInterval} data has been copied
	 * @param rai
	 * 	the {@link RandomAccessibleInterval} that is going to be written into a shared memory region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildSHMA(String name, RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.build(name, rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.build(name, rai);
    	else return SharedMemoryArrayMacOS.build(name, rai);
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
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildNumpyLikeSHMA(RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.buildNumpyFormat(rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.buildNumpyFormat(rai);
    	else return SharedMemoryArrayMacOS.buildNumpyFormat(rai);
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
	 * @param rai
	 * 	the {@link RandomAccessibleInterval} that is going to be written into a shared memory region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildNumpyLikeSHMA(String name, RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.buildNumpyFormat(name, rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.buildNumpyFormat(name, rai);
    	else return SharedMemoryArrayMacOS.buildNumpyFormat(name, rai);
    }
	
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
	static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromSHMA(String memoryName, long[] shape, boolean isFortran, String dataType) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
        else if (PlatformDetection.isLinux())
    		return SharedMemoryArrayLinux.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
        else
    		return SharedMemoryArrayMacOS.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
	}

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
	static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromNumpyLikeSHMA(String memoryName) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.buildImgLib2FromNumpyLikeSHMA(memoryName);
        else if (PlatformDetection.isLinux())
    		return SharedMemoryArrayLinux.buildImgLib2FromNumpyLikeSHMA(memoryName);
        else
    		return SharedMemoryArrayMacOS.buildImgLib2FromNumpyLikeSHMA(memoryName);
	}

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
	static HashMap<String, Object> buildMapFromNumpyLikeSHMA(String memoryName) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.buildMapFromNumpyLikeSHMA(memoryName);
        else if (PlatformDetection.isLinux())
    		return SharedMemoryArrayLinux.buildMapFromNumpyLikeSHMA(memoryName);
        else
    		return SharedMemoryArrayMacOS.buildMapFromNumpyLikeSHMA(memoryName);
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
        if (PlatformDetection.isWindows()) return "Local\\" + UUID.randomUUID().toString();
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
     * 
     * @return whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
     */
    public boolean isNumpyFormat();
}
