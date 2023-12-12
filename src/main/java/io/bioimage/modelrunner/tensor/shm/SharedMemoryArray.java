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
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
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
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
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
	 * 
	 * @param <T>
	 * @param memoryName
	 * @param shape
	 * @param isFortran
	 * @param dataType
	 * @return
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
	 * 
	 * @param <T>
	 * @param memoryName
	 * @return
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
	 * 
	 * @param memoryName
	 * @return
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
	 * 
	 * @param <T>
	 * @param rai
	 * @return
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildNumpyLikeSHMA(RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.buildNumpyFormat(rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.buildNumpyFormat(rai);
    	else return SharedMemoryArrayMacOS.buildNumpyFormat(rai);
    }
	
	/**
	 * 
	 * @param <T>
	 * @param name
	 * @param rai
	 * @return
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildNumpyLikeSHMA(String name, RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.buildNumpyFormat(name, rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.buildNumpyFormat(name, rai);
    	else return SharedMemoryArrayMacOS.buildNumpyFormat(name, rai);
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
	 * 	initial "/", for example: "/shm_block" -> "shm_block".
	 * 	In Windows shared memory block names start either with "Global\\" or "Local\\", this is also removed when 
	 * 	providing a shared memory name to Python. Example: "Local\\shm_block" -> "shm_block"
     */
    public String getNameForPython();
    
    /**
     * 
     * @return the pointer to the shared memory segment
     */
    public Pointer getPointer();
    
    /**
     * 
     * @return
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
