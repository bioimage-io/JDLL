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

import com.sun.jna.Pointer;

import io.bioimage.modelrunner.system.PlatformDetection;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Interface that contains the methods required to create or read Shared Memory Blocks for JDLL
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public interface SharedMemoryArray extends Closeable {
	
	final static String[] SPECIAL_CHARS_LIST = new String[] {"/", "\\", "#", "·", "!", "¡", "¿", "?", "@", "|", "$", ">", "<", ";"};

	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildSHMA(RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.build(rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.build(rai);
    	else return SharedMemoryArrayMacOS.build(rai);
    }
	
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildSHMA(String name, RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.build(name, rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.build(name, rai);
    	else return SharedMemoryArrayMacOS.build(name, rai);
    }
	
	static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromSHMA(String memoryName, long[] shape, boolean isFortran, String dataType) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
        else if (PlatformDetection.isLinux())
    		return SharedMemoryArrayLinux.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
        else
    		return SharedMemoryArrayMacOS.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
	}
	
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildNumpyLikeSHMA(RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.buildNumpyFormat(rai);
    	else if (PlatformDetection.isLinux()) return SharedMemoryArrayLinux.buildNumpyFormat(rai);
    	else return SharedMemoryArrayMacOS.buildNumpyFormat(rai);
    }
	
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
    
    public String getMemoryLocationName();
    
    public String getMemoryLocationPythonName();
    
    public Pointer getPointer();
    
    public int getSize();
    
    public String getOriginalDataType();
    
    public long[] getOriginalShape();
    
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI();
    
    /**
     * 
     * @return whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
     */
    public boolean isNumpyFormat();
}
