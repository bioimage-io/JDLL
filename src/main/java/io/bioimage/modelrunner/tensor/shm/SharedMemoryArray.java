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
	
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray buildSHMA(RandomAccessibleInterval<T> rai) {
        if (PlatformDetection.isWindows()) return SharedMemoryArrayWin.build(rai);
    	else return SharedMemoryArrayPosix.build(rai);
    }
	
	static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromSHMA(String memoryName, long[] shape, boolean isFortran, String dataType) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);
    	else 
    		return SharedMemoryArrayPosix.createImgLib2RaiFromSharedMemoryBlock(memoryName, shape, isFortran, dataType);

	}
    
    public String getMemoryLocationName();
    
    public String getMemoryLocationPythonName();
    
    public Pointer getPointer();
    
    public int getSize();
}
