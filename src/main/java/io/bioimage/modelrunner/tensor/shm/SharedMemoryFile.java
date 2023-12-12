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

import java.io.FileNotFoundException;
import java.io.IOException;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Class that maps {@link Tensor} objects to the shared memory for interprocessing communication
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class SharedMemoryFile
{
	public static <T extends RealType<T> & NativeType<T>> 
	void buildFileFromRai(String fileDir, RandomAccessibleInterval<T> rai) throws FileNotFoundException, IOException {
		DecodeNumpy.writeRaiToNpyFile(fileDir, rai);
	}
	
	public static <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> buildRaiFromFile(String fileName) throws FileNotFoundException, IOException {
		return DecodeNumpy.retrieveImgLib2FromNpy(fileName);
	}
}
