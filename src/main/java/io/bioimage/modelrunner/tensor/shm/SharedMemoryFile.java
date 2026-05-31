/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.numpy.DecodeNumpy;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * TODO get rid of this class in favour of shared memory IPC
 * Class that maps {@link Tensor} objects to a temporal file for inter-processing communication
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class SharedMemoryFile
{
	/**
	 * Create a .npy Numpy file from a {@link RandomAccessibleInterval}
	 *
	 * @param <T> the T type parameter.
	 * @param fileDir the file directory.
	 * @param rai the RAI.
	 * @throws FileNotFoundException if the requested file cannot be found.
	 * @throws IOException if an I/O error occurs.
	 */
	public static <T extends RealType<T> & NativeType<T>> 
	void buildFileFromRai(String fileDir, RandomAccessibleInterval<T> rai) throws FileNotFoundException, IOException {
		DecodeNumpy.saveNpy(fileDir, rai);
	}
	
	/**
	 * Read a {@link RandomAccessibleInterval} from a .npy Numpy file created to share between processes
	 *
	 * @param <T> the T type parameter.
	 * @param fileName the file name.
	 * @return the created random accessible interval.
	 * @throws FileNotFoundException if the requested file cannot be found.
	 * @throws IOException if an I/O error occurs.
	 */
	public static <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> buildRaiFromFile(String fileName) throws FileNotFoundException, IOException {
		return DecodeNumpy.loadNpy(fileName);
	}
}
