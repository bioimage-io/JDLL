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
package io.bioimage.modelrunner.bioimageio.bioengine.tensor;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import net.imglib2.img.Img;
import net.imglib2.type.Type;

public class ImgLib2Builder {

	/**
	 * Not used (Utility class).
	 */
	private ImgLib2Builder() {}

	/**
	 * Creates a {@link Img} from a given {@link BioEngineOutputArray}
	 * 
	 * @param <T>
	 *  the type of the image
	 * @param tensor
	 *  The bioengine tensor data is read from.
	 * @return The Img built from the tensor.
	 * @throws IllegalArgumentException If the tensor type is not supported.
	 */
	@SuppressWarnings("unchecked")
	public static <T extends Type<T>> Img<T> build(BioEngineOutputArray tensor)
		throws IllegalArgumentException
	{
		// Create an Img of the same type of the tensor
		switch (tensor.getDType()) {
			case UINT8:
				return (Img<T>) buildFromTensorUByte(tensor);
			case INT32:
				return (Img<T>) buildFromTensorInt(tensor);
			case FLOAT:
				return (Img<T>) buildFromTensorFloat(tensor);
			case DOUBLE:
				return (Img<T>) buildFromTensorDouble(tensor);
			default:
				throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDType());
		}
	}

}
