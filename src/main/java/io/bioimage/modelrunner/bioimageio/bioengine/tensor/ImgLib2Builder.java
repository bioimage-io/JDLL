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

import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

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
			case BioengineTensor.UBYTE_STR:
				return (Img<T>) buildFromTensorUByte(tensor);
			case BioengineTensor.BYTE_STR:
				return (Img<T>) buildFromTensorUByte(tensor);
			case BioengineTensor.INT16_STR:
				return (Img<T>) buildFromTensorInt16(tensor);
			case BioengineTensor.UINT16_STR:
				return (Img<T>) buildFromTensorUInt16(tensor);
			case BioengineTensor.INT32_STR:
				return (Img<T>) buildFromTensorInt32(tensor);
			case BioengineTensor.UINT32_STR:
				return (Img<T>) buildFromTensorUInt32(tensor);
			case BioengineTensor.INT64_STR:
				return (Img<T>) buildFromTensorInt64(tensor);
			case BioengineTensor.FLOAT32_STR:
				return (Img<T>) buildFromTensorFloat32(tensor);
			case BioengineTensor.FLOAT64_STR:
				return (Img<T>) buildFromTensorFloat64(tensor);
			default:
				throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDType());
		}
	}

	private static Img<LongType> buildFromTensorInt64(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<DoubleType> buildFromTensorFloat64(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<FloatType> buildFromTensorFloat32(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<UnsignedIntType> buildFromTensorUInt32(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<IntType> buildFromTensorInt32(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<UnsignedShortType> buildFromTensorUInt16(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<ShortType> buildFromTensorInt16(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Img<ByteType> buildFromTensorUByte(BioEngineOutputArray tensor) {
		// TODO Auto-generated method stub
		return null;
	}

}
