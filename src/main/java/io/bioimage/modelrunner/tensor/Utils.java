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
package io.bioimage.modelrunner.tensor;

import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.transform.integer.MixedTransform;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.MixedTransformView;
import net.imglib2.view.Views;

/**
 * Provide static methods that can be used to make transformations on tensors
 *
 * @author Carlos Garcia Lopez de Haro
 */
public final class Utils
{
	/**
	 * Method that transposes every dimension of a {@link RandomAccessibleInterval}.
	 * If a {@link RandomAccessibleInterval} with {3, 256, 512} dimensions is provided,
	 * it will be converted into one of {512, 256, 3}
	 * @param <T>
	 * 	possible data types of the {@link RandomAccessibleInterval}
	 * @param rai
	 * 	{@link RandomAccessibleInterval} to be transposed
	 * @return the transposed {@link RandomAccessibleInterval}
	 */
	public static <T extends NumericType<T> & RealType<T>> 
	RandomAccessibleInterval<T> transpose(RandomAccessibleInterval<T> rai){
		long[] max = rai.maxAsPoint().positionAsLongArray();
		long[] min = rai.minAsPoint().positionAsLongArray();
		long[] tensorShape = rai.dimensionsAsLongArray();
		MixedTransform t = new MixedTransform( tensorShape.length, tensorShape.length );
		int[] transposeAxesOrderChange = new int[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) transposeAxesOrderChange[i] = tensorShape.length - 1 - i;
		t.setComponentMapping(transposeAxesOrderChange);
		long[] minMax = new long[tensorShape.length * 2];
		for (int i = 0; i < tensorShape.length; i ++) {
			minMax[i] = min[tensorShape.length - i - 1] - 1;
			minMax[i + tensorShape.length] = max[tensorShape.length - i - 1] - 1;
		}
		return Views.interval(new MixedTransformView<T>( rai, t ), 
				Intervals.createMinMax(minMax));
	}
}
