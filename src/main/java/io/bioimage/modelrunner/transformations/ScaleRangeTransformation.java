/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.transformations;

import java.util.Arrays;

import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class ScaleRangeTransformation extends AbstractTensorTransformation
{

	private static final String name = "scale_range";
	private double minPercentile = 0;
	private double maxPercentile = 100;
	private String axes;
	private Mode mode = Mode.PER_SAMPLE;
	private String tensorName;
	private double eps = Math.pow(10, -6);
	
	public ScaleRangeTransformation()
	{
		super( name );
	}
	
	public void setEps(Object eps) {
		if (eps instanceof Integer) {
			this.eps = Double.valueOf((int) eps);
		} else if (eps instanceof Double) {
			this.eps = (double) eps;
		} else if (eps instanceof String) {
			this.eps = Double.valueOf((String) eps);
		} else {
			throw new IllegalArgumentException("'eps' parameter has to be either and instance of "
					+ Float.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + eps.getClass());
		}
	}
	
	public void setMinPercentile(Object minPercentile) {
		if (minPercentile instanceof Integer) {
			this.minPercentile = Double.valueOf((int) minPercentile) / 100;
		} else if (minPercentile instanceof Double) {
			this.minPercentile = ((double) minPercentile) / 100;
		} else if (minPercentile instanceof String) {
			this.minPercentile = Double.valueOf((String) minPercentile) / 100;
		} else {
			throw new IllegalArgumentException("'minPercentile' parameter has to be either and instance of "
					+ Integer.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + minPercentile.getClass());
		}
	}
	
	public void setMaxPercentile(Object maxPercentile) {
		if (maxPercentile instanceof Integer) {
			this.maxPercentile = Double.valueOf((int) maxPercentile) / 100;
		} else if (maxPercentile instanceof Double) {
			this.maxPercentile = ((double) maxPercentile) / 100;
		} else if (maxPercentile instanceof String) {
			this.maxPercentile = Double.valueOf((String) maxPercentile) / 100;
		} else {
			throw new IllegalArgumentException("'maxPercentile' parameter has to be either and instance of "
					+ Integer.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + maxPercentile.getClass());
		}
	}
	
	public void setAxes(Object axes) {
		if (axes instanceof String )
			this.axes = (String) axes;
		else
			throw new IllegalArgumentException("'axes' parameter has to be an instance of " + String.class
					 + ". The provided argument is " + axes.getClass());
	}
	
	public void setTensorName(Object tensorName) {
		if (tensorName instanceof String )
			this.tensorName = (String) tensorName;
		else
			throw new IllegalArgumentException("'tensorName' parameter has to be an instance of " + String.class
					 + ". The provided argument is " + tensorName.getClass());
	}
	
	public void setMode(Object mode) {
		if (mode instanceof String )
			this.mode = Mode.valueOf(((String) mode).toUpperCase());
		else if (mode instanceof Mode)
			this.mode = (Mode) mode;
		else
			throw new IllegalArgumentException("'mode' parameter has to be either and instance of " + String.class
					+ " or " + Mode.class + ". The provided argument is an instance of: " + mode.getClass());
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		final Tensor< FloatType > output = makeOutput( input );
		applyInPlace(output);
		return output;
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > void applyInPlace(Tensor<R> input) {
		String selectedAxes = "";
		for (String ax : input.getAxesOrderString().split("")) {
			if (axes != null && !axes.toLowerCase().contains(ax.toLowerCase())
					&& !ax.toLowerCase().equals("b"))
				selectedAxes += ax;
		}
		if (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().replace("b", "").length() == selectedAxes.length()) {
			globalScale(input);
		} else if (axes.length() > 0) {
			axesScale(input, selectedAxes);
		}
		
	}
	
	private < R extends RealType< R > & NativeType< R > > void globalScale( final Tensor< R > output ) {
		double minPercentileVal = findPercentileValue(output.getData(), minPercentile);
		double maxPercentileVal = findPercentileValue(output.getData(), maxPercentile);
		scaleRange(output.getData(), maxPercentileVal, minPercentileVal);
	}
	
	private < R extends RealType< R > & NativeType< R > > 
	double findPercentileValue(RandomAccessibleInterval<R> rai, double percentile) {
		final IterableInterval<R> flatImage = Views.iterable(rai);
		long flatSize = Arrays.stream(flatImage.dimensionsAsLongArray()).reduce(1, (a, b) -> a * b);
		double[] flatArr = new double[(int) flatSize];
		
		int count = 0;
		final Cursor<R> cursor = (Cursor<R>) flatImage.cursor();
		while ( cursor.hasNext() )
		{
			cursor.next();
			flatArr[count ++] = cursor.get().getRealDouble();
		}
		Arrays.sort(flatArr);
		
		int percentilePos = (int) (flatSize * percentile);
		percentilePos = percentilePos >= flatArr.length ? flatArr.length - 1 : percentilePos;
		return flatArr[percentilePos];
	}
	
	private < R extends RealType< R > & NativeType< R > > 
	void axesScale( final Tensor< R > output, String axesOfInterest) {
		long[] start = new long[output.getData().numDimensions()];
		long[] dims = output.getData().dimensionsAsLongArray();
		long[] indOfDims = new long[axesOfInterest.length()];
		long[] sizeOfDims = new long[axesOfInterest.length()];
		for (int i = 0; i < indOfDims.length; i ++) {
			indOfDims[i] = output.getAxesOrderString().indexOf(axesOfInterest.split("")[i]);
		}
		for (int i = 0; i < sizeOfDims.length; i ++) {
			sizeOfDims[i] = dims[(int) indOfDims[i]];
		}
		
		long[][] points = getAllCombinations(sizeOfDims);
		for (long[] pp : points) {
			for (int i = 0; i < pp.length; i ++) {
				start[(int) indOfDims[i]] = pp[i];
				dims[(int) indOfDims[i]] = pp[i] + 1;
			}
			// Define the view by defining the length per axis
			long[] end = new long[dims.length];
			for (int i = 0; i < dims.length; i ++) end[i] = dims[i] - start[i];
			IntervalView<R> plane = Views.offsetInterval( output.getData(), start, end );
			double minPercentileVal = findPercentileValue(plane, minPercentile);
			double maxPercentileVal = findPercentileValue(plane, maxPercentile);
			scaleRange(plane, maxPercentileVal, minPercentileVal);
		}
	}
	
	private static long[][] getAllCombinations(long[] arr){
		long n = 1;
		for (long nn : arr) n *= nn;
		long[][] allPoints = new long[(int) n][arr.length];
		for (int i = 0; i < n; i ++) {
			for (int j = 0; j < arr.length; j ++) {
				int factor = 1;
				for (int k = 0; k < j; k ++) {
					factor *= arr[k];
				}
				int auxVal = i / factor;
				int val = auxVal % ((int) arr[j]);
				allPoints[i][j] = val;
			}
		}
		return allPoints;
	}
	
	public static void main(String[] args) {
		test1();
		test2();
	}
	
	public static void test1() {
		float[] arr = new float[9];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		 ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {3, 3});
		 ScaleRangeTransformation preprocessing = new ScaleRangeTransformation();
		 Tensor<FloatType> tt = Tensor.build("name", "xy", rai);
		 preprocessing.applyInPlace(tt);
		 System.out.print(true);
	}
	
	public static void test2() {
		float[] arr = new float[18];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		 ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {3, 3, 2});
		 ScaleRangeTransformation preprocessing = new ScaleRangeTransformation();
		 preprocessing.setAxes("xy");
		 preprocessing.setMaxPercentile(99);
		 preprocessing.setMinPercentile(1);
		 Tensor<FloatType> tt = Tensor.build("name", "xyc", rai);
		 preprocessing.applyInPlace(tt);
		 System.out.print(true);
	}
	
	public < R extends RealType< R > & NativeType< R > > 
	void scaleRange(RandomAccessibleInterval<R> rai, double maxPercentileVal, double minPercentileVal) {
		double diff = maxPercentileVal - minPercentileVal;

        R type = Util.getTypeFromInterval(rai);
        if (type instanceof IntegerType) {
			LoopBuilder.setImages( rai )
			.multiThreaded()
			.forEachPixel( i -> i.setReal(Math.floor((i.getRealDouble() - minPercentileVal) / (diff + eps)) ) );
        } else {
			LoopBuilder.setImages( rai )
			.multiThreaded()
			.forEachPixel( i -> i.setReal(((i.getRealDouble() - minPercentileVal) / (diff + eps)) ) );
        }
	}
}
