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

import java.util.ArrayList;
import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
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

public class FixedZeroMeanUnitVarianceTransformation extends AbstractTensorTransformation
{
	
	// TODO change fixedAxesMeanStd method to what ZeroMeanUnitVarianceTransformation does once n2v has figured it
	// TODO change fixedAxesMeanStd method to what ZeroMeanUnitVarianceTransformation does once n2v has figured it
	// TODO change fixedAxesMeanStd method to what ZeroMeanUnitVarianceTransformation does once n2v has figured it
	// TODO change fixedAxesMeanStd method to what ZeroMeanUnitVarianceTransformation does once n2v has figured it
	// TODO change fixedAxesMeanStd method to what ZeroMeanUnitVarianceTransformation does once n2v has figured it
	// TODO change fixedAxesMeanStd method to what ZeroMeanUnitVarianceTransformation does once n2v has figured it
	
	private static String name = "fixed_zero_mean_unit_variace";
	private Double meanDouble;
	private Double stdDouble;
	private double[] meanArr;
	private double[] stdArr;
	private String axes;
	private double eps = Math.pow(10, -6);

	private static String FIXED_MODE_ERR = "If the mode is 'fixed', the parameters 'mean' and"
			+ " 'std need to be specified";
	private static String NOT_FIXED_MODE_ERR = "Only the mode 'fixed' requires providing the "
			+ "'std' and 'mean parameters.";

	public FixedZeroMeanUnitVarianceTransformation()
	{
		super( name );
		mode = Mode.FIXED;
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
	
	public void setMean(Object mean) {
		if (mean instanceof Integer) {
			this.meanDouble = Double.valueOf((int) mean);
		} else if (mean instanceof Double) {
			this.meanDouble = (double) mean;
		} else if (mean instanceof String) {
			this.meanDouble = Double.valueOf((String) mean);
		} else if (mean instanceof ArrayList) {
			meanArr = new double[((ArrayList) mean).size()];
			int c = 0;
			for (Object elem : (ArrayList) mean) {
				if (elem instanceof Integer) {
					meanArr[c ++] = Double.valueOf((int) elem);
				} else if (elem instanceof Double) {
					meanArr[c ++] = (double) elem;
				} else if (elem instanceof ArrayList) {
					//TODO allow scaling of more complex structures
					throw new IllegalArgumentException("'mean' parameter cannot be an ArrayList containing"
							+ " another ArrayList. At the moment, only transformations of planes is allowed.");
				} else {
					throw new IllegalArgumentException("If the 'mean' parameter is an array, its elements"
							+ "  have to be instances of" + Integer.class + " or " + Double.class
							+ ". The provided ArrayList contains instances of: " + elem.getClass());
				}
			}
		} else {
			throw new IllegalArgumentException("'mean' parameter has to be either and instance of "
					+ Integer.class + ", " + Double.class + " or " + ArrayList.class 
					+ ". The provided argument is an instance of: " + mean.getClass());
		}
	}
	
	public void setStd(Object std) {
		if (std instanceof Integer) {
			this.stdDouble = Double.valueOf((int) std);
		} else if (std instanceof Double) {
			this.stdDouble = (double) std;
		} else if (std instanceof String) {
			this.stdDouble = Double.valueOf((String) std);
		} else if (std instanceof ArrayList) {
			stdArr = new double[((ArrayList) std).size()];
			int c = 0;
			for (Object elem : (ArrayList) std) {
				if (elem instanceof Integer) {
					stdArr[c ++] = Double.valueOf((int) elem);
				} else if (elem instanceof Double) {
					stdArr[c ++] = (double) elem;
				} else if (elem instanceof ArrayList) {
					throw new IllegalArgumentException("'std' parameter cannot be an ArrayList containing"
							+ " another ArrayList. At the moment, only transformations of planes is allowed.");
				} else {
					throw new IllegalArgumentException("If the 'std' parameter is an array, its elements"
							+ "  have to be instances of" + Integer.class + " or " + Double.class
							+ ". The provided ArrayList contains instances of: " + elem.getClass());
				}
			}
		} else {
			throw new IllegalArgumentException("'std' parameter has to be either and instance of "
					+ Integer.class + ", " + Double.class + " or " + ArrayList.class 
					+ ". The provided argument is an instance of: " + std.getClass());
		}
	}
	
	@SuppressWarnings("unchecked")
	public void setAxes(Object axes) {
		if (axes instanceof String && ((String) axes).equals("channel"))
			this.axes = "c";
		else if (axes instanceof String)
			this.axes = (String) axes;
		else if (axes instanceof List) {
			this.axes = "";
			for (Object ax : (List<Object>) axes) {
				if (!(ax instanceof String))
					throw new IllegalArgumentException("JDLL does not currently support this axes format. Please "
							+ "write an issue attaching the rdf.yaml file at: " + Constants.ISSUES_LINK);
				ax = ax.equals("channel") ? "c" : ax;
				this.axes += ax;
			}
		} else if (axes instanceof String[]) {
			String[] axesArr = (String[]) axes;
			this.axes = "";
			for (String ax : axesArr) {
				ax = ax.equals("channel") ? "c" : ax;
				this.axes += ax;
			}
		} else
			throw new IllegalArgumentException("'axes' parameter has to be an instance of " + String.class
					 + ", of a String array or of a List of Strings. The provided argument is " + axes.getClass());
	}
	
	@SuppressWarnings("unchecked")
	public void setAxis(Object axes) {
		if (axes instanceof String && ((String) axes).equals("channel"))
			this.axes = "c";
		else if (axes instanceof String)
			this.axes = (String) axes;
		else if (axes instanceof List) {
			this.axes = "";
			for (Object ax : (List<Object>) axes) {
				if (!(ax instanceof String))
					throw new IllegalArgumentException("JDLL does not currently support this axes format. Please "
							+ "write an issue attaching the rdf.yaml file at: " + Constants.ISSUES_LINK);
				ax = ax.equals("channel") ? "c" : ax;
				this.axes += ax;
			}
		} else if (axes instanceof String[]) {
			String[] axesArr = (String[]) axes;
			this.axes = "";
			for (String ax : axesArr) {
				ax = ax.equals("channel") ? "c" : ax;
				this.axes += ax;
			}
		} else
			throw new IllegalArgumentException("'axes' parameter has to be an instance of " + String.class
					 + ", of a String array or of a List of Strings. The provided argument is " + axes.getClass());
	}
	
	public void checkRequiredArgs() {
		if (this.mode == Mode.FIXED && this.meanArr == null && this.meanDouble == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, name, "mean")
					+ System.lineSeparator() + "If 'mode' parameter equals 'fixed', the 'mean' "
							+ "argument should be provided too.");
		} else if (this.mode == Mode.FIXED && this.stdArr == null && this.stdDouble == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, name, "std")
					+ System.lineSeparator() + "If 'mode' parameter equals 'fixed', the 'std' "
					+ "argument should be provided too.");
		} else if (this.mode == Mode.FIXED && ((stdDouble == null && meanDouble != null)
				|| (stdDouble != null && meanDouble == null))) {
			throw new IllegalArgumentException("Both arguments 'mean' and "
					+ "'std' need to be of the same type. Either a single value or an array.");
		} else if (this.mode == Mode.FIXED && this.meanArr != null && axes == null) {
			throw new IllegalArgumentException("If 'mean' and 'std' are provided as arrays "
					+ "and 'mode' is 'fixed', the corresponding 'axes' argument should be provided too.");
		}
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		checkRequiredArgs();
		final Tensor< FloatType > output = makeOutput( input );
		applyInPlace(output);
		return output;
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > void applyInPlace( final Tensor< R > input )
	{
		checkRequiredArgs();
		String selectedAxes = "";
		for (String ax : input.getAxesOrderString().split("")) {
			if (axes != null && !axes.toLowerCase().contains(ax.toLowerCase()))
				selectedAxes += ax;
		}
		if (mode == Mode.FIXED &&  (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().length() == selectedAxes.length())) {
			if (meanDouble == null && meanArr == null)
				throw new IllegalArgumentException(FIXED_MODE_ERR);
			else if (meanDouble == null)
				throw new IllegalArgumentException("The parameters 'mean' and 'std' "
						+ "cannot be arrays with the introduced 'axes'.");
			fixedModeGlobalMeanStd(input);
		} else if (mode != Mode.FIXED && (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().length() == selectedAxes.length())) {
			if (meanDouble != null || meanArr != null)
				throw new IllegalArgumentException(NOT_FIXED_MODE_ERR);
			notFixedModeGlobalMeanStd(input);
		} else if (mode != Mode.FIXED 
				&& axes.length() <= 2 && axes.length() > 0) {
			if (meanDouble != null || meanArr != null)
				throw new IllegalArgumentException(NOT_FIXED_MODE_ERR);
			notFixedAxesMeanStd(input, selectedAxes);
		} else if (mode == Mode.FIXED 
				&& axes.length() <= 2 && axes.length() > 0) {
			if (meanDouble == null && meanArr == null)
				throw new IllegalArgumentException(FIXED_MODE_ERR);
			else if (meanDouble != null)
				throw new IllegalArgumentException("The parameters 'mean' and ' std' "
						+ "have to be arrays with the introduced 'axes'.");
			fixedAxesMeanStd(input, selectedAxes);
		} else {
			//TODO allow scaling of more complex structures
			throw new IllegalArgumentException("At the moment, only allowed scaling of planes.");
		}
	}
	
	private < R extends RealType< R > & NativeType< R > > void fixedModeGlobalMeanStd( final Tensor< R > output ) {
		zeroMeanUnitVariance(output.getData(), meanDouble.doubleValue(), stdDouble.doubleValue());
	}
	
	private < R extends RealType< R > & NativeType< R > > void notFixedAxesMeanStd( final Tensor< R > output, String axesOfInterest) {
		long[] start = new long[output.getData().numDimensions()];
		long[] dims = output.getData().dimensionsAsLongArray();
		long[] indOfDims = new long[dims.length - axesOfInterest.length()];
		long[] sizeOfDims = new long[dims.length - axesOfInterest.length()];
		for (int i = 0; i < dims.length; i ++) {
			if (axesOfInterest.indexOf(output.getAxesOrderString().split("")[i]) == -1)
				indOfDims[i] = i;
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
			final float[] meanStd = meanStd( plane );
			final float mean = meanStd[ 0 ];
			final float std = meanStd[ 1 ];
			zeroMeanUnitVariance(output.getData(), mean, std);
		}
	}
	
	private < R extends RealType< R > & NativeType< R > > void fixedAxesMeanStd( final Tensor< R > output, String axesOfInterest) {
		long[] start = new long[output.getData().numDimensions()];
		long[] dims = output.getData().dimensionsAsLongArray();
		long[] indOfDims = new long[dims.length - axesOfInterest.length()];
		long[] sizeOfDims = new long[dims.length - axesOfInterest.length()];
		for (int i = 0; i < dims.length; i ++) {
			if (axesOfInterest.indexOf(output.getAxesOrderString().split("")[i]) == -1)
				indOfDims[i] = i;
		}
		for (int i = 0; i < sizeOfDims.length; i ++) {
			sizeOfDims[i] = dims[(int) indOfDims[i]];
		}
		
		long[][] points = getAllCombinations(sizeOfDims);
		int c = 0;
		for (long[] pp : points) {
			for (int i = 0; i < pp.length; i ++) {
				start[(int) indOfDims[i]] = pp[i];
				dims[(int) indOfDims[i]] = pp[i] + 1;
			}
			// Define the view by defining the length per axis
			long[] end = new long[dims.length];
			for (int i = 0; i < dims.length; i ++) end[i] = dims[i] - start[i];
			IntervalView<R> plane = Views.offsetInterval( output.getData(), start, end );
			final float mean = (float) this.meanArr[c];
			final float std = (float) this.stdArr[c ++ ];
			zeroMeanUnitVariance(plane, mean, std);
		}
	}
	
	private < R extends RealType< R > & NativeType< R > > void notFixedModeGlobalMeanStd( final Tensor< R > output ) {

		final float[] meanStd = meanStd( output.getData() );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];
		zeroMeanUnitVariance(output.getData(), mean, std);
	}

	public static < R extends RealType< R > & NativeType< R > > float[] meanStd( final RandomAccessibleInterval< R > rai )
	{
		// Mean.
		double sum = 0.;
		long n = 0;
		for ( final R p : Views.iterable( rai ) )
		{
			sum += p.getRealDouble();
			n++;
		}
		if ( n < 1 )
			throw new IllegalArgumentException( "Tensor must contain at least 2 pixels, got " + n );

		final double mean = sum / n;

		// Variance.
		double sumdx2 = 0.;
		for ( final R p : Views.iterable( rai ) )
		{
			final double dx = p.getRealDouble() - mean;
			sumdx2 += dx * dx;
		}
		final double variance = sumdx2 /  n ;
		final double std = Math.sqrt( variance );

		return new float[] { ( float ) mean, ( float ) std };
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
		//test1();
		test2();
		test3();
	}
	
	public static void test1() {
		float[] arr = new float[9];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		FixedZeroMeanUnitVarianceTransformation preprocessing = new FixedZeroMeanUnitVarianceTransformation();
		preprocessing.setMean(4);
		preprocessing.setStd(4);
		preprocessing.setMode("fixed");
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {3, 3});
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
		FixedZeroMeanUnitVarianceTransformation preprocessing = new FixedZeroMeanUnitVarianceTransformation();
		preprocessing.setAxes("xy");
		preprocessing.setMode("per_sample");
		Tensor<FloatType> tt = Tensor.build("name", "xyc", rai);
		preprocessing.applyInPlace(tt);
		System.out.print(true);
	}
	
	public static void test3() {
		float[] arr = new float[9];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {1, 1, 3, 3});
		FixedZeroMeanUnitVarianceTransformation preprocessing = new FixedZeroMeanUnitVarianceTransformation();
		preprocessing.setAxes("y");
		preprocessing.setMode("fixed");
		preprocessing.setMean(new double[] {1, 4, 7});
		preprocessing.setStd(new double[] {0.81650, 0.81650, 0.81650});
		Tensor<FloatType> tt = Tensor.build("name", "bcyx", rai);
		preprocessing.applyInPlace(tt);
		System.out.print(true);
	}
	
	public < R extends RealType< R > & NativeType< R > > 
	void zeroMeanUnitVariance(RandomAccessibleInterval<R> rai, double mean, double std) {
        R type = Util.getTypeFromInterval(rai);
        if (type instanceof IntegerType) {
			LoopBuilder.setImages( rai )
			.multiThreaded()
			.forEachPixel( i -> i.setReal(Math.floor((i.getRealDouble() - mean) / (std + eps)) ) );
        } else {
			LoopBuilder.setImages( rai )
			.multiThreaded()
			.forEachPixel( i -> i.setReal(((i.getRealDouble() - mean) / (std + eps)) ) );
        }
	}
}
