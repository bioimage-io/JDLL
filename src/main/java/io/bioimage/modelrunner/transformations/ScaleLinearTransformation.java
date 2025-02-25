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
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class ScaleLinearTransformation extends AbstractTensorTransformation
{

	private static final String name = "scale_linear";
	private Double gainDouble = 1d;
	private Double offsetDouble = 0d;
	private double[] gainArr;
	private double[] offsetArr;
	private String axes;
	private double eps = Math.pow(10, -6);

	public ScaleLinearTransformation()
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
	
	public void setGain(Object gain) {
		if (gain instanceof Integer) {
			this.gainDouble = Double.valueOf((int) gain);
		} else if (gain instanceof Double) {
			this.gainDouble = (double) gain;
		} else if (gain instanceof String) {
			this.gainDouble = Double.valueOf((String) gain);
		} else if (gain instanceof ArrayList) {
			gainArr = new double[((ArrayList) gain).size()];
			int c = 0;
			for (Object elem : (ArrayList) gain) {
				if (elem instanceof Integer) {
					gainArr[c ++] = Double.valueOf((int) elem);
				} else if (elem instanceof Double) {
					gainArr[c ++] = (double) elem;
				} else if (elem instanceof ArrayList) {
					throw new IllegalArgumentException("'gain' parameter cannot be an ArrayList containing"
							+ " another ArrayList. At the moment, only transformations of planes is allowed.");
				} else {
					//TODO allow scaling of more complex structures
					throw new IllegalArgumentException("If the 'gain' parameter is an array, its elements"
							+ "  have to be instances of" + Integer.class + " or " + Double.class
							+ ". The provided ArrayList contains instances of: " + elem.getClass());
				}
			}
		} else {
			throw new IllegalArgumentException("'gain' parameter has to be either and instance of "
					+ Integer.class + ", " + Double.class + " or " + ArrayList.class 
					+ ". The provided argument is an instance of: " + gain.getClass());
		}
	}
	
	public void setOffset(Object offset) {
		if (offset instanceof Integer) {
			this.offsetDouble = Double.valueOf((int) offset);
		} else if (offset instanceof Double) {
			this.offsetDouble = (double) offset;
		} else if (offset instanceof String) {
			this.offsetDouble = Double.valueOf((String) offset);
		} else if (offset instanceof ArrayList) {
			offsetArr = new double[((ArrayList) offset).size()];
			int c = 0;
			for (Object elem : (ArrayList) offset) {
				if (elem instanceof Integer) {
					offsetArr[c ++] = Double.valueOf((int) elem);
				} else if (elem instanceof Double) {
					offsetArr[c ++] = (double) elem;
				} else if (elem instanceof ArrayList) {
					throw new IllegalArgumentException("'offset' parameter cannot be an ArrayList containing"
							+ " another ArrayList. At the moment, only transformations of planes is allowed.");
				} else {
					//TODO allow scaling of more complex structures
					throw new IllegalArgumentException("If the 'offset' parameter is an array, its elements"
							+ "  have to be instances of" + Integer.class + " or " + Double.class
							+ ". The provided ArrayList contains instances of: " + elem.getClass());
				}
			}
		} else {
			throw new IllegalArgumentException("'offset' parameter has to be either and instance of "
					+ Integer.class + ", " + Double.class + " or " + ArrayList.class 
					+ ". The provided argument is an instance of: " + offset.getClass());
		}
	}
	
	@SuppressWarnings("unchecked")
	public void setAxes(Object axes) {
		if (axes instanceof String )
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
		if (offsetDouble == null && offsetArr == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, name, "offset"));
		} else if (gainDouble == null && gainArr == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, name, "gain"));
		} else if ((offsetDouble == null && gainDouble != null)
				|| (offsetDouble != null && gainDouble == null)) {
			throw new IllegalArgumentException("Both arguments 'gain' and "
					+ "'offset' need to be of the same type. Either a single value or an array.");
		} else if (offsetArr != null && axes == null) {
			throw new IllegalArgumentException("If 'offset' and 'gain' are provided as arrays, "
					+ "the corresponding 'axes' argument should be provided too.");
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
	public < R extends RealType< R > & NativeType< R > > void applyInPlace(Tensor<R> input) {
		checkRequiredArgs();
		String selectedAxes = "";
		for (String ax : input.getAxesOrderString().split("")) {
			if (axes != null && !axes.toLowerCase().contains(ax.toLowerCase())
					&& !ax.toLowerCase().equals("b"))
				selectedAxes += ax;
		}
		if (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().replace("b", "").length() == selectedAxes.length()) {
			if (gainDouble == null)
				throw new IllegalArgumentException("The 'axes' parameter is not"
						+ " compatible with the parameters 'gain' and 'offset'."
						+ "The parameters gain and offset cannot be arrays with"
						+ " the given axes parameter provided.");
			globalScale(input);
		} else if (axes.length() <= 2 && axes.length() > 0) {
			axesScale(input, selectedAxes);
		} else {
			//TODO allow scaling of more complex structures
			throw new IllegalArgumentException("At the moment, only allowed scaling of planes.");
		}
		
	}
	
	private < R extends RealType< R > & NativeType< R > > void globalScale( final Tensor< R > output ) {
		scaleLinear(output.getData(), gainDouble.doubleValue(), offsetDouble.doubleValue());
	}
	
	private < R extends RealType< R > & NativeType< R > > void axesScale( final Tensor< R > output, String axesOfInterest) {
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
			final float gain = (float) (this.gainArr != null ? gainArr[c] : this.gainDouble);
			final float offset = (float) (this.offsetArr != null ? offsetArr[c] : this.offsetDouble);
			scaleLinear(plane, gain, offset);
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
	
	public < R extends RealType< R > & NativeType< R > > 
	void scaleLinear(RandomAccessibleInterval<R> rai, double gain, double offset) {

        R type = Util.getTypeFromInterval(rai);
        if (type instanceof IntegerType) {
			LoopBuilder.setImages( rai )
			.multiThreaded()
			.forEachPixel( i -> i.setReal(Math.floor(i.getRealDouble() * gain + offset) ) );
        } else {
			LoopBuilder.setImages( rai )
			.multiThreaded()
			.forEachPixel( i -> i.setReal((i.getRealDouble() * gain + offset) ) );
        }
	}
}
