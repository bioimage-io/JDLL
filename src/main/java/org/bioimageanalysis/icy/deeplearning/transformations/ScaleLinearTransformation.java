package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class ScaleLinearTransformation extends AbstractTensorTransformation
{

	private static final String name = "scale_linear";
	private Double gainDouble;
	private Double offsetDouble;
	private double[] gainArr;
	private double[] offsetArr;
	private String axes;

	public ScaleLinearTransformation()
	{
		super( name );
	}
	
	public void setGain(double gain) {
		this.gainDouble = gain;
	}
	
	public void setGain(double[] gain) {
		this.gainArr = gain;
	}
	
	public void setGain(double[][] gain) {
		//TODO allow scaling of more complex structures
		throw new IllegalArgumentException("At the moment, only allowed scaling of planes.");
	}
	
	public void setOffset(double offset) {
		this.offsetDouble = offset;
	}
	
	public void setOffset(double[] offset) {
		this.offsetArr = offset;
	}
	
	public void setOffset(double[][] offset) {
		//TODO allow scaling of more complex structures
		throw new IllegalArgumentException("At the moment, only allowed scaling of planes.");
	}
	
	public void setAxes(String axes) {
		this.axes = axes;
	}
	
	public void checkRequiredArgs() {
		if (offsetDouble == null && offsetArr == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "offset"));
		} else if (gainDouble == null && gainArr == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "gain"));
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
	public void applyInPlace(Tensor<FloatType> input) {
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
	
	private void globalScale( final Tensor< FloatType > output ) {
		LoopBuilder.setImages( output.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( gainDouble.floatValue() * i.get() + offsetDouble.floatValue() ) );
	}
	
	private void axesScale( final Tensor< FloatType > output, String axesOfInterest) {
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
			IntervalView<FloatType> plane = Views.offsetInterval( output.getData(), start, end );
			final float gain = (float) this.gainArr[c];
			final float offset = (float) this.offsetArr[c ++ ];
			LoopBuilder.setImages( plane )
					.multiThreaded()
					.forEachPixel( i -> i.set( i.get() * gain + offset ) );
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
}
