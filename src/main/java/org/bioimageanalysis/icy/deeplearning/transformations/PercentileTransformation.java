package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

public class PercentileTransformation extends DefaultImageTransformation
{

	public static final String name = "percentile";

	private Number minPercentile;

	private Number maxPercentile;

	public Number getMinPercentile()
	{
		return minPercentile;
	}

	public void setMinPercentile( Number minPercentile )
	{
		this.minPercentile = minPercentile;
	}

	public Number getMaxPercentile()
	{
		return maxPercentile;
	}

	public void setMaxPercentile( Number maxPercentile )
	{
		this.maxPercentile = maxPercentile;
	}

	@Override
	public String getName()
	{
		return name;
	}

	private Tensor tensor;

	public PercentileTransformation( Tensor tensor )
	{
		this.tensor = tensor;
	}

	public Tensor apply()
	{
		return tensor;
	}
}
