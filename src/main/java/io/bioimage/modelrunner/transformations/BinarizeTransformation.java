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
package io.bioimage.modelrunner.transformations;

import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class BinarizeTransformation extends AbstractTensorPixelTransformation
{
	
	private static String name = "binarize";
	private Double threshold;
	private Double eps = Math.pow(10, -6);
	

	public BinarizeTransformation()
	{
		super( name );
		super.setDoubleUnitaryOperator(v -> ( v >= threshold ) ? 1d : 0d);
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
	
	public void setThreshold(Object threshold) {
		if (threshold instanceof Integer) {
			this.threshold = Double.valueOf((int) threshold);
		} else if (threshold instanceof Double) {
			this.threshold = (double) threshold;
		} else if (threshold instanceof String) {
			this.threshold = Double.valueOf((String) threshold);
		} else {
			throw new IllegalArgumentException("'threshold' parameter has to be either and instance of "
					+ Integer.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + threshold.getClass());
		}
	}
	
	public void checkRequiredArgs() {
		if (threshold == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, name, "threshold"));
		}
	}

	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		checkRequiredArgs();
		return super.apply(input);
	}

	public < R extends RealType< R > & NativeType< R > >  void applyInPlace( final Tensor< R > input )
	{
		checkRequiredArgs();
		super.applyInPlace(input);
	}
}
