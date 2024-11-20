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

import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

public abstract class AbstractTensorTransformation implements TensorTransformation
{

	private final String name;

	private Mode mode = Mode.FIXED;
	
	protected static String DEFAULT_MISSING_ARG_ERR = "Cannot execute %s BioImage.io transformation because '%s' "
			+ "parameter was not set.";

	protected AbstractTensorTransformation( final String name )
	{
		this.name = name;
	}

	@Override
	public String getName()
	{
		return name;
	}

	@Override
	public void setMode( final Mode mode )
	{
		this.mode = mode;
	}

	@Override
	public Mode getMode()
	{
		return mode;
	}

	protected < R extends RealType< R > & NativeType< R > > Tensor< FloatType > makeOutput( final Tensor< R > input )
	{
		final ImgFactory< FloatType > factory = Util.getArrayOrCellImgFactory( input.getData(), new FloatType() );
		final Img< FloatType > outputImg = factory.create( input.getData() );
		RealTypeConverters.copyFromTo(input.getData(), outputImg);
		// TODO what name final Tensor< FloatType > output = Tensor.build( getName() + '_' + input.getName(), input.getAxesOrderString(), outputImg );
		final Tensor< FloatType > output = Tensor.build( input.getName(), input.getAxesOrderString(), outputImg );
		return output;
	}
}
