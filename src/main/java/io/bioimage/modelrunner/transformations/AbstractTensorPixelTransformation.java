/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.transformations;

import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Abstract classes for tensor transformations where a new pixel value can be
 * calculated solely from the corresponding pixel value in the input. This
 * mapping is specified by a
 *
 * @author Jean-Yves Tinevez
 *
 */
public class AbstractTensorPixelTransformation extends AbstractTensorTransformation
{

	private FloatUnaryOperator fun;

	protected AbstractTensorPixelTransformation( final String name)
	{
		super( name );
	}
	
	protected void setFloatUnitaryOperator(final FloatUnaryOperator fun) {
		this.fun = fun;
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		final Tensor< FloatType > output = makeOutput( input );
		LoopBuilder
				.setImages( input.getData(), output.getData() )
				.multiThreaded()
				.forEachPixel( ( i, o ) -> o.set( fun.applyAsFloat( i.getRealFloat() ) ) );
		return output;
	}

	@Override
	public void applyInPlace( final Tensor< FloatType > input )
	{
		LoopBuilder
				.setImages( input.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( fun.applyAsFloat( i.get() ) ) );
	}

	@FunctionalInterface
	public interface FloatUnaryOperator
	{
		float applyAsFloat( float in );
	}
}
