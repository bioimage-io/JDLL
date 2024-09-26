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
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
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
	private DoubleUnaryOperator dun;
	private ByteUnaryOperator bun;
	private UByteUnaryOperator ubun;
	private ShortUnaryOperator sun;
	private UShortUnaryOperator usun;
	private IntUnaryOperator iun;
	private UIntUnaryOperator uiun;
	private LongUnaryOperator lun;

	protected AbstractTensorPixelTransformation( final String name)
	{
		super( name );
	}
	
	protected void setFloatUnitaryOperator(final FloatUnaryOperator fun) {
		this.fun = fun;
	}
	
	protected void setDoubleUnitaryOperator(final DoubleUnaryOperator fun) {
		this.dun = fun;
	}
	
	protected void setByteUnitaryOperator(final ByteUnaryOperator fun) {
		this.bun = fun;
	}
	
	protected void setUByteUnitaryOperator(final UByteUnaryOperator fun) {
		this.ubun = fun;
	}
	
	protected void setShortUnitaryOperator(final ShortUnaryOperator fun) {
		this.sun = fun;
	}
	
	protected void setUShortUnitaryOperator(final UShortUnaryOperator fun) {
		this.usun = fun;
	}
	
	protected void setIntUnitaryOperator(final IntUnaryOperator fun) {
		this.iun = fun;
	}
	
	protected void setUIntUnitaryOperator(final UIntUnaryOperator fun) {
		this.uiun = fun;
	}
	
	protected void setLongUnitaryOperator(final LongUnaryOperator fun) {
		this.lun = fun;
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		final Tensor< FloatType > output = makeOutput( input );
		applyInPlace(output);
		return output;
	}

	@SuppressWarnings("unchecked")
	@Override
	public < R extends RealType< R > & NativeType< R > >
	void applyInPlace( final Tensor< R > input )
	{
		if (input.getData().getAt(0) instanceof FloatType && fun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<FloatType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( fun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof DoubleType && dun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<DoubleType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( dun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof ByteType && bun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<ByteType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( bun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof UnsignedByteType && ubun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<UnsignedByteType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( ubun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof ShortType && sun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<ShortType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( sun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof UnsignedShortType && usun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<UnsignedShortType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( usun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof IntType && iun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<IntType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( iun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof UnsignedIntType && uiun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<UnsignedIntType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( uiun.applyAs( i.get() ) ) );
		} else if (input.getData().getAt(0) instanceof LongType && lun != null) {
			LoopBuilder
			.setImages( (RandomAccessibleInterval<LongType>) input.getData() )
			.multiThreaded()
			.forEachPixel( i -> i.set( lun.applyAs( i.get() ) ) );
		} else {
			throw new IllegalArgumentException("Unsupported data type.");
		}
	}

	@FunctionalInterface
	public interface FloatUnaryOperator
	{
		float applyAs( float in );
	}

	@FunctionalInterface
	public interface DoubleUnaryOperator
	{
		double applyAs( double in );
	}

	@FunctionalInterface
	public interface ByteUnaryOperator
	{
		byte applyAs( byte in );
	}

	@FunctionalInterface
	public interface UByteUnaryOperator
	{
		int applyAs( int i );
	}

	@FunctionalInterface
	public interface ShortUnaryOperator
	{
		short applyAs( short in );
	}

	@FunctionalInterface
	public interface UShortUnaryOperator
	{
		int applyAs( int i );
	}

	@FunctionalInterface
	public interface IntUnaryOperator
	{
		int applyAs( int in );
	}

	@FunctionalInterface
	public interface UIntUnaryOperator
	{
		long applyAs( long in );
	}

	@FunctionalInterface
	public interface LongUnaryOperator
	{
		long applyAs( long in );
	}
}
