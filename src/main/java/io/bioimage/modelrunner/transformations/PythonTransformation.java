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

import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class PythonTransformation extends AbstractTensorPixelTransformation
{
	private static String name = "python";
	
	private String envYaml;
	
	private String script;
	
	private String method;
	
	private List<Object> args;

	public PythonTransformation()
	{
		super(name);
	}
	
	public void setEnvYaml(Object envYaml) {
		if (envYaml instanceof String) {
			this.envYaml = String.valueOf(envYaml);
		} else {
			throw new IllegalArgumentException("'envYaml' parameter has to be either and instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setScript(Object script) {
		if (script instanceof String) {
			this.script = String.valueOf(script);
		} else {
			throw new IllegalArgumentException("'script' parameter has to be either and instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + script.getClass());
		}
	}
	
	public void setMethod(Object method) {
		if (method instanceof String) {
			this.method = String.valueOf(method);
		} else {
			throw new IllegalArgumentException("'method' parameter has to be either and instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + method.getClass());
		}
	}

	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		super.setFloatUnitaryOperator( v -> ( float ) ( 1. / ( 1. + Math.exp( -v ) ) ) );
		return super.apply(input);
	}

	public void applyInPlace( final Tensor< FloatType > input )
	{
		super.setFloatUnitaryOperator( v -> ( float ) ( 1. / ( 1. + Math.exp( -v ) ) ) );
		super.apply(input);
	}
}
