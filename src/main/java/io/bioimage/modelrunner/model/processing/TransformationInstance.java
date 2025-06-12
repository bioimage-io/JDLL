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
/**
 * 
 */
package io.bioimage.modelrunner.model.processing;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apposed.appose.Types;
import io.bioimage.modelrunner.bioimageio.description.TransformSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.transformations.BinarizeTransformation;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;

/**
 * Class that creates an instance able to run the corresponding Bioimage.io processing routine
 * @author Carlos Jaier Garcia Lopez de Haro
 */

public class TransformationInstance {
	private final String name;
	private final Map<String, Object> args;
	private Class<?> cls;
	private Object instance;
	/**
	 * Package where the BioImage.io transformations are.
	 */
	private final static String TRANSFORMATIONS_PACKAGE = BinarizeTransformation.class.getPackage().getName();
	
	private final static String RUN_NAME = "apply";
	
	private final static String RUN_INPLACE_NAME = "applyInPlace";
	
	protected TransformationInstance(TransformSpec transform) throws RuntimeException, IllegalArgumentException {
		this.name = transform.getName();
		this.args = transform.getKwargs();
		this.build();
	}
	
	/**
	 * Create a {@link TransformationInstance} from a {@link TransformSpec} created from a valid rdf.yaml Bioimage.io
	 * spec file
	 * @param transform
	 * 	{@link TransformSpec} object from an rd.yaml file
	 * @return the {@link TransformationInstance}
	 * @throws RuntimeException if there is any error because the transformation defined by {@link TransformSpec} is not
	 * 	valid or not yet supported
	 * @throws IllegalArgumentException if there is any error because the transformation defined by {@link TransformSpec} is not
	 * 	valid or not yet supported
	 */
	public static TransformationInstance create(TransformSpec transform) throws RuntimeException, IllegalArgumentException {
		return new TransformationInstance(transform);
	}
	
	/**
	 * Run the defined transformation on the input {@link Tensor} of interest.
	 * This method creates a new object for the output tensor, so at the end, 
	 * there is one object for the input and another for the output.
	 * If you want to do the transfromation in-place (modify the input tensor 
	 * instead of creating another one) use {@link #run(Tensor, boolean)}
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the input tensor
	 * @param <R>
	 * 	ImgLib2 data type of the resulting output tensor
	 * @param tensor
	 * 	the input tensor to be processed
	 * @return the output tensor
	 * @throws RuntimeException if there is any error running the transformation
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> run(Tensor<T> tensor) throws RuntimeException {
		return run(tensor, false);
	}
	
	/**
	 * Run the defined transformation on the input {@link Tensor} of interest.
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the input tensor
	 * @param <R>
	 * 	ImgLib2 data type of the resulting output tensor
	 * @param tensor
	 * 	the input tensor to be processed
	 * @param inplace
	 * 	whether to apply the transformation to the input object and modify it or 
	 * 	to create a separate tensor as the output and do the modifications there.
	 * 	With inplace=false, two separate tensors exist after the method is done.
	 * @return the output tensor
	 * @throws RuntimeException if there is any error running the transformation
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> run(Tensor<T> tensor, boolean inplace) throws RuntimeException {
		Method m;
		try {
			if (inplace) {
	            m = cls.getMethod(RUN_INPLACE_NAME, Tensor.class);
	            m.invoke(this.instance, tensor);
	            return Collections.singletonList(Cast.unchecked(tensor));
	        } else {
	            m = cls.getMethod(RUN_NAME, Tensor.class);
	            Object result = m.invoke(this.instance, tensor);
	            
	            // Handle different possible return types
	            if (result == null) {
	                return null;
	            } else if (result instanceof List<?>) {
	                // Cast and verify each element is a Tensor<R>
	                List<?> resultList = (List<?>) result;
	                List<Tensor<R>> outputList = new ArrayList<>();
	                
	                for (Object item : resultList) {
	                    if (item instanceof Tensor<?>) {
	                        @SuppressWarnings("unchecked")
	                        Tensor<R> tensorItem = (Tensor<R>) item;
	                        outputList.add(tensorItem);
	                    } else {
	                        throw new RuntimeException("Invalid return type: Expected Tensor but got " + 
	                            (item != null ? item.getClass().getName() : "null"));
	                    }
	                }
	                return outputList;
	            } else if (result instanceof Tensor<?>) {
	                // Single Tensor result
	                @SuppressWarnings("unchecked")
	                Tensor<R> tensorResult = (Tensor<R>) result;
	                return Collections.singletonList(tensorResult);
	            } else {
	                throw new RuntimeException("Unexpected return type: " + 
	                    (result != null ? result.getClass().getName() : "null"));
	            }
	        }
		} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException 
				| NoSuchMethodException | SecurityException e) {
			throw new RuntimeException(Types.stackTrace(e));
		}
	}
	
	private void build() {
		getTransformationClass();
		createInstanceWithArgs();
	}
	
	/**
	 * Find of the transformation exists in the BioImage.io Java Core
	 * @throws ClassNotFoundException if the BioImage.io transformation does not exist
	 */
	private void getTransformationClass() throws RuntimeException {
		String javaMethodName = snakeCaseToCamelCaseFirstCap(this.name) + "Transformation";
		String clsName = TRANSFORMATIONS_PACKAGE + "." + javaMethodName;
		findClassInClassPath(clsName);
		try {
			this.cls = getClass().getClassLoader().loadClass(clsName);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(Types.stackTrace(e));
		}
	}
	
	/**
	 * Tries to find a given class in the classpath
	 * @throws ClassNotFoundException if the class does not exist in the classpath
	 */
	private void findClassInClassPath(String clsName) throws IllegalArgumentException {
		try {
			Class.forName(clsName, false, TransformationInstance.class.getClassLoader());
		} catch (ClassNotFoundException e) {
			throw new IllegalArgumentException("Invalid method '" + this.name + "' in the specs file. The method does not"
					+ " exist in the Bioimage.io framework.");
		}
	}
	
	/**
	 * Method used to convert Strings in using snake case (snake_case) into camel
	 * case with the first letter as upper case (CamelCase)
	 * @param str
	 * 	the String to be converted
	 * @return String converted into camel case with first upper
	 */
	public static String snakeCaseToCamelCaseFirstCap(String str) {
		while(str.contains("_")) {
            str = str.replaceFirst("_[a-z]", String.valueOf(Character.toUpperCase(str.charAt(str.indexOf("_") + 1))));
        }
		str = str.substring(0, 1).toUpperCase() + str.substring(1);
		return str;
	}
	
	private void createInstanceWithArgs() throws RuntimeException {
		try {
			this.instance = this.cls.getConstructor().newInstance();
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			throw new RuntimeException(Types.stackTrace(e));
		}
		if (args == null)
			return;
		for (String kk : args.keySet()) {
			setArg(kk);
		}
	}
	
	/**
	 * Set the argument in the processing trasnformation instance
	 * @param instance
	 * 	instance of the processing trasnformation
	 * @param argName
	 * 	name of the argument
	 * @throws IllegalArgumentException if no method is found for the given argument
	 * @throws InvocationTargetExceptionif there is any error invoking the method
	 * @throws IllegalAccessException if it is illegal to access the method
	 */
	private void setArg(String argName) {
		Method mm = getMethodForArgument(argName);
		checkArgType(mm);
		try {
			mm.invoke(instance, this.args.get(argName));
		} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
			throw new RuntimeException(Types.stackTrace(e));
		}	
	}
	
	/**
	 * Get the setter that the Java transformation class uses to set the argument of the
	 * pre-processing. The setter has to be named as the argument but in CamelCase with the
	 * first letter in upper case and preceded by set. For example: min_distance -> setMinDistance
	 * @param argName
	 * 	the name of the argument
	 * @return the method name 
	 * @throws IllegalArgumentException if no method is found for the given argument
	 */
	private Method getMethodForArgument(String argName) throws IllegalArgumentException {
		String mName = "set" + snakeCaseToCamelCaseFirstCap(argName);
		// Check that the method exists
		Method[] methods = this.cls.getMethods();
		for (Method mm : methods) {
			if (mm.getName().equals(mName))
				return mm;
		}
		throw new IllegalArgumentException("Setter for argument '" + argName + "' of the processing "
				+ "transformation '" + name + "' not found in the Java transformation class '" + this.cls + "'. "
				+ "A method called '" + mName + "' should be present.");
	}
	
	/**
	 * Method that checks that the type of the arguments provided in the rdf.yaml is correct.
	 * It also returns the setter method to set the argument
	 * 
	 * @param mm
	 * 	the method that executes the pre-processing transformation
	 * @return the method used to provide the argument to the instance
	 * @throws IllegalArgumentException if any of the arguments' type is not correct
	 */
	private void checkArgType(Method mm) throws IllegalArgumentException {
		Parameter[] pps = mm.getParameters();
		if (pps.length == 1 && pps[0].getType() == Object.class)
			return;
		throw new IllegalArgumentException("Setter '" + mm.getName() + "' should have only one input parameter with type Object.class.");
	}

}
