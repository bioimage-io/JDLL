package io.bioimage.modelrunner.model.processing;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.TransformSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.transformations.BinarizeTransformation;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

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
	
	protected TransformationInstance(TransformSpec transform) {
		this.name = transform.getName();
		this.args = transform.getKwargs();
		this.build();
	}
	
	public static TransformationInstance create(TransformSpec transform) {
		return new TransformationInstance(transform);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> run(Tensor<T> tensor){
		return run(tensor, false);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> run(Tensor<T> tensor, boolean inplace) {
		Method m = cls.getMethod(RUN_NAME, List.class);
		m.invoke(this.instance, tensor);
		return null;
	}
	
	private void build() {
		getTransformationClass();
		createInstanceWithArgs();
	}
	
	/**
	 * Find of the transformation exists in the BioImage.io Java Core
	 * @throws ClassNotFoundException if the BioImage.io transformation does not exist
	 */
	private void getTransformationClass() throws ClassNotFoundException {
		String javaMethodName = snakeCaseToCamelCaseFirstCap(this.name) + "Transformation";
		String clsName = TRANSFORMATIONS_PACKAGE + "." + javaMethodName;
		findClassInClassPath(clsName);
		this.cls = getClass().getClassLoader().loadClass(clsName);
	}
	
	/**
	 * Tries to find a given class in the classpath
	 * @throws ClassNotFoundException if the class does not exist in the classpath
	 */
	private void findClassInClassPath(String clsName) throws ClassNotFoundException {
		Class.forName(clsName, false, JavaProcessing.class.getClassLoader());
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
	
	private void createInstanceWithArgs() throws InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException, ClassNotFoundException {
		this.instance = this.cls.getConstructor().newInstance();

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
	public void setArg(String argName) throws IllegalArgumentException, IllegalAccessException, InvocationTargetException {
		Method mm = getMethodForArgument(argName);
		checkArgType(mm);
		mm.invoke(instance, this.args.get(argName));	
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
	public Method getMethodForArgument(String argName) throws IllegalArgumentException {
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
