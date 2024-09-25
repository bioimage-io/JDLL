package io.bioimage.modelrunner.model.processing;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.TransformSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.transformations.BinarizeTransformation;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Class that executes the pre-processing associated to a given tensor
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class Processing {
	/**
	 * Descriptor containing the info about the model
	 */
	private ModelDescriptor descriptor;
	/**
	 * Specifications of the tensor of interest
	 */
	private TensorSpec tensorSpec;
	/**
	 * Map containing all the needed input objects to make the processing.
	 * It has to contain the tensor of interest.
	 */
	private LinkedHashMap<String, Object> inputsMap;
	/**
	 * List containing the names of the processings that need to be applied
	 * to the tensor image
	 */
	private List<TransformSpec> processing;
	// TODO when adding python
	//private static BioImageIoPython interp;
	private static String BIOIMAGEIO_PYTHON_TRANSFORMATIONS_WEB = 
						"https://github.com/bioimage-io/core-bioimage-io-python/blob/b0cea"
						+ "c8fa5b412b1ea811c442697de2150fa1b90/bioimageio/core/prediction_pipeline"
						+ "/_processing.py#L105";
	/**
	 * Package where the BioImage.io transformations are.
	 */
	private final static String TRANSFORMATIONS_PACKAGE = BinarizeTransformation.class.getPackage().getName();

	/**
	 * The object that is going to execute processing on the given image
	 * @param tensorSpec
	 * 	the tensor specifications
	 * @param seq
	 * 	the image corresponding to a tensor where processing is going to be executed
	 */
	private Processing(ModelDescriptor descriptor) {
		this.descriptor = descriptor;
	}
	
	private void buildPreprocessing() throws ClassNotFoundException {
		Map<String, List<Map<String, Object>>> preMap = new HashMap<String, List<Map<String, Object>>>();
		for (TensorSpec tt : this.descriptor.getInputTensors()) {
			List<TransformSpec> preprocessing = tt.getPreprocessing();
			List<Map<String, Object>> list = new ArrayList<Map<String, Object>>();
			for (TransformSpec transformation : preprocessing) {
				Map<String, Object> map = new HashMap<String, Object>();
				String clsName = findMethodInBioImageIo(transformation.getName());
			}
		}
	}
	
	public static Processing init(ModelDescriptor descriptor) {
		return new Processing(descriptor);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> process(List<Tensor<T>> tensorList){
		return process(tensorList, false);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> process(List<Tensor<T>> tensorList, boolean inplace) {
		
		return null;
	}
	
	/**
	 * Execute processing defined with Java.
	 * @param transformation
	 * 	the name of the class that is going to be executed
	 * @param kwargs
	 * 	the args of the transformation to be executed
	 * @throws ClassNotFoundException if the Java processing class is not found in the loaded classes
	 * @throws IOException if there is any issue initializing the Python interpreter
	 * @throws InterruptedException 
	 * @throws IllegalArgumentException 
	 */
	public < T extends RealType< T > & NativeType< T > > void executeJavaProcessing(String transformation, Map<String, Object> kwargs) throws ClassNotFoundException, IOException, IllegalArgumentException, InterruptedException {
		try {
			JavaProcessing preproc = JavaProcessing.definePreprocessing(transformation, kwargs);
			inputsMap = preproc.execute(tensorSpec, inputsMap);
			return;
		} catch (ClassNotFoundException ex) {
			// TODO when adding python
			//System.out.println("Executing processing transformation '" + transformation + "' with Python.");
			throw new IOException("Error running processing transformation: " + transformation);
		}
		// If class not found, execute in Python
		// TODO when adding python
		//executePythonProcessing(transformation, kwargs);
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
	
	/**
	 * Tries to find a given class in the classpath
	 * @throws ClassNotFoundException if the class does not exist in the classpath
	 */
	private void findClassInClassPath(String clsName) throws ClassNotFoundException {
		Class.forName(clsName, false, JavaProcessing.class.getClassLoader());
	}
	
	/**
	 * Find of the transformation exists in the BioImage.io Java Core
	 * @throws ClassNotFoundException if the BioImage.io transformation does not exist
	 */
	private String findMethodInBioImageIo(String methodName) throws ClassNotFoundException {
		String javaMethodName = snakeCaseToCamelCaseFirstCap(methodName) + "Transformation";
		String clsName = TRANSFORMATIONS_PACKAGE + "." + javaMethodName;
		findClassInClassPath(clsName);
		return clsName;
	}
	private LinkedHashMap<String, Object> runJavaTransformationWithArgs(String clsName, Map<String, Object> args) throws InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException, ClassNotFoundException {
		Class<?> transformationClass = getClass().getClassLoader().loadClass(clsName);
		Object transformationObject = transformationClass.getConstructor().newInstance();

		for (String arg : args.keySet()) {
			setArg(transformationObject, arg);
		}
		Method[] publicMethods = transformationClass.getMethods();
		Method transformationMethod = null;
		for (Method mm : publicMethods) {
			if (mm.getName().equals(this.javaMethodName)) {
				transformationMethod = mm;
				break;
			}
		}
		if (transformationMethod == null)
			throw new IllegalArgumentException("The pre-processing transformation class does not contain"
					+ "the method '" + this.javaMethodName + "' needed to call the transformation.");
		// Check that the arguments specified in the rdf.yaml are of the corect type
		return null;
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
	public void setArg(Object instance, String argName) throws IllegalArgumentException, IllegalAccessException, InvocationTargetException {
		String mName = getArgumentSetterName(argName);
		Method mm = checkArgType(argName, mName);
		mm.invoke(instance, args.get(argName));	
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
	public String getArgumentSetterName(String argName) throws IllegalArgumentException {
		String mName = "set" + snakeCaseToCamelCaseFirstCap(argName);
		// Check that the method exists
		Method[] methods = transformationClass.getMethods();
		for (Method mm : methods) {
			if (mm.getName().equals(mName))
				return mName;
		}
		throw new IllegalArgumentException("Setter for argument '" + argName + "' of the processing "
				+ "transformation '" + rdfSpec + "' of tensor '" + tensorName
				+ "' not found in the Java transformation class '" + this.javaClassName + "'. "
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
	private Method checkArgType(String argName, String mName) throws IllegalArgumentException {
		Object arg = this.args.get(argName);
		Method[] methods = this.transformationClass.getMethods();
		List<Method> possibleMethods = new ArrayList<Method>();
		for (Method mm : methods) {
			if (mm.getName().equals(mName)) 
				possibleMethods.add(mm);
		}
		if (possibleMethods.size() == 0)
			getArgumentSetterName(argName);
		for (Method mm : possibleMethods) {
			Parameter[] pps = mm.getParameters();
			if (pps.length != 1) {
				continue;
			}
			if (pps[0].getType() == Object.class)
				return mm;
		}
		throw new IllegalArgumentException("Setter '" + mName + "' should have only one input parameter with type Object.class.");
	}
}
	
	// TODO when adding python
	/**
	 public < T extends RealType< T > & NativeType< T > > void executePythonProcessing(String transformation, Map<String, Object> kwargs) throws IOException, IllegalArgumentException, InterruptedException {
		Objects.requireNonNull(transformation, "The Python transformation needs to be a 'bioimageio.core' transformatio at " + BIOIMAGEIO_PYTHON_TRANSFORMATIONS_WEB);
		Objects.requireNonNull(kwargs);
		PythonUtils pUtils = getPythonConfiguration(transformation);
		try (BioImageIoPython python = BioImageIoPython.activate(JepUtils.createNewPythonInstance(pUtils))){
			Tensor<T> javaTensor;
			if (inputsMap.get(tensorSpec.getName()) instanceof Tensor) {
				javaTensor = (Tensor<T>) inputsMap.get(tensorSpec.getName());
			} else if (inputsMap.get(tensorSpec.getName()) instanceof Sequence) {
				Sequence seq = (Sequence) inputsMap.get(tensorSpec.getName());
				javaTensor = 
						Tensor.build(tensorSpec.getName(), tensorSpec.getAxesOrder(), 
								(RandomAccessibleInterval<T>) SequenceToImgLib2.build(seq, tensorSpec.getAxesOrder()));
			} else {
				throw new IllegalArgumentException("Every BioImage.io core transformation requires a Tensor, or at least a "
						+ "Sequence as input.");
			}
			HashMap<String, Object> pythonKwargs = new HashMap<String, Object>();
			pythonKwargs.put("tensor_name", tensorSpec.getName());
			pythonKwargs.putAll(kwargs);
			Map<String, Object> trans = new HashMap<String, Object>();
			trans.put(TransformSpec.getTransformationNameKey(), transformation); 
			trans.put(TransformSpec.getKwargsKey(), pythonKwargs);
			inputsMap.put(tensorSpec.getName(), 
					python.applyTransformationToTensorInPython(trans, javaTensor));		
		}
	}
	
	private static PythonUtils getPythonConfiguration(String transformation) throws IOException {
		PythonUtils pythonUtils = JepUtils.getPythonJepConfiguration();
		if (pythonUtils == null) {
			JepUtils.openPythonConfigurationIfPythonNotInstalled();
			throw new IOException("Transformation '" + transformation.toUpperCase() + "' seems to be only "
					+ "avaialble in Python. And Python is not configured in your Icy installation. In order "
					+ "to use it please configure Python using the Jep Plugin.");
		}
		return pythonUtils;
	}
	*/
}
