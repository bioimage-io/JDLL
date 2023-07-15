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
package io.bioimage.modelrunner.bioimageio.bioengine.tensor;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import io.bioimage.modelrunner.bioimageio.bioengine.BioengineInterface;


/**
 * Class manage outputs produced by a BioEngine server.
 * The output of the BioEngine is an array of bytes that encode a Map with the actual
 * outputs of the model and a field called "__info__" that contains the information
 * about the output produced.
 * There is an example defined in Python at: 
 * https://gist.github.com/oeway/b6a6b810f94c91bb902e80a2f788b9e2#file-access_triton_service_hyhpa-py-L22
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 */
public class BioEngineOutput {

	/**
	 * The deserialized output of the BioEngine server. The BioEngine encodes the
	 * model output into bytes so it can post the result to local consumers
	 * via Https Request. The deserialized output is basically the output of the
	 * model before the encoding by the BioEngine. 
	 * The output is a Map that consists on:
	 *  - An information field "__info__" containing information about the name,
	 *    shape and data type of the outputs
	 *  - A map for each of the individual outputs, which can be ndarrays or 
	 *    parameters
	 */
	public LinkedHashMap<String, Object> deserializedOutput = new LinkedHashMap<String, Object>();
	/**
	 * Key for the entry in the outputs HashMap that contains information
	 * about the other outputs of the BioEngine. This information covers
	 * the output names, the shape, the datatype and the size
	 */
	private static String outputInfoKey = "__info__";
	/**
	 * Key to the information about the outputs inside the __info__
	 * dictionary
	 */
	private static String outputInfoListKey = "outputs";
	/**
	 * List of dictionaries, containing each of them information about
	 * each of the outputs. This information corresponds to the name of the 
	 * output, the datatype, the shape and the size
	 */
	private List<LinkedHashMap<String, Object>> outputsInfo;
	/**
	 * Key in the __info__ HashMap that contains the name of each
	 * of teh outputs of the model
	 */
	private static String outputInfoNameKey = "name";
	/**
	 * Key in the __info__ HashMap that contains the name of the BioENgine model run.
	 * It can be either the name of the model or "bioimage.model-runner" if the model
	 * being run comes from the bioimage.io repo
	 */
	private static String modelNameKey = "model_name";
	/**
	 * Key for the field in the output dictionary that specifies whether the model
	 * execution has been a success or not
	 */
	private static String successKey = "success";
	/**
	 * For each output,key that defined the output object that it is.
	 * Currently it is only available for the type "ndarray" which corresponds
	 * to an image. If it does not exist, te output is assumed to be a 
	 * parameter
	 */
	private static String outputRTypeKey = "_rtype";
	/**
	 * Key for the field of the BioEngine output dictionary that contains the results 
	 * of the model execution for BioImage.io models ONLY
	 */
	private static String bioimageioResultKey = "outputs";
	/**
	 * Value of the {@link #outputRTypeKey} that corresponds to an
	 * image array
	 */
	private String imageArrayValue = "ndarray";
	/**
	 * Whether the model whose output this object represents comes from the
	 * BioImage.io repo
	 */
	private boolean bioimageio = false;
	/**
	 * List containing the output arrays of the BioEngine model
	 * The entries are {@link BioEngineOuputArray}
	 */
	private List<BioEngineOutputArray> list = new ArrayList<BioEngineOutputArray>();
	/**
	 * Whether the outputs have been closed or not
	 */
	private boolean closed = false;
	/**
	 * Error message displayed when there is an error of execution inside the BioEngine
	 */
	private static String errMsg = "Python error at the BioEngine.\n";
	
	/**
	 * Create the Bioengine input 
	 * @param modelName
	 * 	name of the model that is going to be called
	 * @throws IOException  if there is any error deserializing the raw output bytes
	 * @throws Exception if the BioEngine sends an error message after execution
	 */
	private BioEngineOutput(byte[] rawOutput) throws IOException, Exception {
		deserialize(rawOutput);
		// Remove the bytes from memory
		rawOutput = null;
		processOutputs();
	}
	
	/**
	 * Build the object that contains the inputs to
	 * the BioEngine in the corresponding format
	 * @param rawOutput
	 * 	the raw byte array output of the bioengine
	 * @return an object that can be used to provide
	 * 	inputs to the BioEngine
	 * @throws IOException  if there is any error deserializing the raw output bytes
	 * @throws JsonMappingException  if there is any error deserializing the raw output bytes
	 * @throws JsonParseException  if there is any error deserializing the raw output bytes
	 * @throws Exception if the BioEngine sends an error message after execution
	 */
	public static BioEngineOutput build(byte[] rawOutput) throws IOException,
														Exception {
		return new BioEngineOutput(rawOutput);
	}
	
	/**
	 * Creates readable outputs by the Java software from the BioEngine
	 * output. The BioEngine output consists in bytes for every image and 
	 * parameter. This method will create buffers of data that can easily be
	 * transformed into an Icy sequence or Fiji/ImageJ image 
	 * @throws Exception if the BioEngine sends an error message after execution
	 */
	private void processOutputs() throws Exception {
		lookForErrors();
		setOutputsInfo();
		try {
			createOutputsFromInfo();
		} catch (NullPointerException ex) {
			String msg = "JDLL does not recognize the format of the BioEngine output. Either the "
					+ "output is incorrect or the format of the BioEngine has changed and JDLL has"
					+ " not adapted yet.";
			throw new NullPointerException(msg + System.lineSeparator() + ex.toString());
		} catch (Exception ex) {
			throw new Exception("Empty/Incorrect BioEngine output" + System.lineSeparator() + ex.toString());
		}
	}
	
	@SuppressWarnings("unchecked")
	/**
	 * Find if the BioEngine sends any error message after execution
	 * @throws Exception if the BioEngine sends an error message after execution
	 */
	private void lookForErrors() throws Exception {
		LinkedHashMap<String, Object> results = 
				(LinkedHashMap<String, Object>) this.deserializedOutput.get("result");
		// Special models such as cell pose do not have the same keys as the Bioimage.io models
		if (results == null ||results.get("success") == null)
			return;
		boolean success = (boolean) results.get("success");
		if (!success) {
			String msg = (String) results.get("error");
			msg = errMsg + msg;
			throw new Exception(msg);
		}
	}
	
	/**
	 * Create readable objects from the deserialized output of the BioEngine using
	 * the information provided by the BioEngine
	 * @throws Exception if no output can be retrieved from the data returned by the BioEngine
	 */
	private void createOutputsFromInfo() throws Exception {
		if (this.outputsInfo != null && this.outputsInfo.size() != 0)
			for (LinkedHashMap<String, Object> output : this.outputsInfo) {
				createOutput((String) output.get(outputInfoNameKey));
		} else if (this.deserializedOutput != null && this.deserializedOutput.size() != 0) {
			for (String outputName : this.deserializedOutput.keySet())
				createOutput(outputName);
		}
		if (list == null || list.size() == 0) {
			String msg = "The deserialized BioEngine output did not contain (or\n"
					   + "the program could not find) any information.";
			throw new Exception(msg);
		}
	}
	
	/**
	 * Create an array output readable by Java from a single output map produced
	 * by the BioEngine
	 * @param outputName
	 * 	the name of the output of interest
	 * @param output
	 * 	the map containing the info of interest
	 * @throws Exception 
	 */
	private void createOutputBioImageIo(String outputName, LinkedHashMap<String, Object> output) throws Exception {
		Objects.requireNonNull(output);
		if (output.get(successKey) == null || !((boolean) output.get(successKey)))
			return;
		List<LinkedHashMap<String, Object>> outputList = (List<LinkedHashMap<String, Object>>) output.get(bioimageioResultKey);
		Objects.requireNonNull(outputList);
		createOutputsFromList(outputName, outputList);
	}
	
	/**
	 * Create an array output readable by Java from a single output map produced
	 * by the BioEngine
	 * @param outputName
	 * 	the name of the output of interest
	 * @param output
	 * 	the map containing the info of interest
	 * @throws Exception 
	 */
	private void createOutputFromMap(String outputName, LinkedHashMap<String, Object> output) throws Exception {
		if (output.get(outputRTypeKey) != null && output.get(outputRTypeKey).equals(imageArrayValue)) {
			addOutputToList(outputName, output);
		}
	}
	
	/**
	 * Adds output to list of outputs obtained from the BioEngine
	 * @param outputName
	 * 	name of the output
	 * @param output
	 * 	output Map
	 * @throws Exception 
	 */
	private void addOutputToList(String outputName, LinkedHashMap<String, Object> output) throws Exception {
		try {
			this.list.add(BioEngineOutputArray.buildOutput(outputName, output));
		} catch (IllegalArgumentException ex) {
			throw new IllegalArgumentException("Invalid output" + System.lineSeparator() + ex.toString());
		} catch (Exception ex) {
			throw new Exception("Error retrieving output '" + outputName + "'." 
								+ System.lineSeparator() + ex.toString());
		}
	}
	
	/**
	 * Create an array output readable by Java from a single output List produced
	 * by the BioEngine
	 * @param outputList
	 * 	the list containing the info of interest
	 * @throws Exception 
	 */
	private void createOutputsFromList(String name, List<LinkedHashMap<String, Object>> outputList) throws Exception {
		for (int i = 0; i < outputList.size(); i ++) {
			LinkedHashMap<String, Object> output = outputList.get(i);
			String outputName = name + "_" + i;
			createOutputFromMap(outputName, output);
		}
	}
	
	/**
	 * Create an array output readable by Java from a single entry of the output map produced
	 * by the BioEngine
	 * @param outputName
	 * 	the name of the output of interest
	 * @throws Exception 
	 */
	private void createOutput(String outputName) throws Exception {
		Object outObject = deserializedOutput.get(outputName);
		if (!bioimageio && outObject instanceof LinkedHashMap<?, ?>) {
			createOutputFromMap(outputName, (LinkedHashMap<String, Object>) outObject);
		} else if (bioimageio && outObject instanceof LinkedHashMap<?, ?>) {
			createOutputBioImageIo(outputName, (LinkedHashMap<String, Object>) outObject);
		} else if (this.deserializedOutput.get(outputName) instanceof List<?>) {
			// TODO what to do with other types of output
		}
	}
	
	/**
	 * Set the information about the outputs.  This information is obtained
	 * from the same output of the BioEngine
	 */
	public void setOutputsInfo() {
    	throwExceptionIfClosed();
		if (this.outputsInfo == null) {
			LinkedHashMap<String, Object> __info__ = (LinkedHashMap<String, Object>) this.deserializedOutput.get(outputInfoKey);
			this.outputsInfo = (List<LinkedHashMap<String, Object>>) __info__.get(outputInfoListKey );
			bioimageio = isBioImageIoKey((String) __info__.get(modelNameKey));
		}
	}
	
	/**
	 * REturns the output of the BioEngine server after deserialization.
	 * @return the output of the BioEngine
	 */
	public LinkedHashMap<String, Object> getDeserializedOutput(){
    	throwExceptionIfClosed();
		return this.deserializedOutput;
	}
	
	/**
	 * Get the array outputs produced by the model called in the BioEngine
	 * @return the array outputs produced
	 */
	public List<BioEngineOutputArray> getArrayOutputs(){
    	throwExceptionIfClosed();
		return this.list;
	}

	/**
	 * Close the outputs of the BioEngine to free the memory
	 */
	public void close() {
		list = null;
		deserializedOutput = null;
		outputsInfo = null;
		imageArrayValue = null;
		closed = true;
	}
    
    /**
     * Whether the tensor is closed or not
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
    	return closed;
    }
    
    /**
     * Throw {@link IllegalStateException} if the tensor has been closed
     */
    private void throwExceptionIfClosed() {
    	if (!closed)
    		return;
    	throw new IllegalStateException("The tensor that is trying to be modified has already been "
    			+ "closed.");
    }
    
    /**
     * Return the error message that is displayed when the BioEngine fails
     * @return standard BioEngine error message
     */
    public static String getBioEngineErrorMsg() {
    	return errMsg;
    }
    
    /**
     * Method that deserializes a byte array into a Map<String,Object>
     * @param arr
     * 	array of bytes
     * @return map of deserailized bytes
     * @throws IOException if something goes wrong in the deserialization
     * @throws ClassNotFoundException if the deserialized object is not a Map<Sring,Object>
     */
    private static Map<String, Object> deserialize(byte[] arr) throws IOException, ClassNotFoundException{
    	Map<String, Object> map;
    	try (ByteArrayInputStream byteIn = new ByteArrayInputStream(arr);
    			ObjectInputStream in = new ObjectInputStream(byteIn);){
            map = (Map<String, Object>) in.readObject();
        }
    	return map;
    }
	
	/**
	 * Whether the name of the model corresponds to the key used to run BioImage.io models
	 * @param name
	 * 	name of a model
	 * @return true if the model name correpsonds to the Bioimage.io runner and false otherwise
	 */
	public static boolean isBioImageIoKey(String name) {
		if (name != null && name.equals(BioengineInterface.DEFAULT_BMZ_MODEL_NAME))
			return true;
		else
			return false;
	}
}
