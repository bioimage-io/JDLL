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
package io.bioimage.modelrunner.bioimageio.bioengine;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

import org.msgpack.jackson.dataformat.MessagePackFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

import io.bioimage.modelrunner.bioimageio.bioengine.tensor.BioEngineOutput;
import io.bioimage.modelrunner.bioimageio.bioengine.tensor.BioEngineOutputArray;
import io.bioimage.modelrunner.bioimageio.bioengine.tensor.BioengineTensor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;

/**
 * This class sends the corresponding inputs to the wanted model available
 * on the Bioengine, retrieves the corresponding results and converts them into 
 * JDLL tensors
 * 
 * 
 * Class to create inputs that can be sent to the BioEngine server.
 * The input tensors should be defined as an array of inputs, where
 * each of the inputs can be either a String, int or any other type,
 * but in case the input is an array, it should be encoded as a 
 * {@link HashMap}. It needs to have the key "_rtype" with the
 * corresponding value "ndarray", the key "_rvalue" with an array of
 * bytes corresponding to the data wanted to be encoded, a key "_rshape"
 * which should contain the shape of the array and finally the "_rdtype"
 * corresponding to the array datatype.
 * The array of inputs should be then included in another hashmap under
 * the key "inputs", together with the key "model_name" and the name of
 * the model and "decode_json" true.
 * There is an example defined in Python at: 
 * https://gist.github.com/oeway/b6a6b810f94c91bb902e80a2f788b9e2#file-access_triton_service_hyhpa-py-L22
 * 
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class BioengineInterface implements DeepLearningEngineInterface {
	/**
	 * Server where the Bioengine is hosted
	 */
	private String server;
	/**
	 * Object that contains all the info described in the rdf.yaml file of the model
	 */
	private ModelDescriptor rdf;
	/**
	 * Input for the Bioengine, contains all the required info that needs to be 
	 * The input needs to have:
	 *  -An entry called "inputs", whose value is another Map that contains
	 *   the info about the input tensors
	 *  -An entry called model_name with the name of the model
	 *  -A fixed entry called decoe_json that equals to true
	 */
	Map<String, Object> kwargs = new HashMap<String, Object>();
	/**
	 * Map containing the information needed to provide an input to the 
	 * server for models that are defined in the bioimage.io repo.
	 * Model that are not in the bioimage.io repo do not need this map
	 * This map is sent inside the {@link #kwargs} map
	 */
	private HashMap<String, Object> bioimageioKwargs = new HashMap<String, Object>();
	/**
	 * Bioimage.io model ID of the model of interest
	 */
	private String modelID;
	/**
	 * In the input Bioengine map, key for the model name
	 */
	private static final String MODEL_NAME_KEY = "model_name";
	/**
	 * In the input Bioengine map, key for the input tensors.
	 * THe inputs value should always be a list of objects, which
	 * corresponds to encoded tensors and/or parameters for the
	 * Bioengine to use..
	 * The entries of this list can be either:
	 *  -A @see java.util.LinkedHashMap generated containing the name, shape,
	 *  	dtype and data of a tensor
	 *  -A @see java.util.LinkedHashMap for non dimensional parameters
	 */
	private static final String INPUTS_KEY = "inputs";
	/**
	 * In the input Bioengine map, key for whether to 
	 * return the rdf.yaml of the model or not
	 */
	private static final String RDF_KEY = "return_rdf";
	/**
	 * In the input Bioengine map, key for the Bioimage.io model id
	 */
	private static final String ID_KEY = "model_id";
	/**
	 * Name of the default model used to run a model coming from the BioImage.io repo
	 */
	public static final String DEFAULT_BMZ_MODEL_NAME = "bioengine-model-runner";
	/**
	 * String key corresponding to the decode Json parameter in the 
	 * {@link #kwargs} map
	 */
	private static final String DECODE_JSON_KEY = "decode_json";
	/**
	 * Value corresponding to the decode Json parameter in the 
	 * {@link #kwargs} map. It is fixed.
	 */
	private static final boolean DECODE_JSON_VAL = true;
	/**
	 * Key for the input of the BioEngine corresponding to the type of serialization
	 */
	private static final String SERIALIZATION_KEY = "serialization";
	/**
	 * Value for the BioEngine serialization
	 */
	private static final String SERIALIZATION_VAL = "imjoy";
	
	public static void main(String[] args) throws LoadModelException, RunModelException {
		BioengineInterface bi = new BioengineInterface();
		String path = "C:\\Users\\angel\\OneDrive\\Documentos\\"
				+ "pasteur\\git\\deep-icy\\models\\2D UNet Arabidopsis Ap"
				+ "ical Stem Cells_27062023_125425";
		bi.loadModel(path, path);
		bi.addServer(BioEngineAvailableModels.PUBLIC_BIOENGINE_SERVER);
		final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
		final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512 );
		Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		inputs.add(inpTensor);
		final Img< FloatType > img2 = imgFactory.create( 1, 2, 512, 512 );
		Tensor<FloatType> outTensor = Tensor.build("output0", "bcyx", img2);
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
		outputs.add(outTensor);
		bi.run(inputs, outputs);
		System.out.print(DECODE_JSON_VAL);
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		
		List<Object> inputs = new ArrayList<Object>();
		for (Tensor<?> tt : inputTensors) {
			inputs.add(BioengineTensor.build(tt).getAsMap());
		}
		if (rdf.getName().equals("cellpose-python")) {
    		Map<String, Object> pp = new HashMap<String, Object>();
    		pp.put("diameter", 30);
    		inputs.add(pp);
		}
		if (modelID != null) {
			bioimageioKwargs.put(INPUTS_KEY, inputs);
			ArrayList<Object> auxList = new ArrayList<Object>();
			auxList.add(bioimageioKwargs);
			kwargs.put(INPUTS_KEY, auxList);
		} else {
			kwargs.put(INPUTS_KEY, inputs);
		}
		byte[] byteResult;
		try {
			byteResult = executeModelOnBioEngine(compress(serialize(kwargs)));
			bioimageioKwargs.put(INPUTS_KEY, null);
		} catch (IOException e) {
			throw new RunModelException(e.toString());
		}
		BioEngineOutput bioengineOutputs;
		try {
			bioengineOutputs = BioEngineOutput.build(byteResult);
		} catch (Exception e) {
			throw new RunModelException("Error retrieving the Bioengine results." + System.lineSeparator()
										+ e.toString());
		}
		fillOutputTensors(outputTensors, bioengineOutputs.getArrayOutputs());
	}

	/**
	 * Fill the expected output tensors with the data from teh Bioengine outputs
	 * @param outputTensors
	 * 	tensors expeted as defined by the rdf.yaml file
	 * @param arrayOutputs
	 * 	results from the Bioengine
	 * @throws RunModelException if the number of tensors expected is not the same as
	 * 	the number of tensors returned by the Bioengine
	 */
	private void fillOutputTensors(List<Tensor<?>> outputTensors, List<BioEngineOutputArray> arrayOutputs) throws RunModelException {
		if (outputTensors.size() != arrayOutputs.size()) {
			throw new RunModelException("The rdf.yaml file specifies '" + outputTensors.size()
				+ "' tensors, but the Bioengine has "
				+ "produced '" + arrayOutputs.size() + "' output tensors.");
		}
		int c = 0;
		for (Tensor<?> tt : outputTensors) {
			tt.setData(arrayOutputs.get(c ++).getImg());
		}
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		try {
			rdf = ModelDescriptor.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME, false);
		} catch (Exception e) {
			throw new LoadModelException("The rdf.yaml file for "
					+ "model at '" + modelFolder + "' cannot be read.");
		}

		if (rdf.getName().equals("cellpose-python")) {
			kwargs.put(MODEL_NAME_KEY, "cellpose-python");
			kwargs.put(DECODE_JSON_KEY, DECODE_JSON_VAL);
		} else {
			kwargs.put(MODEL_NAME_KEY, DEFAULT_BMZ_MODEL_NAME);
			kwargs.put(SERIALIZATION_KEY, SERIALIZATION_VAL);
			bioimageioKwargs.put(RDF_KEY, false);
			workaroundModelID();
			findBioEngineWeightsIfPossible();
		}
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public void closeModel() {
		this.bioimageioKwargs = null;
		this.kwargs = null;
		this.modelID = null;
		this.rdf = null;
		this.server = null;
	}
	
	public void addServer(String server) {
		this.server = server;
	}
	
	/** TODO 
	 * Workaround for BioImage.io model runner. It does not work with the full version, it
	 * only works with: major_version_/second_version
	 * @throws LoadModelException if not model ID is not found. Without a model ID, the model
	 * 	 cannot run on the Bioengine
	 */
	private void workaroundModelID() throws LoadModelException {
		modelID = rdf.getModelID();
		if (modelID == null)
			throw new LoadModelException("The selected model does not have a model ID, "
					+ "thus it is not sppported to run on the Bioengine.");
		int nSubversions = modelID.length() - modelID.replace("/", "").length();
		if (nSubversions == 2) {
			modelID = modelID.substring(0, modelID.lastIndexOf("/"));
		}
		bioimageioKwargs.put(ID_KEY, modelID);
	}
    
    /**
     * Identifies if the model has weights that are compatible with the Bioengine.
     * @throws LoadModelException if the model does not have a pair of weights
     * 	compatible with the bioengine. The Bioengine only supports
     *  keras, torchscript and onnx
     */
    private void findBioEngineWeightsIfPossible() throws LoadModelException {
		if (BioEngineAvailableModels.isModelSupportedInBioengine(modelID))
			return;
    	throw new LoadModelException("For some reason, the selected model ('" + modelID
    			+ "') is not supported by the Bioengine. The possible reasons range from"
    			+ " needing inputs or/and outputs in a certain format not supported to "
    			+ "requiring a certain DL framework not supported (too old or too specific)."
    			+ " See the list of the models currently available: " 
    			+ BioEngineAvailableModels.getBioengineJson());
    	/**
    	 * TODO talk with Wei to see whihc are the weights currently supported by the
    	 * Bioengine
    	for (String entry : rdf.getWeights().getSupportedDLFrameworks()) {
    		if (entry.equals(ModelWeight.getKerasID())) {
    			bioimageioKwargs.put(MODEL_WEIGHTS_KEY, ModelWeight.getKerasID());
    			return;
    		} else if (entry.equals(ModelWeight.getOnnxID())) {
    			bioimageioKwargs.put(MODEL_WEIGHTS_KEY, ModelWeight.getOnnxID());
    			return;
    		} else if (entry.equals(ModelWeight.getTorchscriptID())) {
    			bioimageioKwargs.put(MODEL_WEIGHTS_KEY, ModelWeight.getTorchscriptID());
    			return;
    		}
    	}
    	if (!rdf.getWeights().getSupportedDLFrameworks().contains(ModelWeight.getTensorflowID()))
    		throw new LoadModelException("");
    	throw new LoadModelException("The Bioengine does not support the DL framework "
    			+ "compatible with the model selected. The bioengine supports the following"
    			+ " frameworks: " + SUPPORTED_BIOENGINE_WEIGHTS + " whereas the model is "
    			+ "only compatible with: " + rdf.getWeights().getSupportedDLFrameworks());
    	 */
    }
    
    /**
     * Serilize the input map to bytes
     * @param kwargs
     * 	map contianing the info we want to serialize
     * @return an array of bytes that contians the info that we want to send to 
     * 	the Bioengine
     * @throws IOException if there is any error in the serialization
     */
    private static byte[] serialize(Map<String, Object> kwargs) throws IOException {
    	ObjectMapper objectMapper = new ObjectMapper(new MessagePackFactory());

		byte[] bytes = objectMapper.writeValueAsBytes(kwargs);
    	return bytes;
    }
    
    /**
     * Compress an array of bytes
     * @param arr
     * 	the array of bytes we want to compresss
     * @return the compressed array of bytes
     * @throws IOException if there is any error during the compression
     */
    private static byte[] compress(byte[] arr) throws IOException {
		byte[] result = new byte[]{};
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(arr.length);
             GZIPOutputStream gzipOS = new GZIPOutputStream(bos)) {
            gzipOS.write(arr);
            // You need to close it before using bos
            gzipOS.close();
            result = bos.toByteArray();
        }
        return result;
    }
	
	/**
	 * Sends a byte array to a model in the BioEngine server, where inference
	 * is performed and it fetches the output array of bytes produced by the server
	 * @param data
	 * 	the data corresponding to the input to the model
	 * @return the output of the server
	 * @throws ProtocolException if the connection with the server cannot be opened 
	 * 	or the server is not found
	 * @throws MalformedURLException if the url is not correct
	 * @throws IOException if the connection with the server cannot be opened 
	 * 	or the server is not found
	 */
	public byte[] executeModelOnBioEngine(byte[] data) throws ProtocolException, 
																		MalformedURLException, 
																		IOException {
		byte[] result =  sendDataToServerAndReceiveResponse(data);
		// Set received data bytes to null to save memory
		data = null;
		return result;
	}
	
	/**
	 * Creates a connection, sends information and receives a response
	 * @param data
	 * 	byte array we want to send to the server
	 * @return a byte array response from the server
	 * @throws ProtocolExceptionif the connection with the server cannot be opened 
	 * 	or the server is not found
	 * @throws MalformedURLException if the url is not correct
	 * @throws IOException if the connection with the server cannot be opened 
	 * 	or the server is not found
	 */
	private byte[] sendDataToServerAndReceiveResponse(byte[] data) throws ProtocolException, 
																MalformedURLException, 
																IOException {
		HttpURLConnection conn = createConnection(data);
		
		byte[] respon;		
		try (InputStream inputStream = conn.getInputStream();) {
            respon = readInputStream(inputStream);
        } catch (IOException ex) {
            InputStream errorStream = conn.getErrorStream();
            respon = readInputStream(errorStream);
            errorStream.close();
        }
		conn.disconnect();
		
		return respon;
	}
	
	/**
	 * Create a post connection with the BioEngine server
	 * @param data
	 * 	byte array we want to send to the server
	 * @return the connection
	 * @throws ProtocolExceptionif the connection with the server cannot be opened 
	 * 	or the server is not found
	 * @throws MalformedURLException if the url is not correct
	 * @throws IOException if the connection with the server cannot be opened 
	 * 	or the server is not found
	 */
	private HttpURLConnection createConnection(byte[] data) throws ProtocolException,
																	MalformedURLException,
																	IOException{
		URL url = new URL(getExecutionURL());
		HttpURLConnection conn= (HttpURLConnection) url.openConnection();           
		conn.setDoOutput( true );
		conn.setDoInput(true);
		conn.setRequestMethod( "POST" );
		conn.setRequestProperty( "Content-Type", "application/msgpack"); 
		conn.setRequestProperty( "Content-Encoding", "gzip"); 
		conn.setRequestProperty( "Content-Length", Integer.toString(data.length));
		try( DataOutputStream wr = new DataOutputStream( conn.getOutputStream())) {
			  wr.write(data);
			  wr.flush();
		}
		return conn;
	}
	
	/**
	 * Convert an input stream into a byte array
	 * @param inputStream
	 * 	input stream to be converted
	 * @return byte array that contains the same info as the {@link InputStream}
	 * @throws IOException if there is any error accessing the {@link InputStream}
	 */
	private static byte[] readInputStream(InputStream inputStream) throws IOException {
        byte[] bytes;
		try (ByteArrayOutputStream byteOut = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                byteOut.write(buffer, 0, bytesRead);
            }
            bytes = byteOut.toByteArray();
        }
		return bytes;
    }
	
	/**
	 * Get the URL of to send the data to be run in the BioEngine
	 * @return the post BioEngine URL
	 */
	private String getExecutionURL() {
		return server + "/public/services/triton-client/execute";
	}
}
