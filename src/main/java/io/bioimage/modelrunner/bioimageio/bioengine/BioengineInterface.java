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
import java.io.ObjectOutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

import io.bioimage.modelrunner.bioimageio.bioengine.tensor.BioengineTensor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;

public class BioengineInterface implements DeepLearningEngineInterface {
	
	private String server;
	
	private ModelDescriptor rdf;
	
	Map<String, Object> kwargs = new HashMap<String, Object>();
	/**
	 * Map containing the instances needed to provide an input to the 
	 * server for models that are defined in the bioimage.io repo.
	 */
	private HashMap<String, Object> bioimageioKwargs = new HashMap<String, Object>();
	
	private String modelID;
	
	private static final String MODEL_NAME_KEY = "";
	
	private static final String INPUTS_KEY = "";
	
	private static final String RDF_KEY = "";
	
	private static final String ID_KEY = "";
	/**
	 * Name of the default model used to run a model coming from the BioImage.io repo
	 */
	private static final String DEFAULT_BMZ_MODEL_NAME = "bioengine-model-runner";
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
	/**
	 * Optional key to run a Bioimage.io model to specify in which weights
	 * the model is going to run
	 */
	private static String MODEL_WEIGHTS_KEY = "weight_format";

	@Override
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
		
		try {
			byte[] byteResult = executeModelOnBioEngine(compress(serialize(kwargs)));
		} catch (IOException e) {
			throw new RunModelException(e.toString());
		}
		
	}

	@Override
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
	public void closeModel() {
		// TODO Auto-generated method stub
		
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
     * Identifies the weights that are compatible with the Bioengine. The BioEngine
     * canot run Tf 1 weights
     */
    private void findBioEngineWeightsIfPossible() {
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
    	try (ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
    			ObjectOutputStream out = new ObjectOutputStream(byteOut);){
            out.writeObject(kwargs);
            return byteOut.toByteArray();
        }
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
	 * @throws ProtocolExceptionif the connection with the server cannot be opened 
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
	 * Creates a connectio, sends information and receives a response
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
		try {
			respon = IOUtils.toByteArray(conn.getInputStream());
		} catch (Exception ex) {
			InputStream aa = conn.getErrorStream();
			respon = IOUtils.toByteArray(aa);
		}
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
	 * Get the URL of to send the data to be run in the BioEngine
	 * @return the post BioEngine URL
	 */
	private String getExecutionURL() {
		return server + "/public/services/triton-client/execute";
	}
}
