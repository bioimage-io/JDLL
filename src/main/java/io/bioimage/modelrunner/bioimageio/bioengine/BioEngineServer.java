package io.bioimage.modelrunner.bioimageio.bioengine;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.zip.GZIPOutputStream;

import org.apache.commons.io.IOUtils;
import org.bioimageanalysis.icy.deepicy.model.description.ModelDescriptor;
import org.msgpack.jackson.dataformat.MessagePackFactory;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Class to interact with the Bioimage.io BioEngine server.
 * 
 * @author Carlos Javier García López de Haro
 */
public class BioEngineServer {

	/**
	 * Server where the BioEngine is hosted
	 */
	private String server;
	/**
	 * Name of the model of interest
	 */
	private String modelName;
	/**
	 * For BioEngine execution calls, the data that is going to
	 * be sent to the server to make inference on
	 */
	private byte[] data;
	/**
	 * Name of the default model used to run a model coming from the BioImage.io repo
	 */
	private static String defaultBioimageIoModelName = "bioengine-model-runner";
	
	/**
	 * Class to interact with a server that hosts the BioENgine
	 * @param server
	 * 	name of the server
	 */
	private BioEngineServer(String server) {
		this.server = server;
	}
	
	/**
	 * Creates a BioEngine server
	 * @param server
	 * 	the name of the server where the BioEngine is hosted
	 * @return the {@link BioEngineServer} object
	 */
	public static BioEngineServer build(String server) {
		return new BioEngineServer(server);
	}
	
	/**
	 * Gets the {@link ModelDescriptor} of the model of interest by reading
	 * the model rdf.yaml from the BioEngine server
	 * @param modelName
	 * 	model of interest
	 * @return the model specs
	 * @throws Exception if the rdf.yaml of the model does not exist or 
	 * 	cannot be read properly
	 */
	public ModelDescriptor getModelDescriptor(String modelName) throws Exception {
		this.modelName = modelName;
		return ModelDescriptor.loadFromJsonTextString(getConfigStr());
	}
	
	/**
	 * Retrieves a Json file in String format from a web address containing a Json file
	 * @param urlStr
	 * 	the Url where the json is
	 * @return
	 * 	a json file in String format
	 * @throws MalformedURLException if the URL is not correct
	 * @throws ProtocolException if the server cannot be accessed
	 * @throws IOException if any other issue happens
	 */
	public static String getJsonStringFromHttpRequest(String urlStr) throws MalformedURLException,
																			ProtocolException,
																			IOException {
		URL url = new URL(urlStr);
		HttpURLConnection con = (HttpURLConnection) url.openConnection();
		con.setRequestMethod("GET");
		con.setRequestProperty("Accept-Charset", "UTF-8");
		InputStream response = con.getInputStream();
		String responseBody = null;
		try (Scanner scanner = new Scanner(response)) {
		    responseBody = scanner.useDelimiter("\\A").next();
		}
		return responseBody;
	}
	
	/**
	 * Provides the String url where the rdf.yaml file (config file)
	 * of the wanted model is located
	 * @return the String URL
	 */
	private String getConfigUrl() {
		return server + "/public/services/triton-client/get_config?model_name=" + modelName;
	}
	
	/**
	 * Obtain the rdf.yaml file that contains the specifications of the model
	 * 
	 */
	private String getConfigStr() {
		String jsonStr = null;
		try {
			jsonStr = getJsonStringFromHttpRequest(getConfigUrl());
		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (ProtocolException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return jsonStr;
	}
	
	/**
	 * Get the URL of to send the data to be run in the BioEngine
	 * @return the post BioEngine URL
	 */
	private String getExecutionURL() {
		return server + "/public/services/triton-client/execute";
	}
	
	/**
	 * Create a post connection with the BioEngine server
	 * @return the connection
	 * @throws ProtocolExceptionif the connection with the server cannot be opened 
	 * 	or the server is not found
	 * @throws MalformedURLException if the url is not correct
	 * @throws IOException if the connection with the server cannot be opened 
	 * 	or the server is not found
	 */
	private HttpURLConnection createConnection() throws ProtocolException,
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
	 * Creates a connectio, sends information and receives a response
	 * @return a byte array response from the server
	 * @throws ProtocolExceptionif the connection with the server cannot be opened 
	 * 	or the server is not found
	 * @throws MalformedURLException if the url is not correct
	 * @throws IOException if the connection with the server cannot be opened 
	 * 	or the server is not found
	 */
	private byte[] sendDataToServerAndReceiveResponse() throws ProtocolException, 
																MalformedURLException, 
																IOException {
		HttpURLConnection conn = createConnection();
		
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
		this.data = data;
		byte[] result =  sendDataToServerAndReceiveResponse();
		// Set received data bytes to null to save memory
		this.data = null;
		return result;
	}
	
	public static byte[] compress(byte[] uncompressedData) throws IOException {
		byte[] result = new byte[]{};
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(uncompressedData.length);
             GZIPOutputStream gzipOS = new GZIPOutputStream(bos)) {
            gzipOS.write(uncompressedData);
            // You need to close it before using bos
            gzipOS.close();
            result = bos.toByteArray();
            bos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
	}
	
	/**
	 * Whether the name of the model corresponds to the key used to run BioImage.io models
	 * @param name
	 * 	name of a model
	 * @return true if the model name correpsonds to the Bioimage.io runner and false otherwise
	 */
	public static boolean isBioImageIoKey(String name) {
		if (name != null && name.equals(defaultBioimageIoModelName))
			return true;
		else
			return false;
	}
	
	/**
	 * REturn the name of the model used to run the BioImage.io repo models
	 * @return the name of the general model runner for Bioimage.io
	 */
	public static String bioImageIoModelName() {
		return defaultBioimageIoModelName;
	}
	
	public static void main(String[] args) throws IOException {
		float[] data = new float[196608];
		ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 4);        
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
        floatBuffer.put(data);
        byte[] array = byteBuffer.array();
		int[] shape = new int[] {3, 256, 256};
		String type = "float32";
		
		Map<String, Object> map = new HashMap<String, Object>();
		map.put("_rtype", "ndarray");
		map.put("_rvalue", array);
		map.put("_rshape", shape);
		map.put("_rdtype", type);
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("diameter", 30);
		List<Object> inputs = new ArrayList<Object>();
		inputs.add(map);
		inputs.add(params);
		
		Map<String, Object> kwargs = new HashMap<String, Object>();
		kwargs.put("inputs", inputs);
		kwargs.put("model_name", "cellpose-python");
		kwargs.put("decode_json", true);
		
		ObjectMapper objectMapper = new ObjectMapper(new MessagePackFactory());

		byte[] bytes = objectMapper.writeValueAsBytes(kwargs);
		//byte[] compressed = compressByteArray(bytes);
		byte[] compressed = compress(bytes);
		
		try {
			int    postDataLength = compressed.length;
			String request        = "https://ai.imjoy.io/public/services/triton-client/execute";
			URL    url            = new URL( request );
			HttpURLConnection conn= (HttpURLConnection) url.openConnection();           
			conn.setDoOutput( true );
			//conn.setInstanceFollowRedirects( false );
			conn.setDoInput(true);
			conn.setRequestMethod( "POST" );
			conn.setRequestProperty( "Content-Type", "application/msgpack"); 
			conn.setRequestProperty( "Content-Encoding", "gzip"); 
			conn.setRequestProperty( "Content-Length", Integer.toString( postDataLength ));
			conn.setUseCaches( false );
			try( DataOutputStream wr = new DataOutputStream( conn.getOutputStream())) {
			  wr.write( compressed );
			  wr.flush();
			}
			/*
			OutputStream wr = conn.getOutputStream();
			wr.write( compressed );
			wr.flush();
			wr.close();
			*/
			Map<String, List<String>> aa = conn.getHeaderFields();
						
			byte[] respon = IOUtils.toByteArray(conn.getInputStream());
			//System.out.println(conn.getResponseMessage());
	
			// Deserialize the byte array to a Map
			Map<String, Object> deserialized = objectMapper.readValue(respon, new TypeReference<Map<String, Object>>() {});
			Object bb = deserialized.get("mask");
			System.out.println(deserialized); // => {name=komamitsu, age=42}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

        
	}
}
