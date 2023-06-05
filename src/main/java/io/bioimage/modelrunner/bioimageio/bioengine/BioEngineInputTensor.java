package io.bioimage.modelrunner.bioimageio.bioengine;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
 * Class to create the map that encodes the array inputs that need to be
 * passed to the BioEngine.
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
 * @author Carlos Javier García López de Haro
 */
public class BioEngineInputTensor {
	
	/**
	 * Map containing the instances needed to provide an input to the 
	 * server.
	 * The input needs to have:
	 *  -An entry called "inputs", whose value is another Map that contains
	 *   the info about the input tensors
	 *  -An entry called model_name with the name of the model
	 *  -A fixed entry called decoe_json that equals to true
	 */
	private Map<String, Object> inputs = new HashMap<String, Object>();
	/**
	 * String key corresponding to the type of the array in the
	 * {@link #inputs} map
	 */
	private static String typeKey = "_rtype";
	/**
	 * Value corresponding to the type of the array in the
	 * {@link #inputs} map
	 */
	private String ndarrayVal = "ndarray";
	/**
	 * Value corresponding to a parameter
	 */
	private String paramVal = "parameter";
	/**
	 * String key corresponding to the value of the array in the
	 * {@link #inputs} map
	 */
	private String valKey = "_rvalue";
	/**
	 * String key corresponding to the shape of the array in the
	 * {@link #inputs} map
	 */
	private String shapeKey = "_rshape";
	/**
	 * String key corresponding to the dtype of the array in the
	 * {@link #inputs} map
	 */
	private String dtypeKey = "_rdtype";
	/**
	 * String corresponding to the dtype of the array in the
	 * {@link #inputs} map
	 */
	private String dtypeVal;
	/**
	 * Buffer representation of the array input of interest
	 */
	private Buffer buff;
	/**
	 * String used as tag for the int32 np dtype
	 */
	private static String int32Str = "int32";
	/**
	 * String used as tag for the float32 np dtype
	 */
	private static String float32Str = "float32";
	/**
	 * String used as tag for the float64 np dtype
	 */
	private static String float64Str = "float64";
	/**
	 * String used as tag for the byte or int8 np dtype
	 */
	private static String byteStr = "int8";
	/**
	 * String used as tag for the byte or int16 np dtype
	 */
	private static String int16Str = "int16";
	
	/**
	 * Create the BioEngine input 
	 * @param buff
	 * 	buffer containing the information about the array of interest
	 * @param shape
	 * 	shape of the array of interest
	 * @throws Exception if the data type of the buffer is not within the allowed ones 
	 */
	private BioEngineInputTensor(Buffer buff, int[] shape) throws Exception {
		this.inputs.put(typeKey, ndarrayVal);
		this.inputs.put(shapeKey, shape);
		this.buff = buff;
		setDType();
		this.inputs.put(valKey, bufferInputToByteArray());
	}

	private BioEngineInputTensor(Map<String, Object> params) throws Exception {
		this.inputs.put(typeKey, paramVal);
		this.inputs.put(shapeKey, new int[] {1});
		this.inputs.put(this.dtypeKey, "BYTES");
		this.inputs.put(valKey, params);
	}
	
	public static Map<String, Object> buildParamInput(Map<String, Object> params) throws Exception {
		return new BioEngineInputTensor(params).getInputsMap();
	}
	
	/**
	 * Returns a {@link #BioEngineInputTensor(FloatBuffer, int[])} linked map
	 * @param buff
	 * 	buffer containing the information about the array of interest
	 * @param shape
	 * 	shape of the array of interest
	 * @return the LinkedHashMap produced by the {@link #BioEngineInputTensor(FloatBuffer, int[])}
	 * 	class. The map is actually what is used as an input in the BioEngine
	 * @throws Exception if the data type of the buffer is not within the allowed ones 
	 */
	public static Map<String, Object> build(Buffer buff, int[] shape) throws Exception {
		return new BioEngineInputTensor(buff, shape).getInputsMap();
	}
	
	/**
	 * Transforms the FloatBuffer of interest into a byteArray to
	 * be able to send it to the BioEngine server
	 * @return the byte array
	 * @throws Exception if the data type of the buffer is not within the allowed ones 
	 */
	private byte[] bufferInputToByteArray() throws Exception {
		// Set the number of bytes used by the byte buffer depending on its type
		int nBytes = dtypeVal.contains("16") ? 2 : 
						(dtypeVal.contains("32") ? 4 : 
							(dtypeVal.contains("64") ? 8 : 1));
		// Important that the ByteOrder is LittleEndian, as numpy.tobytes() uses that order by default
		// and it is what is used in the BioEngine
		ByteBuffer byteBuffer = ByteBuffer.allocate(buff.capacity() * nBytes).order(ByteOrder.LITTLE_ENDIAN);
		// Fill the buffer from data coming from each of the buffers
		if (this.buff instanceof IntBuffer) {
			byteBuffer.asIntBuffer().put((IntBuffer) buff);
		} else if (this.buff instanceof ByteBuffer) {
			byteBuffer.put((ByteBuffer) buff);
		} else if (this.buff instanceof FloatBuffer) {
			byteBuffer.asFloatBuffer().put(((FloatBuffer) buff).array());
		} else if (this.buff instanceof DoubleBuffer) {
			byteBuffer.asDoubleBuffer().put((DoubleBuffer) buff);
		} else if (this.buff instanceof ShortBuffer) {
			byteBuffer.asShortBuffer().put((ShortBuffer) buff);
		} else {
			throw new Exception("The type of the image is not within the allowd types.\n"
					+ "Allowed types are: " + int32Str + ", " 
					+ float32Str + ", " + float64Str + ", " + byteStr + ", " + int16Str + ".");
		}
		byte[] bytearray = byteBuffer.array();
		
		return bytearray;
	}
	
	/**
	 * Convert a signed byte array into an unsigned byte array
	 * @param signed
	 * 	the signed byte array
	 * @return an unsigned byte array
	 */
	public static byte[] signed2unsignedByte(byte[] signed) {
		byte[] unsigned = new byte[signed.length];
		int c = 0;
		for (byte b : signed)
			unsigned[c ++] = (byte) (b & 0xFF);
		return unsigned;
	}
	
	/**
	 * Set the input dType as the dtype of the data
	 * @throws Exception if the data type of the buffer is not within the allowed ones 
	 */
	public void setDType() throws Exception {
		if (this.buff instanceof IntBuffer) {
			dtypeVal = int32Str;
		} else if (this.buff instanceof ByteBuffer) {
			dtypeVal = byteStr;
		} else if (this.buff instanceof FloatBuffer) {
			dtypeVal = float32Str;
		} else if (this.buff instanceof DoubleBuffer) {
			dtypeVal = float64Str;
		} else if (this.buff instanceof ShortBuffer) {
			dtypeVal = int16Str;
		} else {
			throw new Exception("The type of the image is not within the allowd types.\n"
					+ "Allowed types are: " + int32Str + ", " 
					+ float32Str + ", " + float64Str + ", " + byteStr + ", " + int16Str + ".");
		}
		this.inputs.put(this.dtypeKey, dtypeVal);
	}
	
	/**
	 * REturn the encoded map of the array tnsor. This is what actually
	 * goes into the BioEngine
	 * @return the encoded array input tensor
	 */
	public Map<String, Object> getInputsMap(){
		return inputs;
	}
	
	public String getType() {
		return (String) this.inputs.get(typeKey);
	}
	
	public String getDataType() {
		return (String) this.inputs.get(dtypeKey);
	}
	
	public int[] getShape() {
		List<Integer> list = (List<Integer>) this.inputs.get(shapeKey);
		int[] array = new int[list.size()];
		for (int i = 0; i < list.size(); i ++)
			array[i] = (int) list.get(i);
		return array;
	}
	
	public Object getData() {
		return this.inputs.get(valKey);
	}
	
	public void addParams(Map<String, Object> nParams) {
		if (!this.inputs.get(typeKey).equals(paramVal))
			throw new IllegalArgumentException();
		Map<String, Object> nMap = (Map<String, Object>) getData();
		nMap.putAll(nParams);
		this.inputs.put(valKey, nMap);
	}
}
