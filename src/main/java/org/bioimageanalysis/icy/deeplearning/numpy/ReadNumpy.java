package org.bioimageanalysis.icy.deeplearning.numpy;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Map;



/**
 * TODO
 * TODO
 * TODO
 * TODO
 * Move code here to decode numpy once it is robust to make it more clean and robust
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class ReadNumpy {

	
    private static final int VERSION = 3;
	private static String NUMPY_EXTENSION_Y = ".npy";
	private static String NUMPY_EXTENSION_Z = ".npz";
    private static final byte[] ZIP_PREFIX = {'P', 'K', (byte) 0x03, (byte) 0x04};
    private static final byte[] ZIP_SUFFIX = {'P', 'K', (byte) 0x05, (byte) 0x06};
    private static final int ARRAY_ALIGN = 64;
    private static final String MAGIC_NUMBER = "NDAR";
	
	/**
	 * TODO allow different type of encodings and other options that exist 
	 * in Python
	 * 
	 * Method to load Numpy files as ImgLib2 RandomAccessibleIntervals
	 * @param file
	 * 	file containing the npy array
	 * @param mmapMode
	 * @param allowPickle
	 * @param fixImports
	 * @param encoding
	 * @param maxHeaderSize
	 * 	max size of teh header
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void readArray(String file, String mmapMode, boolean allowPickle,
			boolean fixImports, String encoding, long maxHeaderSize) throws FileNotFoundException, IOException {
		if (!encoding.contentEquals("ASCII"))
			throw new IllegalArgumentException("Only loading .npy files encoded "
					+ "in ASCII is supported at the moment.");
		
		File npyFile = new File(file);
    	if (!npyFile.isFile() || !file.endsWith(NUMPY_EXTENSION_Y)
    			|| !file.endsWith(NUMPY_EXTENSION_Z)) {
    		throw new IllegalArgumentException("Path provided does not correspond to a Numpy file: " + file);
    	}
    	try (InputStream targetStream = new FileInputStream(npyFile)) {
    		readArray(targetStream, mmapMode, allowPickle, 
    				fixImports, maxHeaderSize);
    	}
	}
	
	public static void readArray(InputStream is, String mmapMode, boolean allowPickle,
			boolean fixImports, long maxHeaderSize) {
		DataInputStream fid;
        if (is instanceof DataInputStream) {
        	fid = (DataInputStream) is;
        } else {
        	fid = new DataInputStream(is);
        }
	}
    
    public static void readArray(DataInputStream fp, boolean allowPickle,
    		Map<Object, Object> pickleKwargs, int maxHeaderSize) {
    	if (allowPickle)
    		maxHeaderSize = 2^64;
    	//byte version = readMagic(fp);
    	//checkVersion(version);
    	//Object[] headerInfo = readArrayHeader(fp, version, maxHeaderSize);
    	Object[] headerInfo = null;
    	long[] shape = (long[]) headerInfo[0];
    	boolean fortranOrder = (boolean) headerInfo[1];
    	Object dtype = (String) headerInfo[2];
    	long count;
    	if (shape.length == 0)
    		count = 1;
		else
			count = Arrays.stream(shape).reduce(Long::sum).getAsLong();
    	
    	if (!(dtype instanceof String)) {
    		throw new IllegalArgumentException("Pickled objects saved as Numpy"
    				+ " arrays not supported at the moment.");
    	} else if (isFileObj(fp)) {
    		//fromFile(fp, dtype, count);
    	} else {
    		
    	}
    	
    }
    
    public static boolean isFileObj(Object f) {
    	if (f instanceof FileInputStream || f instanceof FileOutputStream)
    		return true;
    	else if (f instanceof BufferedReader)
    		return true;
    	else if (f instanceof BufferedWriter)
    		return true;
    	else
    		return false;
    }
    
    public static void fromFile(DataInputStream fp, String dtype, long count) throws IOException {
    	if (dtype == null)
    		throw new IOException("fromfile() needs a 'dtype' or 'formats' argument.");
    	
    }

}
