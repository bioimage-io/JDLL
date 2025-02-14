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
package io.bioimage.modelrunner.utils;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import io.bioimage.modelrunner.system.PlatformDetection;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;

/**
 * Class that contains common useful static methods that can be used anywhere in the software
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class CommonUtils {
	
	private static String USER_AGENT;
	
	public static Calendar cal;
	
	/**
	 * Gets the filename of the file in an URL from the url String
	 * @param str
	 * 	the URL string
	 * @return the file name of the file in the URL
	 * @throws MalformedURLException if the String does not correspond to an URL
	 */
	public static String getFileNameFromURLString(String str) throws MalformedURLException {
		if (str.startsWith(Constants.ZENODO_DOMAIN) && str.endsWith(Constants.ZENODO_ANNOYING_SUFFIX))
			str = str.substring(0, str.length() - Constants.ZENODO_ANNOYING_SUFFIX.length());
		URL url = new URL(str);
		return new File(url.getPath()).getName();
	}
	
	/**
	 * Get the ImgLib2 datatype from a String datatype.
	 * String data types are "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "float32", "float64"
	 * @param <T>
	 * 	what the possible ImgLib2 data types have to extend
	 * @param dataType
	 * 	the String data type of interest
	 * @return the ImgLib2 data type referred by the string
	 */
	public static <T extends RealType<T> & NativeType<T>> T getImgLib2DataType(String dataType) {
		T type;
		if (dataType.equals("int8")) {
			type = Cast.unchecked(new ByteType());
		} else if (dataType.equals("uint8")) {
			type = Cast.unchecked(new UnsignedByteType());
		} else if (dataType.equals("int16")) {
			type = Cast.unchecked(new ShortType());
		} else if (dataType.equals("uint16")) {
			type = Cast.unchecked(new UnsignedShortType());
		} else if (dataType.equals("int32")) {
			type = Cast.unchecked(new IntType());
		} else if (dataType.equals("uint32")) {
			type = Cast.unchecked(new UnsignedIntType());
		} else if (dataType.equals("int64")) {
			type = Cast.unchecked(new LongType());
		} else if (dataType.equals("float32")) {
			type = Cast.unchecked(new FloatType());
		} else if (dataType.equals("float64")) {
			type = Cast.unchecked(new DoubleType());
		} else {
			throw new IllegalArgumentException("Unsupported data type: " + dataType);
		}
		return type;
	}
	
	/**
	 * Get a the String name of an ImgLib2 {@link RandomAccessibleInterval}
	 * String data types are "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "float32", "float64"
	 * @param <T>
	 * 	what the possible ImgLib2 data types have to extend
	 * @param rai
	 * 	the image from which the data type will be retrieved
	 * @return a String containing the data type of the {@link RandomAccessibleInterval}
	 */
	public static <T extends RealType<T> & NativeType<T>>
	String getDataTypeFromRAI(RandomAccessibleInterval<T> rai) {
		if (rai.getAt(rai.minAsLongArray()) instanceof ByteType) {
			return "int8";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof UnsignedByteType) {
			return "uint8";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof ShortType) {
			return "int16";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof UnsignedShortType) {
			return "uint16";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof IntType) {
			return "int32";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof UnsignedIntType) {
			return "uint";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof LongType) {
			return "int64";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof FloatType) {
			return "float32";
		} else if (rai.getAt(rai.minAsLongArray()) instanceof DoubleType) {
			return "float64";
		} else {
			throw new IllegalArgumentException("Data type not supported: " 
					+ rai.getAt(rai.minAsLongArray()).getClass());
		}
	}
	
	/**
	 * Get a the String name of an ImgLib2 {@link T} type
	 * String data types are "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "float32", "float64"
	 * @param <T>
	 * 	what the possible ImgLib2 data types have to extend
	 * @param type
	 * 	the ImgLib2 type.
	 * @return a String representing the data type represented by the argument 'type'
	 */
	public static <T extends RealType<T> & NativeType<T>>
	String getDataType(T type) {
		if (type instanceof ByteType) {
			return "int8";
		} else if (type instanceof UnsignedByteType) {
			return "uint8";
		} else if (type instanceof ShortType) {
			return "int16";
		} else if (type instanceof UnsignedShortType) {
			return "uint16";
		} else if (type instanceof IntType) {
			return "int32";
		} else if (type instanceof UnsignedIntType) {
			return "uint32";
		} else if (type instanceof LongType) {
			return "int64";
		} else if (type instanceof FloatType) {
			return "float32";
		} else if (type instanceof DoubleType) {
			return "float64";
		} else {
			throw new IllegalArgumentException("Data type not supported: " + type.getClass());
		}
	}
	
	public static String getTime() {
		if (cal == null)
			cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
		String dateString = sdf.format(cal.getTime());
		return dateString;
	}
	
	public static boolean int32Overflows(int[] arr, int bytesPerValue) {
		double div = Integer.MAX_VALUE / bytesPerValue;
		for (int a : arr)
			div = div / (double) a;
		if (div <  1)
			return true;
		return false;
	}
	
	public static boolean int32Overflows(long[] arr, int bytesPerValue) {
		double div = Integer.MAX_VALUE / bytesPerValue;
		for (long a : arr)
			div = div / (double) a;
		if (div <  1)
			return true;
		return false;
	}
	
	public static boolean int64Overflows(int[] arr, int bytesPerValue) {
		double div = Long.MAX_VALUE / bytesPerValue;
		for (int a : arr)
			div = div / (double) a;
		if (div <  1)
			return true;
		return false;
	}
	
	public static boolean int64Overflows(long[] arr, int bytesPerValue) {
		double div = Long.MAX_VALUE / bytesPerValue;
		for (long a : arr)
			div = div / (double) a;
		if (div <  1)
			return true;
		return false;
	}
	
	public static void main(String[] args) {
		System.out.println(getTime());
	}
	
	/**
	 * Create a user agent for browsing
	 * @return the user agent
	 */
	public static String getJDLLUserAgent() {
		if (USER_AGENT != null)
			return USER_AGENT;
		String os = "Unknown";
		if (PlatformDetection.isWindows()) {
			os = "Windows";
		} else if (PlatformDetection.isLinux()) {
			os = "Linux";
		} else if (PlatformDetection.isMacOS()) {
			os = "MacOS";
		}
		USER_AGENT = "jdll/" + Constants.JDLL_VERSION + "(" + os + "; Java " + PlatformDetection.getJavaVersion();
		return USER_AGENT;
	}
}
