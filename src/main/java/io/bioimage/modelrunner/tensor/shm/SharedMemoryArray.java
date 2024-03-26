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
package io.bioimage.modelrunner.tensor.shm;

import java.io.Closeable;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.file.FileAlreadyExistsException;
import java.util.UUID;

import com.sun.jna.Pointer;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.utils.CommonUtils;
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

/**
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * TODO separate unlink and close
 * 
 * Interface to interact with shared memory segments retrieving the underlying information 
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public interface SharedMemoryArray extends Closeable {

	/**
	 * Constant to specify that the shared memory segment that is going to be open is only for 
	 * reading
	 */
    public static final int O_RDONLY = 0;
	/**
	 * Constant to specify that the shared memory segment that is going to be open is for 
	 * reading and/or writing
	 */
    public static final int O_RDWR = 2;
	/**
	 * Constant to specify that the shared memory segment that is going to be open will
	 * be created if it does not exist
	 */
    public static final int O_CREAT = 64;
	/**
	 * Constant to specify that the shared memory regions mapped can be read but not written
	 */
    public static final int PROT_READ = 0x1;
	/**
	 * Constant to specify that the shared memory regions mapped can be written
	 */
    public static final int PROT_WRITE = 0x2;
	/**
	 * Constant to specify that the shared memory regions mapped can be shared with other processes
	 */
    public static final int MAP_SHARED = 0x01;
	/**
	 * List of special characters that should not be used to name shared memory segments
	 */
	final static String[] SPECIAL_CHARS_LIST = new String[] {"/", "\\", "#", "·", "!", "¡", "¿", "?", "@", "|", "$", ">", "<", ";"};


	/**
	 * This method creates (or retrieves if it already exists) a shared memory segment with the wanted name. The byte size is defined by the
	 * 'shape' and 'datatype' arguments.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but the size required by the
	 * shape and data type is not the same as the size of the existing shared memory segment, an exception will
	 * be thrown.
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 
	 * 		SharedMemoryArray shma = SharedMemoryArray.readOrCreate("shm_example", new long[]{2048}, new FloatType());	
	 * 
	 * An exception will be thrown because the required number of bytes is 2048 * 4 (4 bytes per float) = 8196 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region that has been created
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the specified shape and datatype
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray readOrCreate(String name, long[] shape, T datatype) throws FileAlreadyExistsException {
		String strDType = CommonUtils.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, null, false);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, null, false);
    	else 
    		return SharedMemoryArrayMacOS.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, null, false);
	}

	/**
	 * This method creates (or retrieves if it already exists) a shared memory segment with the wanted name. The byte size is defined by the
	 * 'shape' and 'datatype' arguments.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but the size required by the
	 * shape and data type is not the same as the size of the existing shared memory segment, an exception will
	 * be thrown.
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 
	 * 		SharedMemoryArray shma = SharedMemoryArray.readOrCreate("shm_example", new long[]{2048}, new FloatType());	
	 * 
	 * An exception will be thrown because the required number of bytes is 2048 * 4 (4 bytes per float) = 8196 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory region that has been created
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @param isFortran
	 * 	whether the nd array that is going to be stored in the shared memory segment will be flattened in fortran order
	 * 	or not (c-order)
	 * @param isNpy
	 * 	whether th end array that is going to be stored in the shared memory segment will be preceded by a header 
	 * 	containing the information of the nd array. The header will follow the style of Numpy npy files. Note that 
	 * 	the header will occupy some bytes at the begining of the array but it could be useful to blindly retrieve the 
	 * 	array in the shared memory segment
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the specified shape and datatype
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray readOrCreate(String name, long[] shape, T datatype, boolean isFortran, boolean isNpy) throws FileAlreadyExistsException {
		String strDType = CommonUtils.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, isFortran, isNpy);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, isFortran, isNpy);
    	else 
    		return SharedMemoryArrayMacOS.readOrCreate(name, size * DecodeNumpy.DATA_TYPES_MAP.get(strDType), shape, strDType, isFortran, isNpy);
	}

	/**
	 * This method creates (or retrieves if it already exists) a shared memory segment with the wanted name and wanted size.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but its size is
	 * different than the specified size, an exception will be thrown.
	 * 
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 
	 * 		SharedMemoryArray shma = SharedMemoryArray.readOrCreate("shm_example", 8196;	
	 * 
	 * An exception will be thrown because the required number of bytes is 8196 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * 
     * @param name
     * 	name of the shared memory region that has been created
	 * @param size
	 * 	number of bytes that the shared memory region will be
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the wanted size
	 */
	static SharedMemoryArray readOrCreate(String name, int size) throws FileAlreadyExistsException {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.readOrCreate(name, size);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.readOrCreate(name, size);
    	else 
    		return SharedMemoryArrayMacOS.readOrCreate(name, size);
	}

	/**
	 * This method creates a shared memory segment with the wanted size to allocate an nd array
	 * with the wanted data type and shape. The byte size is defined by the
	 * 'shape' and 'datatype' arguments.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray create(long[] shape, T datatype) {
		String strDType = CommonUtils.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.create(size, shape, strDType, true, false);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.create(size, shape, strDType, true, false);
    	else 
    		return SharedMemoryArrayMacOS.create(size, shape, strDType, true, false);
	}

	/**
	 * This method creates a shared memory segment with the wanted size to allocate an nd array
	 * with the wanted data type and shape. The byte size is defined by the
	 * 'shape' and 'datatype' arguments.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * 
	 * @param <T>
     * 	possible ImgLib2 data types of the wanted {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of an ndimensional array that could be stored in the shared memory region
	 * @param datatype
	 * 	datatype of the data that is going to be stored in the region
	 * @param isFortran
	 * 	whether the nd array that is going to be stored in the shared memory segment will be flattened in fortran order
	 * 	or not (c-order)
	 * @param isNpy
	 * 	whether the end array that is going to be stored in the shared memory segment will be preceded by a header 
	 * 	containing the information of the nd array. The header will follow the style of Numpy npy files. Note that 
	 * 	the header will occupy some bytes at the begining of the array but it could be useful to blindly retrieve the 
	 * 	array in the shared memory segment
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray create(long[] shape, T datatype, boolean isFortran, boolean isNpy) {
		String strDType = CommonUtils.getDataType(datatype);
    	int size = 1;
    	for (long i : shape) {size *= i;}
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.create(size, shape, strDType, isNpy, isFortran);
    	else if (PlatformDetection.isLinux())
    		return SharedMemoryArrayLinux.create(size, shape, strDType, isNpy, isFortran);
    	else 
    		return SharedMemoryArrayMacOS.create(size, shape, strDType, isNpy, isFortran);
	}

	/**
	 * This method creates a segment on the Shared Memory region of the computer with the size
	 * needed to store an image of the wanted characteristics.
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * 
	 * @param size
	 * 	byte size wanted to allocate
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static SharedMemoryArray create(int size) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.create(size);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.create(size);
    	else 
    		return SharedMemoryArrayMacOS.create(size);
	}

	/**
	 * Wraps an existing shared memory segment to allow the user its manipulation.
	 * The name should be the same as the name of the shared memory segment.
	 * 
	 *  The shared memory segment has a defined size and characteristics such as how the
	 *  nd arrays are saved (with fortran or c order, with Numpy npy format or not, ...).
	 *  
	 *  The {@link SharedMemoryArray} instance retrieved can be used to modify the underlying shared
	 *  memory segment
	 * 
	 * 
	 * @param name
	 * 	name of the shared memory segment to be accessed
	 * @return a {@link SharedMemoryArray} instance that helps handling the data written to the shared memory region
	 */
	static SharedMemoryArray read(String name) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.read(name);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.read(name);
    	else 
    		return SharedMemoryArrayMacOS.read(name);
	}

	/**
	 * Get the size of the shared memory segment at the location of the provided name
	 * @param name
	 * 	name of the shared memory segment
	 * @return the size in bytes of the shared memory segment of interest
	 */
	static long getSize(String name) {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.getSHMSize(name);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.getSHMSize(name);
    	else 
    		return SharedMemoryArrayMacOS.getSHMSize(name);
	}

	/**
	 * Creates (or retrieves if it already exists) a {@link SharedMemoryArray} instance that wraps a shared memory segment where the 
	 * nd array represented by the {@link RandomAccessibleInterval} is going to be copied.
	 * 
	 * The name should not be already taken or the shared memory segment at the location of the 
	 * name should be of the same byte suze as the required to write the {@link RandomAccessibleInterval}.
	 * 
	 * The {@link RandomAccessibleInterval} is flattened in c-order and with a header specifying the 
	 * characteristics of the nd array (datatype, dimensions...) at the beginning of the byte array. This
	 * follows the Numpy npy format. Note that this header increases the byte size of the shared memory segment.
	 * 
	 * @param <T>
	 * 	the possible ImgLib2 data types that the {@link RandomAccessibleInterval} can have
	 * @param name
	 * 	the name of the shared memory segment where the {@link RandomAccessibleInterval} is going to be copied
	 * @param rai
	 * 	the nd array that is going to be copied to the shared memory segment
	 * @return an instance of {@link SharedMemoryArray} that allows manipulation of the shared memory region
	 * @throws FileAlreadyExistsException if the shared memory segment already exists and has a different size than
	 * 	the required to copy the nd array {@link RandomAccessibleInterval} instance 
	 */
	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai) throws FileAlreadyExistsException {
		return createSHMAFromRAI(name, rai, false, true);
    }

	/**
	 * Creates a {@link SharedMemoryArray} instance that wraps a shared memory segment where the 
	 * nd array represented by the {@link RandomAccessibleInterval} is going to be copied.
	 * 
	 * The {@link RandomAccessibleInterval} is flattened in c-order and with a header specifying the 
	 * characteristics of the nd array (datatype, dimensions...) at the beginning of the byte array. This
	 * follows the Numpy npy format. Note that this header increases the byte size of the shared memory segment.
	 * 
	 * @param <T>
	 * 	the possible ImgLib2 data types that the {@link RandomAccessibleInterval} can have
	 * @param rai
	 * 	the nd array that is going to be copied to the shared memory segment
	 * @return an instance of {@link SharedMemoryArray} that allows manipulation of the shared memory region
	 */
	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(RandomAccessibleInterval<T> rai) {
		return createSHMAFromRAI(rai, false, true);
    }

	/**
	 * Creates a {@link SharedMemoryArray} instance that wraps a shared memory segment where the 
	 * nd array represented by the {@link RandomAccessibleInterval} is going to be copied.
	 * 
	 * The {@link RandomAccessibleInterval} is flattened in c-order and with a header specifying the 
	 * characteristics of the nd array (datatype, dimensions...) at the beginning of the byte array. This
	 * follows the Numpy npy format.
	 * 
	 * @param <T>
	 * 	the possible ImgLib2 data types that the {@link RandomAccessibleInterval} can have
	 * @param rai
	 * 	the nd array that is going to be copied to the shared memory segment
	 * @param isFortranOrder
	 * 	whether the {@link RandomAccessibleInterval} nd array is save in fortran order or not (c-order)
	 * @param isNumpy
	 * 	whether the shared memory segment starts with a byte array header that converted into string 
	 * 	provides information about the array, such as shape, data type or byte order. Note that this header increases
	 * 	the shared memory segment byte size.
	 * @return an instance of {@link SharedMemoryArray} that allows manipulation of the shared memory region
	 */
	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) {
		try {
			return createSHMAFromRAI(SharedMemoryArray.createShmName(), rai, isFortranOrder, isNumpy);
		} catch (FileAlreadyExistsException e) {
			throw new RuntimeException("Unexpected exception", e);
		}
    }

	/**
	 * Creates (or retrieves if it already exists) a {@link SharedMemoryArray} instance that wraps a shared memory segment where the 
	 * nd array represented by the {@link RandomAccessibleInterval} is going to be copied.
	 * 
	 * The name should not be already taken or the shared memory segment at the location of the 
	 * name should be of the same byte suze as the required to write the {@link RandomAccessibleInterval}.
	 * 
	 * 
	 * @param <T>
	 * 	the possible ImgLib2 data types that the {@link RandomAccessibleInterval} can have
	 * @param name
	 * 	the name of the shared memory segment where the {@link RandomAccessibleInterval} is going to be copied
	 * @param rai
	 * 	the nd array that is going to be copied to the shared memory segment
	 * @param isFortranOrder
	 * 	whether the {@link RandomAccessibleInterval} nd array is save in fortran order or not (c-order)
	 * @param isNumpy
	 * 	whether the shared memory segment starts with a byte array header that converted into string 
	 * 	provides information about the array, such as shape, data type or byte order. Note that this header increases
	 * 	the shared memory segment byte size.
	 * @return an instance of {@link SharedMemoryArray} that allows manipulation of the shared memory region
	 * @throws FileAlreadyExistsException if the shared memory segment already exists and has a different size than
	 * 	the required to copy the nd array {@link RandomAccessibleInterval} instance 
	 */
	public static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArray createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) throws FileAlreadyExistsException {
        if (PlatformDetection.isWindows()) 
        	return SharedMemoryArrayWin.createSHMAFromRAI(name, rai, isFortranOrder, isNumpy);
    	else if (PlatformDetection.isLinux()) 
    		return SharedMemoryArrayLinux.createSHMAFromRAI(name, rai, isFortranOrder, isNumpy);
    	else 
    		return SharedMemoryArrayMacOS.createSHMAFromRAI(name, rai, isFortranOrder, isNumpy);
    }
	
	/**
	 * Checks whether the String provided  can be used as the name given to a shared memory segment
	 * @param name
	 * 	the string that wants to be used as a name to a shared memory segment
	 * @throws IllegalArgumentException if the name does not fulfill the required conditions
	 */
	static void checkMemorySegmentName(String name) throws IllegalArgumentException {
		String auxName;
		if (PlatformDetection.isWindows() && name.startsWith("Local\\"))
			auxName = name.substring("Local\\".length());
		else if (name.startsWith("/"))
				auxName = name.substring(1);
		else 
			auxName = name;
		for (String specialChar : SPECIAL_CHARS_LIST) {
			if (auxName.contains(specialChar))
				throw new IllegalArgumentException("Argument 'name' should not contain the special character '" + specialChar + "'.");
		}
		if (PlatformDetection.isMacOS() && auxName.length() > SharedMemoryArrayMacOS.MACOS_MAX_LENGTH - 1)
			throw new IllegalArgumentException("Parameter 'name' cannot have more than " 
									+ (SharedMemoryArrayMacOS.MACOS_MAX_LENGTH - 1) + " characters. Shared memory segments "
									+ "cannot have names with more than " + (SharedMemoryArrayMacOS.MACOS_MAX_LENGTH - 1) + " characters.");
	}
	
	/**
	 * Create a random unique name for a shared memory segment
	 * @return a random unique name for a shared memory segment
	 */
	static String createShmName() {
        if (PlatformDetection.isWindows()) return "Local" + File.separator + UUID.randomUUID().toString();
    	else if (PlatformDetection.isLinux()) return "/shm-" + UUID.randomUUID();
    	else return ("/shm-" + UUID.randomUUID()).substring(0, SharedMemoryArrayMacOS.MACOS_MAX_LENGTH);
	}
    
	/**
	 * Get the number of bytes that is required to store the data in an nd array of a certain data type.
	 * The size of only the array in the corresponding data type, with no header.
	 * For example a [2, 3] float32 array would be 2 * 3 * 4 bytes (per float32 number) = 24 bytes
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of the array
	 * @param type
	 * 	ImgLib2 data type of the array
	 * @return the number of bytes needed to store the nd array
	 */
	public static <T extends RealType<T> & NativeType<T>> int getArrayByteSize(long[] shape, T type) {
		return getArrayByteSize(shape, type, false);
	}
    
	/**
	 * Get the number of bytes that is required to store the data in an nd array of a certain data type
	 * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape of the array
	 * @param type
	 * 	ImgLib2 data type of the array
	 * @param isNpy
	 * 	whether the array is stored with a Numpy npy header at the beginning
	 * @return the number of bytes needed to store the nd array
	 */
	public static <T extends RealType<T> & NativeType<T>> int getArrayByteSize(long[] shape, T type, boolean isNpy) {
		int noByteSize = 1;
		int headerSize = 0;
		if (isNpy) headerSize = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(shape, type);
		for (long l : shape) {noByteSize *= l;}
		if (type instanceof ByteType || type instanceof UnsignedByteType) {
			return noByteSize * 1 + headerSize;
		} else if (type instanceof ShortType || type instanceof UnsignedShortType) {
			return noByteSize * 2 + headerSize;
		} else if (type instanceof IntType || type instanceof UnsignedIntType
				|| type instanceof FloatType) {
			return noByteSize * 4 + headerSize;
		} else if (type instanceof LongType || type instanceof DoubleType) {
			return noByteSize * 8 + headerSize;
		} else {
			throw new IllegalArgumentException("Type not supported: " + type.getClass().toString());
		}
	}
    
	/**
	 * 
	 * @return the unique name for the shared memory, specified as a string. When creating a new shared memory bloc.k instance
	 * 	{@link SharedMemoryArray} a name can be supploed, and if not it will be generated automatically.
	 * 	Two shared memory blocks existing at the same time cannot share the name.
	 * 	In Unix based systems, Shared memory segment names start with "/", for example "/shm_block"
	 * 	In Windows shared memory block names start either with "Global\\" or "Local\\". Example: "Local\\shm_block" 
	 */
    public String getName();
    
    /**
     * 
	 * @return the unique name for the shared memory, specified as a string and as 
	 * 	the Python package multiprocessing.shared_memory returns it. For Unix based systems it removes the 
	 * 	initial "/", for example: "/shm_block" -&gt; "shm_block".
	 * 	In Windows shared memory block names start either with "Global\\" or "Local\\", this is also removed when 
	 * 	providing a shared memory name to Python. Example: "Local\\shm_block" -&gt; "shm_block"
     */
    public String getNameForPython();
    
    /**
     * 
     * @return the pointer to the shared memory segment
     */
    public Pointer getPointer();
    
    /**
     * 
     * @return get number of bytes in the shared memory segment
     */
    public int getSize();
    
    /**
     * 
     * @return the data type of the array that was flattened and copied into the shared memory segment
     */
    public String getOriginalDataType();
    
    /**
     * 
     * @return the shape (array dimensions) of the array that was flattened and copied into the shared memory segment
     */
    public long[] getOriginalShape();
    
    /**
     * Retrieve the {@link RandomAccessibleInterval} defined in the shared memory segment.
     * This method references the shared memory segment, thus every change in the {@link RandomAccessibleInterval}
     * will be reflected in the shared memory segment. 
     * This method assumes that the shape and data type have already been defined or that the shared memory segment
     * contains a header at the beginning with he info on how to reconstruct the nd array (saved as Numpy npy) format.
     * If that is not the case use {@link #getSharedRAI(long[], RealType)}.
     * 
     * IMPORTANT: once the shared memory segment is closed ({@link #close()}, trying to copy or manipulate the 
     * data in the {@link RandomAccessibleInterval} might result in a segmentation error. If you want to close the
     * shared memory segment and keep the {@link RandomAccessibleInterval}, copy it into a new standard one (not backed by a shared memory segment).
     * 
     * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
     * @return the randomAccessible interval that is defined in the shared memory segment
     */
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI();

    /**
     * Retrieve the {@link RandomAccessibleInterval} defined in the shared memory segment.
     * This method references the shared memory segment, thus every change in the {@link RandomAccessibleInterval}
     * will be reflected in the shared memory segment. 
     * Unless defined differently using {@link #create(long[], RealType, boolean, boolean)} or {@link #readOrCreate(String, long[], RealType, boolean, boolean)}
     * this method assumes that the data is saved in c-order. To change it use {@link #getSharedRAI(long[], RealType, boolean)}
     * 
     * IMPORTANT: once the shared memory segment is closed ({@link #close()}, trying to copy or manipulate the 
     * data in the {@link RandomAccessibleInterval} might result in a segmentation error. If you want to close the
     * shared memory segment and keep the {@link RandomAccessibleInterval}, copy it into a new standard one (not backed by a shared memory segment).
     * 
     * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape (array dimensions) into which the flat array of the shared memory segment will be reconstructed
	 * @param dataType
	 * 	the data type into which the bytes in the shared memory region will be converted
     * @return the randomAccessible interval that is defined in the shared memory segment
     */
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType);

    /**
     * Retrieve the {@link RandomAccessibleInterval} defined in the shared memory segment.
     * This method references the shared memory segment, thus every change in the {@link RandomAccessibleInterval}
     * will be reflected in the shared memory segment. 
     * 
     * IMPORTANT: once the shared memory segment is closed ({@link #close()}, trying to copy or manipulate the 
     * data in the {@link RandomAccessibleInterval} might result in a segmentation error. If you want to close the
     * shared memory segment and keep the {@link RandomAccessibleInterval}, copy it into a new standard one (not backed by a shared memory segment).
     * 
     * @param <T>
     * 	possible ImgLib2 data types of the retrieved {@link RandomAccessibleInterval}
	 * @param shape
	 * 	shape (array dimensions) into which the flat array of the shared memory segment will be reconstructed
	 * @param dataType
	 * 	the data type into which the bytes in the shared memory region will be converted
	 * @param isFortran
	 * 	whether the nd array has been flattened using fortran order or not (c-order)
     * @return the randomAccessible interval that is defined in the shared memory segment
     */
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType, boolean isFortran);
    
    /**
     * Copy the ByteBuffer to the shared memory array.
     * @param buffer
     * 	the ByteBuffer to be copied to the shared memory region
     */
    public void setBuffer(ByteBuffer buffer);
    
    /**
     * 
     * @return the {@link ByteBuffer} with all the bytes of the Shared memory segment
     */
    public ByteBuffer getDataBuffer();
    
    /**
     * This method is only different from {@link #getDataBuffer()} if the shm segment is saved in 
     * Numpy Npy format, which contains a header at the beginning of the shm segment with information
     * about he array
     * @return the {@link ByteBuffer} with all the bytes of the Shared memory segment except those dedicated to the header
     */
    public ByteBuffer getDataBufferNoHeader();
    
    /**
     * 
     * @return whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
     */
    public boolean isNumpyFormat();
}
