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

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.util.Arrays;
import java.util.Objects;
import java.util.regex.Matcher;

import com.sun.jna.Pointer;
import com.sun.jna.platform.win32.Kernel32;
import com.sun.jna.platform.win32.WinBase;
import com.sun.jna.platform.win32.WinNT;
import com.sun.jna.platform.win32.WinNT.HANDLE;
import com.sun.jna.platform.win32.BaseTSD;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.ByteAccess;
import net.imglib2.img.basictypeaccess.DoubleAccess;
import net.imglib2.img.basictypeaccess.FloatAccess;
import net.imglib2.img.basictypeaccess.IntAccess;
import net.imglib2.img.basictypeaccess.LongAccess;
import net.imglib2.img.basictypeaccess.ShortAccess;
import net.imglib2.img.basictypeaccess.nio.ByteBufferAccess;
import net.imglib2.img.basictypeaccess.nio.DoubleBufferAccess;
import net.imglib2.img.basictypeaccess.nio.FloatBufferAccess;
import net.imglib2.img.basictypeaccess.nio.IntBufferAccess;
import net.imglib2.img.basictypeaccess.nio.LongBufferAccess;
import net.imglib2.img.basictypeaccess.nio.ShortBufferAccess;
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
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Class that maps {@link RandomAccessibleInterval} objects to the shared memory for interprocessing communication
 * in Windows
 * @author Carlos Garcia Lopez de Haro
 */
public class SharedMemoryArrayWin implements SharedMemoryArray
{
	/**
	 * reference to the file that covers the shared memory region
	 */
	private WinNT.HANDLE hMapFile;
	/**
	 * Pointer referencing the shared memory byte array
	 */
	private Pointer mappedPointer;
	/**
	 * Pointer referencing the shared memory byte array
	 */
	private Pointer writePointer;
	/**
	 * Name of the file containing the shared memory segment. In Unix based systems consits of "/" + file_name.
	 * In Linux the shared memory segments can be inspected at /dev/shm.
	 * For MacOS the name can only have a certain length, {@value #MACOS_MAX_LENGTH}
	 */
	private final String memoryName;
	/**
	 * Size of the shared memory block
	 */
	private int size;
	/**
	 * Shared memory segments store bytes. This field represents the original data type of the array that was written
	 * into the bytes of the shared memory segment. It is helful to retrieve the object later.
	 */
	private String originalDataType;
	/**
	 * Shared memory segments are flat arrays, only one dimension. This field keeps the dimensions of the array before
	 * flattening it and copying it to the shared memory.
	 */
	private long[] originalDims;
	/**
	 * Whether the shared memory segment has been already closed and unlinked or not
	 */
	private boolean unlinked = false;
	/**
	 * Whether the shared memory segment has numpy format or not. Numpy format means that 
	 * it comes with a header indicating shape, dtype and order. If false it is just hte array 
	 * of bytes corresponding to the values of the array, no header
	 */
	private Boolean isNumpyFormat = null;
	/**
	 * This parameter makes sense for nd-arrays. Whether the n-dimensional array is flattened followin
	 * fortran order or not (c order)
	 */
	private boolean isFortran = false;
	private static final int SEC_RESERVE = 0x4000000;
	/**
	 * TODO change size to long, this number overflows and becomes negative
	 */
	private static final int DEFAULT_RESERVED_MEMORY = 1024 * 1024 * 1024 * 2;
	
	/**
	 * This method creates a shared memory segment with the wanted name. The byte size is defined by the
	 * 'size' argument, but it has to be coherent with 'shape', 'dtype' and 'isNumpy' arguments.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but the size required by the
	 * shape and data type is not the same as the size of the existing shared memory segment, an exception will
	 * be thrown.
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 		
	 * 		SharedMemoryArrayWin("shm_example", 4096, "float32", new long[]{1024}, false, false);
	 * 
	 * An exception will be thrown because the required number of bytes is 1024 * 4 (4 bytes per float) = 4096 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param size
	 * 	the byte size of the shared memory segment
	 * @param dtype
	 * 	the data type of the nd array that is written from the shared memory segment
	 * @param shape
	 * 	the dimensions of the nd array that is written from the shared memory segment
	 * @param isNumpy
	 * 	whether an nd array is saved to the shared memory segment in Numpy npy format, that is with a header at the 
	 * 	beginning that increases the byte size
	 * @param isFortran
	 * 	whether nd arrays are stored with fortran order or not (c-order)
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the specified shape and datatype
	 */
	protected SharedMemoryArrayWin(int size, String dtype, long[] shape, Boolean isNumpy, boolean isFortran) throws FileAlreadyExistsException
    {
    	this(SharedMemoryArray.createShmName(), size, dtype, shape, isNumpy, isFortran);
    }
    
	/**
	 * This method creates (or retrieves if it already exists) a shared memory segment with the wanted name. The byte size is defined by the
	 * 'size' argument, but it has to be coherent with 'shape', 'dtype' and 'isNumpy' arguments.
	 * If a memory segment with the provided name already exists, it is wrapped, with read and write permissions.
	 * 
	 * The byte size of the shared memory segment cannot be modified.
	 * 
	 * If a shared memory segment already exists in the location of the name provided, but the size required by the
	 * shape and data type is not the same as the size of the existing shared memory segment, an exception will
	 * be thrown.
	 * For example if a shared memory segment of size 1024 has been created at "shm_example" and we try:
	 * 		
	 * 		SharedMemoryArrayWin("shm_example", 4096, "float32", new long[]{1024}, false, false);
	 * 
	 * An exception will be thrown because the required number of bytes is 1024 * 4 (4 bytes per float) = 4096 bytes &gt; 1024 bytes
	 * 
	 * 
	 * It is useful to allocate in advance the space that a certain {@link RandomAccessibleInterval}
	 * will need. The image can then reference this shared memory region.
	 * An instance of {@link SharedMemoryArray} is created that helps managing the shared memory data.
	 * 
	 * @param name
	 * 	name of the file name that is going to be used to identify the shared memory segment
	 * @param size
	 * 	the byte size of the shared memory segment
	 * @param dtype
	 * 	the data type of the nd array that is written from the shared memory segment
	 * @param shape
	 * 	the dimensions of the nd array that is written from the shared memory segment
	 * @param isNumpy
	 * 	whether an nd array is saved to the shared memory segment in Numpy npy format, that is with a header at the 
	 * 	beginning that increases the byte size
	 * @param isFortran
	 * 	whether nd arrays are stored with fortran order or not (c-order)
	 * @throws FileAlreadyExistsException if a shared memory array with the same name exists and its byte size
	 *                                    does not match the specified shape and datatype
	 */
	protected SharedMemoryArrayWin(String name, int size, String dtype, long[] shape, Boolean isNumpy, boolean isFortran) throws FileAlreadyExistsException
    {
		if (size < 0)
			throw new IllegalArgumentException("The size of a shared memory segment cannot be negative.");
    	memoryName = name;
    	this.originalDataType = dtype;
    	this.originalDims = shape;
    	this.size = size;
    	this.isNumpyFormat = isNumpy;
    	this.isFortran = isFortran;
    	int flag = WinNT.PAGE_READWRITE;
    	boolean write = true;
    	if (checkSHMExists(memoryName)) {
        	long prevSize = getSHMSize(name);
        	if (prevSize != 0 && prevSize != DEFAULT_RESERVED_MEMORY && prevSize < size)
        		throw new FileAlreadyExistsException("Shared memory segment already exists with different dimensions, data type or format. "
        				+ "Size of existing shared memory segment: " + prevSize + ", size of proposed object: " + size);
    	}
		
    	if (size < 1) {
    		flag = WinNT.PAGE_READWRITE | SEC_RESERVE;
    		size = DEFAULT_RESERVED_MEMORY;
    		write = false;
    	}
        hMapFile = Kernel32.INSTANCE.CreateFileMapping(
                WinBase.INVALID_HANDLE_VALUE,
                null,
                flag,
                0,
                size,
                memoryName
        );
        
        if (hMapFile == null) {
            throw new RuntimeException("Error creating shared memory array. CreateFileMapping failed: "
            		+ "" + Kernel32.INSTANCE.GetLastError());
        }
        
        // Map the shared memory
        mappedPointer = Kernel32.INSTANCE.MapViewOfFile(
                hMapFile,
                WinNT.FILE_MAP_WRITE,
                0,
                0,
                size
        );
        
        if (mappedPointer == null) {
            Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("Error creating shared memory array. "
            		+ "Please check that a shared memory segment with another size has "
            		+ "not previously been created on the same memory region with the same name: " + this.memoryName 
            		+ ". MapViewOfFile failed: "
            		+ "" + Kernel32.INSTANCE.GetLastError());
        }
        if (write) {
    	    writePointer = Kernel32.INSTANCE.VirtualAllocEx(Kernel32.INSTANCE.GetCurrentProcess(), 
    	    		mappedPointer, 
    	    		new BaseTSD.SIZE_T(size), WinNT.MEM_COMMIT, WinNT.PAGE_READWRITE);
    	    if (writePointer == null) {
    	    	close();
                throw new RuntimeException("Error committing to the shared memory pages. Errno: "
                		+ "" + Kernel32.INSTANCE.GetLastError());
    	    }
        }
    }
	
	private boolean checkSHMExists(String memoryName) {
    	SharedMemoryArray.checkMemorySegmentName(memoryName);
    	if (!memoryName.startsWith("Local" + File.separator) && !memoryName.startsWith("Global" + File.separator))
    		memoryName = "Local" + File.separator+ memoryName;
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping( WinNT.FILE_MAP_READ, false, memoryName);
		if (hMapFile == null)
			return false;
    	Kernel32.INSTANCE.CloseHandle(hMapFile);
		return true;
	}
    
    /**
     * MEthod to find the size of an already created shared memory segment
     * @param memoryName
     * 	the name of the shared memory segment
     * @return the size in bytes of the shared memory segment
     */
    protected static long getSHMSize(String memoryName) {
    	SharedMemoryArray.checkMemorySegmentName(memoryName);
    	if (!memoryName.startsWith("Local" + File.separator) && !memoryName.startsWith("Global" + File.separator))
    		memoryName = "Local" + File.separator+ memoryName;
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping( WinNT.FILE_MAP_READ, false, memoryName);

    	if (hMapFile == null) {
            throw new RuntimeException("Shared memory segment might not exist: " + memoryName
            			+ ". OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        // Map the shared memory object into the current process's address space
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(hMapFile, WinNT.FILE_MAP_READ, 0, 0, 0);
        if (pSharedMemory == null) {
        	Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        Kernel32.MEMORY_BASIC_INFORMATION mbi = new Kernel32.MEMORY_BASIC_INFORMATION();
        
        if (Kernel32.INSTANCE.VirtualQueryEx(
        		Kernel32.INSTANCE.GetCurrentProcess(), pSharedMemory, mbi, new BaseTSD.SIZE_T((long) mbi.size())
        		).intValue() == 0) {
            throw new RuntimeException("Unable to retrieve the size of the shm segment located at '" 
        		+ "'. Errno: " + Kernel32.INSTANCE.GetLastError());
        }
        int size = mbi.regionSize.intValue();

        Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
    	Kernel32.INSTANCE.CloseHandle(hMapFile);
    	
    	return size;
    }

    /**
     * Private constructor to create an instance for the specific case when it is wrapping an ImgLib2
     * {@link RandomAccessibleInterval}
     * @param name
     * 	name of the shared memory segment
     */
	private SharedMemoryArrayWin(String name) {
		this.memoryName = name;
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayWin readOrCreate(String name, int size, long[] shape, String strDType, Boolean isNumpy, boolean isFortran) throws FileAlreadyExistsException {
		return new SharedMemoryArrayWin(name, size, strDType, shape, isNumpy, isFortran);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayWin readOrCreate(String name, int size) throws FileAlreadyExistsException {
		return new SharedMemoryArrayWin(name, size, null, null, null, false);
	}

	protected static <T extends RealType<T> & NativeType<T>>
	SharedMemoryArrayWin create(int size, long[] shape, String strDType, Boolean isNumpy, boolean isFortran) {
		try {
			return new SharedMemoryArrayWin(size, strDType, shape, isNumpy, isFortran);
		} catch (FileAlreadyExistsException e) {
			throw new RuntimeException("Unexpected error.", e);
		}
	}

	protected static SharedMemoryArrayWin create(int size) {
		try {
			return new SharedMemoryArrayWin(size, null, null, null, false);
		} catch (FileAlreadyExistsException e) {
			throw new RuntimeException("Unexpected error.", e);
		}
	}

    protected static <T extends RealType<T> & NativeType<T>> 
    SharedMemoryArrayWin createSHMAFromRAI(String name, RandomAccessibleInterval<T> rai, boolean isFortranOrder, boolean isNumpy) throws FileAlreadyExistsException
    {
    	Objects.requireNonNull(rai, "Please provide a non-null RandomAccessibleInterval");
    	SharedMemoryArray.checkMemorySegmentName(name);
    	if (!name.startsWith("Local" + File.separator) && !name.startsWith("Global" + File.separator))
    		name = "Local" + File.separator+ name;
    	SharedMemoryArrayWin shma = null;
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt8(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	int size = 1;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildUint8(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt16(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	int size = 2;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildUint16(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildUint32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildInt64(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	int size = 4;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildFloat32(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	int size = 8;
        	for (long i : rai.dimensionsAsLongArray()) {size *= i;}
        	if (isNumpy) size = (int) DecodeNumpy.calculateNpyStyleByteArrayLength(rai);
        	shma = new SharedMemoryArrayWin(name, size, CommonUtils.getDataType(rai), rai.dimensionsAsLongArray(), isNumpy, isFortranOrder);
        	shma.buildFloat64(Cast.unchecked(rai), isFortranOrder, isNumpy);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
		return shma;
    }
    
    /**
     * Retrive an existing Shared memory segment and wrap it into a {@link SharedMemoryArrayLinux} 
     * @param memoryName
     * 	the name of the segment
     * @return the {@link SharedMemoryArrayLinux}  pointing to the shared memory segment
     */
    protected static SharedMemoryArrayWin read(String memoryName) {
		if (!memoryName.startsWith("Local" + File.separator) && !memoryName.startsWith("Global" + File.separator))
			memoryName = "Local" + File.separator + memoryName;
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping( WinNT.FILE_MAP_ALL_ACCESS, false, memoryName);
        if (hMapFile == null) {
            throw new RuntimeException("OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        // Map the shared memory object into the current process's address space
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(hMapFile, WinNT.FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (pSharedMemory == null) {
        	Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        Kernel32.MEMORY_BASIC_INFORMATION mbi = new Kernel32.MEMORY_BASIC_INFORMATION();
        
        if (Kernel32.INSTANCE.VirtualQueryEx(
        		Kernel32.INSTANCE.GetCurrentProcess(), pSharedMemory, mbi, new BaseTSD.SIZE_T((long) mbi.size())
        		).intValue() == 0) {
	        Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("Unable to retrieve the size of the shm segment located at '" 
        		+ memoryName + "'. Errno: " + Kernel32.INSTANCE.GetLastError());
        }
        int size = mbi.regionSize.intValue();

        Pointer writePointer = Kernel32.INSTANCE.VirtualAllocEx(Kernel32.INSTANCE.GetCurrentProcess(), 
        		pSharedMemory, new BaseTSD.SIZE_T(size), WinNT.MEM_COMMIT, WinNT.PAGE_READWRITE);
	    if (writePointer == null) {
	        Kernel32.INSTANCE.CloseHandle(hMapFile);
	        Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
            throw new RuntimeException("Error committing to the shared memory pages. Errno: "
            		+ "" + Kernel32.INSTANCE.GetLastError());
	    }
        SharedMemoryArrayWin shm = new SharedMemoryArrayWin(memoryName);
        shm.hMapFile = hMapFile;
        shm.writePointer = writePointer;
        shm.size = (int) size;
        shm.mappedPointer = pSharedMemory;
        shm.findNumpyFormat();
        return shm;
    }
    
    /**
     * Add a byte array to the shm segment
     * @param arr
     * 	the byte array that is going to be added
     */
    private void addByteArray(byte[] arr) {
    	for (int i = 0; i < arr.length; i ++) {
    		this.writePointer.setByte(i, arr[i]);
    	}
    }
    
    private static <T extends RealType<T> & NativeType<T>>
    byte[] getNpyHeader(RandomAccessibleInterval<T> tensor) {
    	String strHeader = "{'descr': '<";
    	strHeader += DecodeNumpy.getDataType(tensor.getAt(tensor.minAsLongArray()));
    	strHeader += "', 'fortran_order': False, 'shape': (";
    	for (long ll : tensor.dimensionsAsLongArray()) strHeader += ll + ", ";
    	strHeader = strHeader.substring(0, strHeader.length() - 2);
    	strHeader += "), }" + System.lineSeparator();
    	byte[] bufInverse = strHeader.getBytes(StandardCharsets.UTF_8);
    	byte[] major = {1};
        byte[] minor = {0};
        byte[] len = new byte[2];
        len[0] = (byte) (short) strHeader.length();
        len[1] = (byte) (((short) strHeader.length()) >> 8);
        int totalLen = DecodeNumpy.NUMPY_PREFIX.length + 2 + 2 + bufInverse.length;
        byte[] total = new byte[totalLen];
        int c = 0;
        for (int i = 0; i < DecodeNumpy.NUMPY_PREFIX.length; i ++)
        	total[c ++] = DecodeNumpy.NUMPY_PREFIX[i];
        total[c ++] = major[0];
        total[c ++] = minor[0];
        total[c ++] = len[0];
        total[c ++] = len[1];
        for (int i = 0; i < bufInverse.length; i ++)
        	total[c ++] = bufInverse[i];
        return total;
    }

    private void buildInt8(RandomAccessibleInterval<ByteType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<ByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setByte(offset + (i ++), cursor.get().get());
		}
    }

    private void buildUint8(RandomAccessibleInterval<UnsignedByteType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<UnsignedByteType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setByte(offset + (i ++), cursor.get().getByte());
		}
    }

    private void buildInt16(RandomAccessibleInterval<ShortType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<ShortType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setShort(offset + (i * Short.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildUint16(RandomAccessibleInterval<UnsignedShortType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<UnsignedShortType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setShort(offset + (i * Short.BYTES), cursor.get().getShort());
			i ++;
		}
    }

    private void buildInt32(RandomAccessibleInterval<IntType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<IntType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setInt(offset + (i * Integer.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildUint32(RandomAccessibleInterval<UnsignedIntType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<UnsignedIntType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setInt(offset + (i * Integer.BYTES), cursor.get().getInt());
			i ++;
		}
    }

    private void buildInt64(RandomAccessibleInterval<LongType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<LongType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setLong(offset + (i * Long.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildFloat32(RandomAccessibleInterval<FloatType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<FloatType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setFloat(offset + (i * Float.BYTES), cursor.get().get());
			i ++;
		}
    }

    private void buildFloat64(RandomAccessibleInterval<DoubleType> tensor, boolean isFortranOrder, boolean isNumpy)
    {
    	byte[] header = new byte[0];
    	if (isNumpy) header = getNpyHeader(tensor);
    	long offset = 0;
    	for (byte b : header) {
			this.mappedPointer.setByte(offset, b);
    		offset ++;
    	}
    	if (!isFortranOrder) tensor = Utils.transpose(tensor);
		Cursor<DoubleType> cursor = Views.flatIterable(tensor).cursor();
		long i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			this.mappedPointer.setDouble(offset + (i * Double.BYTES), cursor.get().get());
			i ++;
		}
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public String getName() {
    	return this.memoryName;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public String getNameForPython() {
    	return this.memoryName.substring(("Local" + File.separator).length());
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public Pointer getPointer() {
    	return this.mappedPointer;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public HANDLE getSharedMemoryBlockID() {
    	return this.hMapFile;
    }

    /**
     * {@inheritDoc}
     */
	@Override
    public int getSize() {
    	return this.size;
    }

	@Override
    /**
     * {@inheritDoc}
     */
	public String getOriginalDataType() {
		return this.originalDataType;
	}

	@Override
    /**
     * {@inheritDoc}
     */
	public long[] getOriginalShape() {
		return this.originalDims;
	}
	
	@Override
    /**
     * {@inheritDoc}
     */
	public boolean isNumpyFormat() {
		if (this.isNumpyFormat == null) {
			findNumpyFormat();
		}
		return this.isNumpyFormat;
	}

	@Override
	/**
	 * {@inheritDoc}
	 * 
	 * Unmap and close the shared memory. Necessary to eliminate the shared memory block
	 */
	public void close() {
		if (unlinked) return;
		if (writePointer != null) Kernel32.INSTANCE.UnmapViewOfFile(this.writePointer);
        Kernel32.INSTANCE.UnmapViewOfFile(mappedPointer);
        Kernel32.INSTANCE.CloseHandle(hMapFile);
        unlinked = true;
	}
	
	/**
	 * Find whether the shared memory segment wrapped by the {@link SharedMemoryArray} is in Numpy npy format of not.
	 */
	private void findNumpyFormat() {
		this.isNumpyFormat = true;
		try {
			int offset = 0;
			byte[] buf = new byte[DecodeNumpy.NUMPY_PREFIX.length];
			mappedPointer.getByteBuffer(offset, DecodeNumpy.NUMPY_PREFIX.length).get(buf, 0, DecodeNumpy.NUMPY_PREFIX.length);
	        if (!Arrays.equals(buf, DecodeNumpy.NUMPY_PREFIX)) {
	            throw new IllegalArgumentException("Malformed  or unsopported Numpy array");
	        }
	        offset = DecodeNumpy.NUMPY_PREFIX.length;
	        byte major = mappedPointer.getByteBuffer(offset, 1).get();
	        offset ++;
	        byte minor = mappedPointer.getByteBuffer(offset, 1).get();
	        offset ++;
	        if (major < 1 || major > 3 || minor != 0) {
	            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
	        }
	        int len = major == 1 ? 2 : 4;
	        ByteBuffer bb = mappedPointer.getByteBuffer(offset, len);
	        offset += len;
	        bb.order(ByteOrder.LITTLE_ENDIAN);
	        if (major == 1) {
	            len = bb.getShort();
	        } else {
	            len = bb.getInt();
	        }
	        buf = new byte[len];
	        mappedPointer.getByteBuffer(offset, len).get(buf, 0, len);
	        offset += len;
	        String header = new String(buf, StandardCharsets.UTF_8);
	        Matcher m = DecodeNumpy.HEADER_PATTERN.matcher(header);
	        if (!m.find()) {
	            throw new IllegalArgumentException("Invalid numpy header: " + header);
	        }
	        String typeStr = m.group(1);
	        String fortranOrder = m.group(2).trim();
	        String shapeStr = m.group(3);
	        long[] shape = new long[0];
	        if (!shapeStr.isEmpty()) {
	            String[] tokens = shapeStr.split(", ?");
	            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
	        }
	        char order = typeStr.charAt(0);
	    	if (order != '>' && order != '<' && order != '|') {
	        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
	        }
	        String dtype = DecodeNumpy.getDataType(typeStr.substring(1));
	        this.originalDims = shape;
	        this.originalDataType = dtype;
	        this.isFortran = fortranOrder.equals("True");
		} catch (Exception ex) {
			this.isNumpyFormat = false;
		}
	}
    
    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI() {
    	if ((this.originalDims == null || this.originalDataType == null) && !this.isNumpyFormat())
    		throw new IllegalArgumentException("The shared memory segment is not stored in Numpy format and the shape and/or "
    				+ "data type are not known. Please provide information about them and use the method "
    				+ "'getSharedRAI(long[] shape, boolean isFortran, T dataType)'.");
    	if (this.isNumpyFormat()) {
    		return buildImgLib2FromNumpyLikeSHMA();
    	} else {
    		return buildFromSharedMemoryBlock(this.mappedPointer, this.originalDims, this.originalDataType, this.isFortran);
    	}
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType) {
		return buildFromSharedMemoryBlock(mappedPointer, shape, dataType, isFortran, 0, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getSharedRAI(long[] shape, T dataType, boolean isFortran) {
		return buildFromSharedMemoryBlock(mappedPointer, shape, dataType, isFortran, 0, ByteOrder.LITTLE_ENDIAN);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void setBuffer(ByteBuffer buffer) {
    	if (buffer.capacity() > this.size) {
    		throw new IllegalArgumentException("The buffer capacity has to be smaller or equal "
    				+ "than the size of the shared memory segment.");
    	}
    	this.mappedPointer.write(0, buffer.array(), 0, buffer.capacity());
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public ByteBuffer getDataBuffer() {
    	return mappedPointer.getByteBuffer(0, this.size);
    }

	@Override
	/**
	 * {@inheritDoc}
	 */
	public ByteBuffer getDataBufferNoHeader() {
    	int offset = 0;
    	if (this.isNumpyFormat()) {
    		long flatSize = 1;
    		for (long l : this.originalDims) flatSize *= l;
    		offset =  (int) (this.size - DecodeNumpy.DATA_TYPES_MAP.get(this.originalDataType) * flatSize);
    	}
    	return mappedPointer.getByteBuffer(offset, this.size - offset);
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildFromSharedMemoryBlock(Pointer pSharedMemory, long[] shape, String type, boolean isFortran) {
		T dataType = CommonUtils.getImgLib2DataType(type);
		return buildFromSharedMemoryBlock(pSharedMemory, shape, dataType, isFortran, 0, ByteOrder.LITTLE_ENDIAN);
	}

	// TODO support boolean
	protected <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildImgLib2FromNumpyLikeSHMA() {
		int offset = 0;
		byte[] buf = new byte[DecodeNumpy.NUMPY_PREFIX.length];
		mappedPointer.getByteBuffer(offset, DecodeNumpy.NUMPY_PREFIX.length).get(buf, 0, DecodeNumpy.NUMPY_PREFIX.length);
        if (!Arrays.equals(buf, DecodeNumpy.NUMPY_PREFIX)) {
            throw new IllegalArgumentException("Malformed  or unsopported Numpy array");
        }
        offset = DecodeNumpy.NUMPY_PREFIX.length;
        byte major = mappedPointer.getByteBuffer(offset, 1).get();
        offset ++;
        byte minor = mappedPointer.getByteBuffer(offset, 1).get();
        offset ++;
        if (major < 1 || major > 3 || minor != 0) {
            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
        }
        int len = major == 1 ? 2 : 4;
        ByteBuffer bb = mappedPointer.getByteBuffer(offset, len);
        offset += len;
        bb.order(ByteOrder.LITTLE_ENDIAN);
        if (major == 1) {
            len = bb.getShort();
        } else {
            len = bb.getInt();
        }
        buf = new byte[len];
        mappedPointer.getByteBuffer(offset, len).get(buf, 0, len);
        offset += len;
        String header = new String(buf, StandardCharsets.UTF_8);
        Matcher m = DecodeNumpy.HEADER_PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        String fortranOrder = m.group(2).trim();
        String shapeStr = m.group(3);
        long[] shape = new long[0];
        if (!shapeStr.isEmpty()) {
            String[] tokens = shapeStr.split(", ?");
            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
        }
        char order = typeStr.charAt(0);
        ByteOrder byteOrder = null;
        if (order == '>') {
        	byteOrder = ByteOrder.BIG_ENDIAN;
        } else if (order == '<') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        } else if (order == '|') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        	new IOException("Numpy .npy file did not specify the byte order of the array."
        			+ " It was automatically opened as little endian but this does not guarantee"
        			+ " the that the file is open correctly. Caution is advised.").printStackTrace();
    	} else {
        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
        }
        String dtype = DecodeNumpy.getDataType(typeStr.substring(1));
        long numBytes = DecodeNumpy.DATA_TYPES_MAP.get(dtype);
    	long count;
    	if (shape.length == 0)
    		count = 1;
		else
			count = Arrays.stream(shape).reduce(Math::multiplyExact).getAsLong();
        len = Math.toIntExact(count * numBytes);
        
        return buildFromSharedMemoryBlock(this.mappedPointer, shape, 
        		Cast.unchecked(CommonUtils.getImgLib2DataType(dtype)), fortranOrder.equals("True"), offset, byteOrder);
	}

	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> buildFromSharedMemoryBlock(Pointer pSharedMemory, long[] shape, T dataType, 
			boolean isFortran, int offset, ByteOrder order) {
		long[] transposedShape = new long[shape.length];
		for (int i = 0; i < shape.length; i ++) {transposedShape[i] = shape[shape.length - i - 1];}
		if (dataType instanceof ByteType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.bytes(access, transposedShape)));
		} else if (dataType instanceof ByteType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.bytes(access, shape));
		} else if (dataType instanceof UnsignedByteType && isFortran) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.unsignedBytes(access, shape));
		} else if (dataType instanceof UnsignedByteType) {
			int arrSize = 1;
			for (long l : shape) {arrSize *= l;}
    		ByteAccess access = new ByteBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedBytes(access, transposedShape)));
		} else if (dataType instanceof ShortType && isFortran) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.shorts(access, shape));
		} else if (dataType instanceof ShortType) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.shorts(access, transposedShape)));
		} else if (dataType instanceof UnsignedShortType && isFortran) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.unsignedShorts(access, shape));
		} else if (dataType instanceof UnsignedShortType) {
			int arrSize = 2;
			for (long l : shape) {arrSize *= l;}
			ShortAccess access = new ShortBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedShorts(access, transposedShape)));
		} else if (dataType instanceof IntType && isFortran) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.ints(access, shape));
		} else if (dataType instanceof IntType) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.ints(access, transposedShape)));
		} else if (dataType instanceof UnsignedIntType && isFortran) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.unsignedInts(access, shape));
		} else if (dataType instanceof UnsignedIntType) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			IntAccess access = new IntBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.unsignedInts(access, transposedShape)));
		} else if (dataType instanceof LongType && isFortran) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			LongAccess access = new LongBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.longs(access, shape));
		} else if (dataType instanceof LongType) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			LongAccess access = new LongBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.longs(access, transposedShape)));
		} else if (dataType instanceof FloatType && isFortran) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			FloatAccess access = new FloatBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.floats(access, shape));
		} else if (dataType instanceof FloatType) {
			int arrSize = 4;
			for (long l : shape) {arrSize *= l;}
			FloatAccess access = new FloatBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.floats(access, transposedShape)));
		} else if (dataType instanceof DoubleType && isFortran) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			DoubleAccess access = new DoubleBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(ArrayImgs.doubles(access, shape));
		} else if (dataType instanceof DoubleType) {
			int arrSize = 8;
			for (long l : shape) {arrSize *= l;}
			DoubleAccess access = new DoubleBufferAccess(pSharedMemory.getByteBuffer(offset, arrSize).order(order), true);
			return Cast.unchecked(Utils.transpose(ArrayImgs.doubles(access, transposedShape)));
		} else {
    		throw new IllegalArgumentException("Type not supported: " + dataType.getClass().toString());
		}
	}
	
	/**
	 * Windows shared memory persists until every reference to it from every process 
	 * unlinks it. Is not the same as in Unix systems where just one reference to unlink unlinks it
	 * completely and no othere process can come and use it anymore
	 * 
	 * @param args
	 * 	random
	 */
	public static void main(String[] args) {
		WinNT.HANDLE hMapFile = Kernel32.INSTANCE.OpenFileMapping( WinNT.FILE_MAP_READ, false, "Local\\wnsm_89f5c80c");

    	if (hMapFile == null) {
            throw new RuntimeException("OpenFileMapping failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        // Map the shared memory object into the current process's address space
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(hMapFile, WinNT.FILE_MAP_READ, 0, 0, 0);
        if (pSharedMemory == null) {
        	Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed with error: " + Kernel32.INSTANCE.GetLastError());
        }
        Kernel32.MEMORY_BASIC_INFORMATION mbi = new Kernel32.MEMORY_BASIC_INFORMATION();
        
        if (Kernel32.INSTANCE.VirtualQueryEx(
        		Kernel32.INSTANCE.GetCurrentProcess(), pSharedMemory, mbi, new BaseTSD.SIZE_T((long) mbi.size())
        		).intValue() == 0) {
            throw new RuntimeException("Unable to retrieve the size of the shm segment located at '" 
        		+ "'. Errno: " + Kernel32.INSTANCE.GetLastError());
        }
        int size = mbi.regionSize.intValue();

        Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
    	Kernel32.INSTANCE.CloseHandle(hMapFile);
	}
}
