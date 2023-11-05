package io.bioimage.modelrunner;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Types;

import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.numeric.real.FloatType;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.platform.win32.Kernel32;
import com.sun.jna.platform.win32.WinNT;
import com.sun.jna.platform.win32.WinBase;

public class NativeMemoryMapping {

	public static void main(String[] args) {
        final String sharedMemoryName = "Local\\MySharedMemory";
        final int size = 1024; // Size of the shared memory block
        
        // Create a file mapping for shared memory
        WinNT.HANDLE hMapFile = Kernel32.INSTANCE.CreateFileMapping(
                WinBase.INVALID_HANDLE_VALUE,
                null,
                WinNT.PAGE_READWRITE,
                0,
                size,
                sharedMemoryName
        );
        
        if (hMapFile == null) {
            throw new RuntimeException("CreateFileMapping failed");
        }
        
        // Map the shared memory
        Pointer pSharedMemory = Kernel32.INSTANCE.MapViewOfFile(
                hMapFile,
                WinNT.FILE_MAP_WRITE,
                0,
                0,
                size
        );
        
        if (pSharedMemory == null) {
            Kernel32.INSTANCE.CloseHandle(hMapFile);
            throw new RuntimeException("MapViewOfFile failed");
        }
        
        // Now you can write to the shared memory as if it were an array of longs
        for (int i = 0; i < size / Long.BYTES; i++) {
        	pSharedMemory.setLong((long) i * Long.BYTES, i);
        }
        
        // Read from shared memory
        for (int i = 0; i < size / Long.BYTES; i++) {
            long value = pSharedMemory.getLong((long) i * Long.BYTES);
            System.out.println("Value at index " + i + ": " + value);
        }
        
        // Unmap and close the shared memory (in a real application, do this in a finally block)
        Kernel32.INSTANCE.UnmapViewOfFile(pSharedMemory);
        Kernel32.INSTANCE.CloseHandle(hMapFile);
    }
}
