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

public class NativeMemoryMapping {
	public interface CLibrary extends com.sun.jna.Library {
        CLibrary INSTANCE = (CLibrary)
        		com.sun.jna.Native.load("Kernel32", CLibrary.class);

        public int CreateFileMapping(int arg1, int arg2, int arg3, int arg4, int arg5, String arg6);
    }

    public static void main(String[] args) {
        CLibrary.INSTANCE.CreateFileMapping("Hello, World\n");
        for (int i=0;i < args.length;i++) {
            CLibrary.INSTANCE.printf("Argument %d: %s\n", i, args[i]);
        }
    }
}
