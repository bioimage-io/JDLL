package org.bioimageanalysis.icy.deeplearning.test;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadEngineException;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Platform;

public class TestDjlJavaApi {

	public static void main(String[] args) throws LoadEngineException, Exception {
		NDManager aa = NDManager.newBaseManager();
		NDArray a = aa.ones(new Shape(new long[] {8,8}));
		System.out.print(true);
		String enginesDir = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines";
		boolean cpu = true;
		boolean gpu = true;
		URL[] urls = new URL[6];
		urls[0] = new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Pytorch-1.7.1-1.7.1-windows-x86_64-cpu-gpu\\pytorch-engine-0.10.0.jar").toURI().toURL();
		urls[1] = new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Pytorch-1.7.1-1.7.1-windows-x86_64-cpu-gpu\\pytorch-native-auto-1.7.1.jar").toURI().toURL();
		urls[2] = new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Pytorch-1.7.1-1.7.1-windows-x86_64-cpu-gpu\\api-0.10.0.jar").toURI().toURL();
		urls[3] = new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Pytorch-1.7.1-1.7.1-windows-x86_64-cpu-gpu\\jna-5.3.0.jar").toURI().toURL();
		urls[4] = new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Pytorch-1.7.1-1.7.1-windows-x86_64-cpu-gpu\\commons-compress-1.20.jar").toURI().toURL();
		urls[5] = new File("C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Pytorch-1.7.1-1.7.1-windows-x86_64-cpu-gpu\\gson-2.8.5.jar").toURI().toURL();
		URLClassLoader classloader = new URLClassLoader(urls);
		Thread.currentThread().setContextClassLoader(classloader);
		Class<?> c = Thread.currentThread().getContextClassLoader().loadClass("ai.djl.pytorch.engine.PtEngineProvider");
		Class<?> c2 = Thread.currentThread().getContextClassLoader().loadClass("ai.djl.pytorch.engine.PtEngine");
		Class<?> c3 = Thread.currentThread().getContextClassLoader().loadClass("ai.djl.pytorch.jni.LibUtils");
		Method loadM = c3.getMethod("loadLibrary");
		loadM.invoke(null);
		Method mm = c2.getMethod("getInstance");
		Engine enginept = (Engine) mm.invoke(null);
		EngineProvider ep = (EngineProvider) c.newInstance();
		TestWrapper ep2 = new TestWrapper(ep);
		ep2.setEngine(enginept);
		Engine.registerEngine(ep2);
		ep2.getEngine();
		
		
		EngineInfo engineInfo = EngineInfo.defineDLEngine("torchscript", "1.7.1", enginesDir, cpu, gpu);
		
		String modelFolder = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\EnhancerMitochondriaEM2D_18062022_023545";
		String modelSource = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\EnhancerMitochondriaEM2D_18062022_023545\\weights-torchscript.pt";
		
		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo, null);
		model.getEngineClassLoader().setEngineClassLoader();
		model.loadModel();
		System.out.println("done");
	}
}
