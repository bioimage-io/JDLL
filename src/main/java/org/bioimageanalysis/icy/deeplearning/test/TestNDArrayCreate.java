package org.bioimageanalysis.icy.deeplearning.test;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;


import ai.djl.ndarray.NDManager;

public class TestNDArrayCreate {

	public static void main(String[] args) throws MalformedURLException {
		String enginePath = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\engines\\Tensorflow-1.15.0-1.15.0-windows-x86_64-cpu";
		URL[] urls = new URL[new File(enginePath).listFiles().length];
		int c = 0;
		for (File ff : new File(enginePath).listFiles()) {
			urls[c ++] = ff.toURI().toURL();
		}
	    URLClassLoader engineClassloader = new URLClassLoader(urls);
	    Thread.currentThread().setContextClassLoader(engineClassloader);
	    
		
		NDManager.newBaseManager();
		System.out.print("");
	}
}
