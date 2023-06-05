package io.bioimage.modelrunner;

import java.io.File;
import java.io.IOException;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;

public class PythonConnection {

	
	public static void main(String[] args) {
		Environment env = Appose.conda(new File("/path/to/environment.yml")).build();
		try (Service python = env.python()) {
		    Task task = python.task("5 + 6");
		    task.waitFor();
		    Object result = task.outputs.get("result");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
