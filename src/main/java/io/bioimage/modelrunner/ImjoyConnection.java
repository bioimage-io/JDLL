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

public class ImjoyConnection {
	
	private static final String NAME_KEY = "name";
	
	private static final String AXES_KEY = "axes";
	
	private static final String DATA_KEY = "data";
	
	private static final String CODE = 
			"import asyncio" + System.lineSeparator()
			+ "import numpy as np" + System.lineSeparator()
			+ "async def run(image, task):\r\n"
			+ "  from imjoy_rpc.hypha import connect_to_server\r\n"
			+ "\r\n"
			+ "  SERVER_URL = \"https://ai.imjoy.io\"\r\n"
			+ "  server = await connect_to_server(\r\n"
			+ "      {\"name\": \"test client\", \"server_url\": SERVER_URL}\r\n"
			+ "  )\r\n"
			+ "  ops = await server.get_service(\"bioengine-ops\")\r\n"
			+ "  mask = await ops.cellpose_predict(image)\r\n"
			+ "  task.update(\"Cellpose run\")\r\n"
			+ "  task.outputs['out'] = mask.flatten().tolist()"+ System.lineSeparator()
			+ "asyncio.run(run(np.array(tensor).reshape(3, 256, 256).astype('float32'), task))";
	
	
	public static void main(String[] args) {
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		final Img< FloatType > img1 = imgFactory.create( 1, 3, 256, 256 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
		List<Tensor<FloatType>> inputs = new ArrayList<Tensor<FloatType>>();
		inputs.add(inpTensor);

		

		LinkedHashMap<String, Object> inputMap = new LinkedHashMap<>();
		for (Tensor<FloatType> input : inputs) {
			HashMap<String, Object> tensorMap = new HashMap<String, Object>();
			//tensorMap.put(AXES_KEY, input.getAxesOrderString());
			//tensorMap.put(DATA_KEY, ImgLib2ToArray.buildFloat(input.getData()));
			// inputMap.put(NAME_KEY, tensorMap);
			inputMap.put("tensor", ImgLib2ToArray.build(input.getData()));
		}
        
        Environment env = Appose.base(new File("C:\\Users\\angel\\git\\jep\\miniconda\\envs\\stardist")).build();
        try (Service python = env.python()) {
        	python.debug(line -> {
        		System.err.println(line);
        	});
            Task task = python.task(CODE, inputMap);
            System.out.println("here");
            task.listen(event -> {
                switch (event.responseType) {
	                case UPDATE:
	                    System.out.println("Progress: " + task.message);
	                    break;
	                case COMPLETION:
	                    Object numer =  task.outputs.get("out");
	                    System.out.println("Task complete. Result: " + numer);
	                    break;
	                case CANCELATION:
	                    System.out.println("Task canceled");
	                    break;
	                case FAILURE:
	                    System.out.println("Task failed: " + task.error);
	                    break;
                }
            });
            task.waitFor();
            System.out.println("here2");
            Object result = task.outputs.get("out");
            System.out.println("here3");
            if (result instanceof Integer)
            	System.out.print(result);
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		
	}
}
