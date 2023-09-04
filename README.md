[![Build Status](https://github.com/bioimage-io/model-runner-java/actions/workflows/build.yml/badge.svg)](https://github.com/bioimage-io/model-runner-java/actions/workflows/build.yml)

# JDLL (Java Deep Learning Library)

This project provides a Java library for running Deep Learning (DL) models agnostically, enabling communication between Java software and various Deep Learning frameworks (engines). It also allows the use of multiple DL frameworks in the same session, manages the different DL frameworks and brings the models from the 
[Bioimage.io](https://bioimage.io/#/) repository to Java.

It is intended for developers and was originally built by the DeepIcy team as the backend of the DeepIcy plugin.

[JDLL](<https://github.com/bioimage-io/model-runner-java/tree/main>) is able to load models and make inference, create tensors, download Bioiamge.io models and manage the supported DL frameworks. The library is designed in a modular way, allowing the main software to avoid dealing with the various objects and structures required by different DL frameworks. Instead the Java model runner provides interfaces for models and tensors that handle internally their creation and inference in the differnet Java engines. The main software only needs to interact with the Java model runner and does not need to worry whether the model is in PyTorch, Tensorflow or other framework.

**FOR A MORE COMPREHENSIVE AND COMPLETE EXPLANATION OF JDLL, PLEASE VISIT THE [WIKI](https://github.com/bioimage-io/JDLL/wiki).**

# Supported engines

Currently, the following frameworks are supported:

| Framework                          | Source code                                                    |
|---------------------------------|----------------------------------------------------------------|
| PyTorch                         | https://github.com/bioimage-io/pytorch-java-interface          |
| Tensorflow 1                    | https://github.com/bioimage-io/tensorflow-1-java-interface     |
| Tensorflow 2 API 0.2.0          | https://github.com/bioimage-io/tensorflow-2-java-interface-0.2 |
| Tensorflow 2 all APIs but 0.2.0 | https://github.com/bioimage-io/tensorflow-2-java-interface     |
| Onnx                            | https://github.com/bioimage-io/onnx-java-interface             |

The information about the engines supported currently by the model runner, for which OS and architectures and which JAR files are required for each of the engines is stored in [this json file](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/resources/availableDLVersions.json) and can be found [here](https://github.com/bioimage-io/JDLL/wiki/List-of-supported-engines).

Note that the model runner will be in **constant development** and that it is open to community collaboration, so **pull requests** to the official repository of the model runner to improve functionality or to add new engines are **very welcomed**.


# Quickstart

## 0. Setting Up the Model Runner

Download the dependency and include it in your project

   In order to benefit from the library, include the dependency in your code. The dependency can be added manually or using a dependency manager such as Maven. If you are using Maven, add the following dependency to the project pom file:

   ```xml
   <dependency>
     <groupId>io.bioimage</groupId>
     <artifactId>dl-modelrunner</artifactId>
     <version>0.3.10</version>
   </dependency>
   ```

   and add to `<repositories>` the following:

   ```xml
   <repository>
     <id>scijava.public</id>
     <url>https://maven.scijava.org/content/groups/public</url>
   </repository>
   ```
## 1. Getting a model (optional)
If a model from the supported by JDLL is already available you can skip this step. Note that for Tensorflow the models need to be saved in the [`SavdModel`](https://www.tensorflow.org/guide/saved_model) format.

If no model is available, a good starting point is downloading any of the models of the [Bioimage.io respository](https://bioimage.io/#/). The download can be done manually or using JDLL. Bioimag.io is seamlessly integrated into JDLL, offering multiple methods to effortlessly mange and use its models.

Here is an emaple of how JDLL can be used to download any Bioimage.io model, in this case the [`B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet`](https://bioimage.io/#/?tags=placid-llama&id=10.5281%2Fzenodo.7261974) model.
```java
// Name of the model of interest, note that each model is unique. The names are case sensitive.
String modelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet";
// Directory where the model folder will be downloaded
String modelsDir = "/path/to/wanted/model/directory";

// First create an instance of the Bioimage.io repository
BioimageioRepo br = BioimageioRepo.connect();
try {
	br.downloadByName(modelName, modelsDir);
        System.out.println("Great success!");
} catch (IOException | InterruptedException e) {
	// If the download is interrumpted or any of the model files cannot be downloaded
	// and exception will be thrown
	e.printStackTrace();
        System.out.println("Error downloading the model :(");
}
```
Output:
```
Great success!
```
More information on how to download  Bioimage.io models can be found [here](https://github.com/bioimage-io/JDLL/wiki/Engine-Installation-(EngineManagement)).



## 2. Installing DL engines
JDLL is installed empty. Several models might require different Deep Learning framework versions, each of them consuming considerable amounts of disk space. In order to make JDLL setup light and fast JDLL is installed without default DL engines. The user can then get the Dl engines that they want depending on their needs.

JDLL provides the needed methods to install the wanted engines in an easy manner. Following the above example, find below some code that can be used to install a DL engine. As it can be observed the model that was downloaded [supports Tensorflow 2 and Keras weights](https://github.com/bioimage-io/collection-bioimage-io/blob/19ea59e662410c3ee49b7da184730919336d7568/rdfs/10.5281/zenodo.7261974/7782776/rdf.yaml#L146). Keras is not supported so in order to load and run the model, Tensorflow weights need to be installed.

```
String framework = "tensorflow";
String version = "2.11.0";
boolean cpu = true;
boolean gpu = true;

String enginesDir = "/path/to/wanted/engines/dir";
boolean installed = EngineManagement.installEngineWithArgsInDir(framework, version, cpu, gpu, enginesDir);
if (installed)
	System.out.println("Great success!");
else
	System.out.println("Error installing");
```
Output:
```
Great success!
```

As previously mentioned, JDLL integrates deeply with Bioimage.io models. An easier way to install the engines needed for Bioimage.io models is shown in the code below.

In the example it is shown how simply providing the name of the model of interest, JDLL will know which engines to install.
```java
String modelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet";
String enginesDir = "/path/to/wanted/engines/dir";
boolean installed =  installEnginesForModelByNameinDir(modelName, enginesDir)
if (installed)
	System.out.println("Great success!");
else
	System.out.println("Error installing");
```
Output:
```
Great success!
```
The Wiki convers extensively engine installation ([here](https://github.com/bioimage-io/JDLL/wiki/Understanding-engine-installation) and [here](https://github.com/bioimage-io/JDLL/wiki/Engine-Installation-(EngineManagement))). In addtion JDLL also includes methods to manage the engines and know: [the information about each engine](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-I-(DeepLearningVersion)), [which engines are supported](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-II-(AvailableEngines)) and [which engines have already been installed](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-III-(InstalledEngines))



## 3. Creating the tensors
Once the model and the engine are already installed it is the moment to start the process of running the model on the tensors. In this section, creation of the tensors will be explained.

JDLL tensors are agnostic to the DL framework to be used, they are always creted in the same way. JDLL manages internally the conversion of the agnostif tensor into the framework specific tensor once the mdoel is going to be run. The unified method of creating tensors facilitates the integration of every supported DL framework into any software.

JDLL tensors use ImgLib2 to store the tensor information. In practice, JDLL tensors are just wrappers of ImgLib2 `RandomAccessibleIntervals` that contain all the data needed to convert them back and forth into the framework specific tensors.

The example below will show how to create the input and output tensors required to run the [example model](https://bioimage.io/#/?tags=placid-llama&id=10.5281%2Fzenodo.7261974). As per its [rdf.yaml file](https://github.com/bioimage-io/collection-bioimage-io/blob/19ea59e662410c3ee49b7da184730919336d7568/rdfs/10.5281/zenodo.7261974/7782776/rdf.yaml), the model has one input named `input_1`, with `bxyc` axes ([explanation here](https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md)) and a required shape of[1, 512, 512, 1]. The ouptut of the model is named `conv2d_19` with with `bxyc` axes and fixed shape [1, 512, 512, 3].
```
final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
final Img< FloatType > img1 = imgFactory.create( 1, 512, 512, 1 );
// Create the input tensor with the nameand axes given by the rdf.yaml file
// and add it to the list of input tensors
Tensor<FloatType> inpTensor = Tensor.build("input_1", "bxyc", img1);

// Ouput tensors can be created empty, if the output shape is not known.
// Note that this method does not preallocate memory for the output tensor
Tensor<T> ouptutEmptyTensor = Tensor.buildEmptyTensor("conv2d_19", "bxyc");

// Or ouptut tensors can also be built blank, to pre-allocate memory
// if the shape and data type are known.
Tensor<FloatType> ouptutBlankTensor = Tensor.buildBlankTensor("conv2d_19",
			"bxyc", new long[] {1, 512, 512, 3}, new FloatType());

```

More information about tensors can be found in the [JDLL wiki](https://github.com/bioimage-io/JDLL/wiki/JDLL-tensors-I-(Tensor)).


## 4. Loading the model



## 5. Running the model
Once the model has been loaded and the input and output tensors have been created. Running the model is simple. The input tensors should be added to a `List<?>` in the same order the model expects. Same for the ouptuts in another `List<?>`.
```
List<Tensor<?>> inputList = new ArrayList<List<Tensor<?>>>();
List<Tensor<?>> outputputList = new ArrayList<List<Tensor<?>>>();

inputList.add(inputTensor);
outputputList.add(outputEmptyTensor);
System.out.println("Ouptut tensor is empty: " + outputEmptyTensor.isEmpty());

model.runModel(inputList, outputputList);
System.out.println("Ouptut tensor after inference is empty: " + outputEmptyTensor.isEmpty());
```
Output:
```
Ouptut tensor is empty: true
Ouptut tensor after inference is empty: false
```


## 6. Closing the model and the tensors
Models and tensors need to be closed to be released and free the memory that they were using
```
model.close();
inputTensor.close;
ouptutBlankTensor.close();
ouptutEmptyTensor.close();
```



## Loading and running a model with JDLL

The Java model runner was developed with the objective of being as easy as possible to implement in already existing Java softwares.
There are three key points: loading a model, creating the tensors, and making inference with the model on the tensors.

### 1. Loading a model

In order to load a model, the library needs to know first in which framework the model is going to be loaded, and then where is the model of interest.

The user needs to give information about the DL framework. For that the creation of an object called [`EngineInfo`](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/engine/EngineInfo.java) is required. An `EngineInfo` object has to be created with the framework name that is given by the [Bioimage.io specs](https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md). **Tensorflow** should be `tensorflow_saved_model_bundled`, **PyTorch for Java**, `torchscript` and **Onnx**, `onnx`.

The other required parameters are the version of the framework in Python (sometimes it differs from the Java API version) that wants to be loaded (1.15.0, 1.9.1, 15...) and the directory where all the engines folders are stored. Looking at the previous example this directory would be `C:\Users\carlos\icy\engines`.
With this information an example code snippet would be:

```java
EngineInfo engineInfo = EngineInfo.defineDLEngine("pytorch", "1.9.1", "C:\Users\carlos\icy\engines");
```

The `engineInfo` object is needed to know which of the engines has to be loaded. **Note that `EngineInfo.defineDLEngine(...)` will only try to load the exact same engine that is specified.** If it is not installed the method will fail when trying to load the engine.

In order to check if a engine version is installed:

```
String engine = "tensorflow";
String version = "1.13.1";
String enginesDir = "/path/to/engines";
boolean installed = InstalledEnginescheckEngineVersionInstalled(engine, version, enginesDir);
```

It is also possible to **load an engine version compatible with the wanted one**. Compatible engine versions are those from teh same DL frameworks that share the same major version number. For example Pytorch 1.13.1 and 1.11.0 are compatible but Tensorflow 1.15.0 and Tensorflow 2.7.0 are NOT compatible.

The following method can be used to try to load a compatible engine version if the particular version does not exist:

```java
EngineInfo engineInfo = EngineInfo.defineCompatibleDLEngine("pytorch", "1.9.1", "C:\Users\carlos\icy\engines");
```

In this case, if Pytorch 1.9.1 is not installed but Pytorch 1.13.1 is, loading the model will load using Pytorch 1.13.1 instead of failing. In order to know which version has been loaded:
```
System.out.println(engineInfo.getVersion());
```

**NOTE THAT THIS MIGHT BE A SOURCE OF ERRORS AS NOT EVERY ENGINE JDLL DEFINES AS COMPATIBLE IS ACTUALLY COMPATIBLE.** If Pytorch 1.12.0 includes a new functionality that was not included in Pytorch 1.9.1 and we try to load a Pytorch 1.12.0 model that uses that functionality with the Pytorch 1.9.1 engine,
**WE WILL GET AN ERROR IN THE MODEL INFERENCE STEP.**

This engine info must be used to load the corresponding model. Model loading requires 3 parameters, the model folder (directory where all the files for a model are stored), the model source (path to the file that is specified in the weights&rarr;source field in the `rdf.yaml` file) and the `EngineInfo` object previously created.

An example code to load a model would be:

```java
String modelPath = "C:\Users\carlos\icy\models\EnhancerMitochondriaEM2D_13102022_171141";
String modelSource = modelPath + "weights-torchscript.pt";
Model model = Model.createDeepLearningModel(modelPath, modelSource, engineInfo);
```

The above piece of code would call the corresponding engine instance in a separate classloader and load the model in its corresponding engine. This model can now be used to make inference.

### 2. Creating agnostic tensors

The java model runner implements its own agnostic tensors that act as a vehicle to communicate between the main Java software and the Java Deep Learning framework.

Thanks to the agnostic tensors the main program does not have to deal with the creation of different tensors depending on the DL framework, unifying the task.

Agnostic tensors use ImgLib2 `RandomAccessibleInterval`s as the backend to store the data. ImgLib2 provides an all-in-Java fast and lightweight framework to handle the data and communicate with particular Deep Learning frameworks.
The creation of tensors in the main program side is reduced to the creation of ImgLib2 `RandomAccessibleInteval`s (or objects that extend them).

Once the ImgLib2 object is created, the creation of a model runner tensor is simple. Apart from the data as ImgLib2 it requires the name of the tensor and the axes order of the tensor (as defined in the `rdf.yaml`).

An example would be:

```java
RandomAccessibleInterval<FloatType> data = ...;
Tensor tensor = Tensor.build("name", "bcyx", data);
```

Note that it is also necessary to generate the agnostic tensors that correspond to the output of the model.

These tensors are going to host the results of the inference.

Output tensors can be created as empty tensors and only contain the name and axes order of the output tensor:

```java
// Without allocation of memory
Tensor.buildEmptyTensor("outputName", "bcyx");
// Allocating memory
Tensor<FloatType> outTensor = Tensor.buildEmptyTensorAndAllocateMemory("output0",
                                                                       "bcyx",
                                                                       new long[] {1, 2, 512, 512},
                                                                       new FloatType());
```

Or can be constructed with an ImgLib2 object with the expected shape and data type of the output to allocate memory prior to execution.

```java
RandomAccessibleInterval<FloatType> expectedData = ...;
Tensor output = Tensor.build("outputName", "bcyx", expectedData);
```

### 3. Making inference

Once the model and tensors have been defined, everything is ready to make inference.

The process should be relatively easy to implement in the main software.

All the input tensors should be put together in a `List`, same for the output tensors. Then the model should be called as `model.runModel(....)`. The output list of tensors is then updated inplace.

```java
// List that will contain the input tensors
List<Tensors> inputTensors = new ArrayList<Tensor>();
// List that will contain the output tensors
List<Tensors> outputTensors = new ArrayList<Tensor>();
inputTensors.add(inputTensor);
outputTensors.add(outputTensor);
model.runModel(inputTensors, outputTensors);
// The results of applying inference will be // stored in the Tensors of the list ‘outputTensors’ variable
```
   
      
   
## Examples

* [ExampleLoadAndRunModel](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleLoadAndRunModel.java) (PyTorch)
* [ExampleLoadTensorflow1Tensorflow2](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleLoadTensorflow1Tensorflow2.java)
* [ExampleDownloadEngine](https://github.com/bioimage-io/JDLL/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleDownloadEngine.java)
* [ExampleDownloadModel](https://github.com/bioimage-io/JDLL/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleDownloadModel.java)


## Acknowledgements.

This library was initially developed by the Icy team following the development of the DeepImageJ project. 
We acknowledge the financial support of France-Bioimaging.
We also acknowledge the AI for Life consortium that supported a hackathon in Milan (February 2023) during which we received feedback and advices from an excellent community.

## References
* If you used one of the material provided within JDLL, please consider citing their authors' work. 
* [C. García-López-de-Haro, S. Dallongeville, T. Musset, E. Gomez de Mariscal, D. Sage, W. Ouyang, A. Munoz-Barrutia, J. Tinevez, J. Olivo-Marin,
*JDLL: A library to run Deep Learning models on Java bioimage informatics platforms*, arXiv preprint arXiv:2306.04796 (2023).](https://arxiv.org/abs/2306.04796)

```bibtex
@article{de2023jdll,
  title={JDLL: A library to run Deep Learning models on Java bioimage informatics platforms},
  author={de Haro, Carlos Garcia Lopez and Dallongeville, Stephane and Musset, Thomas and de Mariscal, Estibaliz Gomez and Sage, Daniel and Ouyang, Wei and Munoz-Barrutia, Arrate and Tinevez, Jean-Yves and Olivo-Marin, Jean-Christophe},
  journal={arXiv preprint arXiv:2306.04796},
  year={2023}
}
```
