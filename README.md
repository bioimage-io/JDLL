[![Build Status](https://github.com/bioimage-io/model-runner-java/actions/workflows/build.yml/badge.svg)](https://github.com/bioimage-io/model-runner-java/actions/workflows/build.yml)

# JDLL (Java Deep Learning Library)

This project provides a Java library for running Deep Learning (DL) models agnostically, enabling communication between Java software and various Deep Learning frameworks (engines). It also allows the use of multiple DL frameworks in the same session, manages the different DL frameworks and brings the models from the 
[Bioimage.io](https://bioimage.io/#/) repository to Java.

JDLL was originally developed by the Icy team as the foundational component for the DeepIcy plugin; however, it evolved over time into a standalone library designed to facilitate the integration of Deep Learning methodologies into other software applications. Its first target was software developers but the Icy team realised that other potential end-users, such as Bioimage analysts with some Python background, could also benefit from it using JDLL for [scripting with Jython](https://imagej.net/scripting/jython/). Some example scripts can be found [here](https://github.com/bioimage-io/JDLL/tree/main/scripts). Also have a look at the section [*Quickstart for end-users*](https://github.com/bioimage-io/JDLL#quickstart-for-analystsscripting) to get more information about how to use JDLL for scripting.

[JDLL](<https://github.com/bioimage-io/model-runner-java/tree/main>) is able to load models and make inference, create tensors, download Bioimage.io models and manage the supported DL frameworks. The library is designed in a modular way, allowing the main software to avoid dealing with the various objects and structures required by different DL frameworks. Instead the Java model runner provides interfaces for models and tensors that handle internally their creation and inference in the differnet Java engines. The main software only needs to interact with the Java model runner and does not need to worry whether the model is in PyTorch, Tensorflow or other framework.

**FOR A MORE COMPREHENSIVE AND COMPLETE EXPLANATION OF JDLL, PLEASE VISIT THE [WIKI](https://github.com/bioimage-io/JDLL/wiki).**
## Table of Contents

1. [Supported engines](https://github.com/bioimage-io/JDLL#supported-engines)
2. [Quickstart](https://github.com/bioimage-io/JDLL#quickstart)
3. [Quickstart for developers](https://github.com/bioimage-io/JDLL#quickstart-for-developers)
4. [Quickstart for analysts/scripting](https://github.com/bioimage-io/JDLL#quickstart-for-analystsscripting)
5. [Examples for developers](https://github.com/bioimage-io/JDLL#examples)
6. [Scripting examples](https://github.com/bioimage-io/JDLL#scripting-examples)
7. [Acknowledgements](https://github.com/bioimage-io/JDLL#acknowledgements)
8. [References](https://github.com/bioimage-io/JDLL#references)

# Supported engines

Currently, the following frameworks are supported:

| Framework                       | Source code                                                    | Tag used in JDLL to refer to the framework     |
|---------------------------------|----------------------------------------------------------------|------------------------------------------------|
| PyTorch                         | https://github.com/bioimage-io/pytorch-java-interface          | `pytorch` or `torchscript`                      |
| Tensorflow 1                    | https://github.com/bioimage-io/tensorflow-1-java-interface     | `tensorflow` or `tensorflow_saved_model_bundle` |
| Tensorflow 2 API 0.2.0          | https://github.com/bioimage-io/tensorflow-2-java-interface-0.2.0 | `tensorflow` or `tensorflow_saved_model_bundle` |
| Tensorflow 2 API 0.3-0.4 | https://github.com/bioimage-io/tensorflow-2-java-interface     | `tensorflow` or `tensorflow_saved_model_bundle` |
| Tensorflow 2 API 0.5.0 | https://github.com/bioimage-io/tensorflow-2-java-interface-0.5.0 | `tensorflow` or `tensorflow_saved_model_bundle` |
| Onnx                            | https://github.com/bioimage-io/onnx-java-interface             | `onnx`                                         |

The information about the engines supported currently by JDLL, for which OS and architectures and which JAR files are required for each of the engines is stored in [this json file](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/resources/availableDLVersions.json) and can be found [here](https://github.com/bioimage-io/JDLL/wiki/List-of-supported-engines).

Note that JDLL will be in **constant development** and that it is open to community collaboration, so **pull requests** to the official repository of JDLL to improve functionality or to add new engines are **very welcomed**.

# Quickstart
As mentioned above, JDLL was originally developed for the purpose of seamlessly integrating Deep Learning capabilities into other software applications. Its main objective was to empower these software applications to effectively execute Deep Learning models. For that use case, JDLL was mainly oriented for application developers that wanted to incorporate DL methods in their softwares.

But after a couple of stable releases, the JDLL team noticed that because of its user-friendly interface, the library could be accessible not just to software developers but also to everyday users with basic scripting knowledge in languages like Python or Matlab. This makes it suitable for anyone interested in crafting processing routines that require Deep Learning models. Using a Java based or Java compatible scripting laguage such as Jython, one can easily use JDLL to run DL models.

Due to the dual possibilities of JDLL, there are 2 Quickstarts available depending on the interests of the user:
- [Quickstart for developers](https://github.com/bioimage-io/JDLL#quickstart-for-developers) if the user is interested in the integration of JDLL into their application.
- [Quickstart for analysts/scripting](https://github.com/bioimage-io/JDLL#quickstart-for-analystsscripting) if the user is interested in creating scripts in Jython that can run DL models easily to improve their processing routines. Not much experience with Python/Jython is required.



# Quickstart for developers
The next section (~10 min read) provides essential instructions for using JDLL divided in the following subsections:
- [0. Setting Up JDLL](https://github.com/bioimage-io/JDLL#0-setting-up-jdll)
- [1. Downloading a model (optional)](https://github.com/bioimage-io/JDLL#1-getting-a-model-optional)
- [2. Installing DL engines](https://github.com/bioimage-io/JDLL#2-installing-dl-engines)
- [3. Creating the tensors](https://github.com/bioimage-io/JDLL#3-creating-the-tensors)
- [4. Loading the model](https://github.com/bioimage-io/JDLL#4-loading-the-model)
- [5. Running the model](https://github.com/bioimage-io/JDLL#5-running-the-model)
- [6. Closing the model and the tensors](https://github.com/bioimage-io/JDLL#6-closing-the-model-and-the-tensors)

## 0. Setting Up JDLL

Download the dependency and include it in your project

   In order to benefit from the library, include the dependency in your code. The dependency can be added manually or using a dependency manager such as Maven. If you are using Maven, add the following dependency to the project pom file:

   ```xml
   <dependency>
     <groupId>io.bioimage</groupId>
     <artifactId>dl-modelrunner</artifactId>
     <version>0.4.0</version>
   </dependency>
   ```

   and add to `<repositories>` the following:

   ```xml
   <repository>
     <id>scijava.public</id>
     <url>https://maven.scijava.org/content/groups/public</url>
   </repository>
   ```
## 1. Downloading a model (optional)
If a model from the supported by JDLL is already available you can skip this step. Note that for Tensorflow the models need to be saved in the [`SavedModel`](https://www.tensorflow.org/guide/saved_model) format.

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
        System.out.println("Error downloading the model");
}
```
Output:
```
Great success!
```
More information on how to download  Bioimage.io models can be found [here](https://github.com/bioimage-io/JDLL/wiki/Engine-Installation-(EngineInstall)).



## 2. Installing DL engines
JDLL is installed empty. Several models might require different Deep Learning framework versions, each of them consuming considerable amounts of disk space. In order to make JDLL setup light and fast JDLL is installed without default DL engines. The user can then get the Dl engines that they want depending on their needs.

JDLL provides the needed methods to install the wanted engines in an easy manner. Following the above example, find below some code that can be used to install a DL engine. As it can be observed the model that was downloaded [supports Tensorflow 2 and Keras weights](https://github.com/bioimage-io/collection-bioimage-io/blob/19ea59e662410c3ee49b7da184730919336d7568/rdfs/10.5281/zenodo.7261974/7782776/rdf.yaml#L146). Keras is not supported so in order to load and run the model, Tensorflow weights need to be installed.

```
String framework = "tensorflow";
String version = "2.11.0";
boolean cpu = true;
boolean gpu = true;

String enginesDir = "/path/to/wanted/engines/dir";
boolean installed = EngineInstall.installEngineWithArgsInDir(framework, version, cpu, gpu, enginesDir);
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
boolean installed =  EngineInstall.installEnginesForModelByNameinDir(modelName, enginesDir)
if (installed)
	System.out.println("Great success!");
else
	System.out.println("Error installing");
```
Output:
```
Great success!
```
The Wiki covers extensively engine installation ([here](https://github.com/bioimage-io/JDLL/wiki/Understanding-engine-installation) and [here](https://github.com/bioimage-io/JDLL/wiki/Engine-Installation-(EngineInstall))). In addtion JDLL also includes methods to manage the engines and know: [the information about each engine](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-I-(DeepLearningVersion)), [which engines are supported](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-II-(AvailableEngines)) and [which engines have already been installed](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-III-(InstalledEngines))



## 3. Creating the tensors
Once the model and the engine are already installed it is the moment to start the process of running the model on the tensors. In this section, creation of the tensors will be explained.

JDLL tensors are agnostic to the DL framework to be used, they are always creted in the same way. JDLL manages internally the conversion of the agnostic tensor into the framework specific tensor once the model is going to be run. The unified method of creating tensors facilitates the integration of every supported DL framework into any software.

JDLL tensors use ImgLib2 to store the tensor information. In practice, JDLL tensors are just wrappers of ImgLib2 `RandomAccessibleIntervals` that contain all the data needed to convert them back and forth into the framework specific tensors.

The example below will show how to create the input and output tensors required to run the [example model](https://bioimage.io/#/?tags=placid-llama&id=10.5281%2Fzenodo.7261974). As per its [rdf.yaml file](https://github.com/bioimage-io/collection-bioimage-io/blob/19ea59e662410c3ee49b7da184730919336d7568/rdfs/10.5281/zenodo.7261974/7782776/rdf.yaml), the model has one input named `input_1`, with `bxyc` axes ([explanation here](https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md)) and a required shape of[1, 512, 512, 1]. The ouptut of the model is named `conv2d_19` with with `bxyc` axes and fixed shape [1, 512, 512, 3].
```java
final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
final Img< FloatType > img1 = imgFactory.create( new long[] {1, 512, 512, 1} );
// Create the input tensor with the nameand axes given by the rdf.yaml file
// and add it to the list of input tensors
Tensor<FloatType> inpTensor = Tensor.build("input_1", "bxyc", img1);

// Ouput tensors can be created empty, if the output shape is not known.
// Note that this method does not preallocate memory for the output tensor
Tensor<T> outputEmptyTensor = Tensor.buildEmptyTensor("conv2d_19", "bxyc");

// Or ouptut tensors can also be built blank, to pre-allocate memory
// if the shape and data type are known.
Tensor<FloatType> outputBlankTensor = Tensor.buildBlankTensor("conv2d_19",
			"bxyc", new long[] {1, 512, 512, 3}, new FloatType());

```

More information about tensors can be found in the [JDLL wiki](https://github.com/bioimage-io/JDLL/wiki/JDLL-tensors-I-(Tensor)).


## 4. Loading the model
Before making inference with a model, it needs to be loaded. Similar to [tensor creation](https://github.com/bioimage-io/JDLL#3-creating-the-tensors) JDLL provides an unified way to load models from any DL framework.

Loading a model implies first defining which engine is going to be used for the model and then loading the model. The engine is defined as an instance of the class `io.bioimage.modelrunner.engine.EngineInfo`. The engine used to load a model can be either the exact same version wanted, or a [compatible](https://github.com/bioimage-io/JDLL/wiki/Potential-errors#compatible-versions) one. Using the exact same version guarantees that the model loading and inference are going to be smooth but implies that many more engines will have to be installed.

An example of defining the `EngineInfo` instance needed to load a model is shown below. The engine required in this code is required to be the exact engine wanted.

Note that `String enginesDir` is the directory where the wanted engines have been installed. [Click here and look at the example in the redirected section](https://github.com/bioimage-io/JDLL#2-installing-dl-engines).
```java
String framework = "tensorflow";
String version = "2.11.0";
boolean cpu = true;
boolean gpu = true;
String enginesDir = "/path/to/wanted/engines/dir";

EngineInfo enigneInfoExact = EngineInfo.defineDLEngine(framework, version, cpu, gpu, enginesDir);
```

In order to require a compatible engine, not the exact one:

```java
String framework = "tensorflow";
String version = "2.11.0";
boolean cpu = true;
boolean gpu = true;
String enginesDir = "/path/to/wanted/engines/dir";

EngineInfo enigneInfoCompat = EngineInfo.defineCompatibleDLEngine(framework, version, cpu, gpu, enginesDir);
```

The developer acknowledges that the class `io.bioimage.modelrunner.engine.EngineInfo` can be difficult to understand. This is why the [wiki contains a detailed sections trying to explain it in an understandable manner with several examples](https://github.com/bioimage-io/JDLL/wiki/Load-and-run-models-I-(EngineInfo)).

Once the `EngineInfo` instance has been created, loading the model is easy. The only parameters needed now are the path to the `model folder` and the path to the `source file`.
The model folder is the folder that contains the `.pb` file in Tensorflow, the `.pt` file in Pytorch and the `.onnx` file in Onnx. The source file is not needed for Tensorflow, is the path to the `.pt` file in Pytorch and the path to the `.onnx` file in Onnx.

Then with the arguments `String modelFolder`, `String modelSource` and the previously created `EngineInfo enigneInfoExact` or `EngineInfo enigneInfoCompat` an instance of `io.bioimage.modelrunner.model.Model` can be created. Tha object then can be loaded and run inference.


```
//Path to the example model folder
String modelFolder = "path/to/models/dir/B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet";
String modelSource = null; // Not needed in Tensorflow

Model model = Model.create(modelFolder, modelSource, enigneInfoCompat);
model.loadModel();

System.out.println("Great sucess!");
```

Output:
```
Great sucess!
```

JDLL tight integration with Bioimage.io models makes loading them easier. To load a Bioimage.io model it is not necessary to create the `EngineInfo` object.

For Bioimage.io models, loading is reduced to the next example:
```
// Path to the Bioimage.io model folder
String modelFolder = "path/to/models/dir/B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet";
String enginesDir = "/path/to/wanted/engines/dir";

Model bioimageioModel = Model.createBioimageioModel(modelFolder, enginesDir);
bioimageioModel.load();

System.out.println("Great sucess!");
```

Output:
```
Great sucess!
```

More information about loading models and models in general can be found in [this Wiki page](https://github.com/bioimage-io/JDLL/wiki/Load-and-run-models-II-(Model)).



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
outputBlankTensor.close();
outputEmptyTensor.close();
```








































# Quickstart for analysts/scripting
This section (~15 min read) explains how to use JDLL in a [Jython](https://www.jython.org/) script. Jython is an implementation of Python in Java, thus it allows calling Java methods and classes in a Pythonic way. With Jython, Java methods and classes can be called in a script as Python methods. 

Scripting is powerfull and usefull because it allows creating processing routines in a simple manner that otherwise would require a full software for them. This is the reason why many software applications such as Icy, Fiji or Napari have a script editor. Scripts can perform complex tasks reducing the need of developing specific plugins in each platform. Scripts are also mostly compatible with every software application that supports the language they are writen in.

The examples for JDLL use Jython because it supports Java and uses Python syntax. Python is one of the mostwidely used programming languages and also one of the easiers to learn. Python has a big community and many Open Source libraries that can be used in Jython too. However, note that Jython at the moment only supports Python2. The support for [Python 3 is still in development](https://github.com/jython/jython/tree/main). A good tutorial on how to create Jython scripts can be found [here](https://imagej.net/scripting/jython/).

In addition, Jython scripts can be then used in softwares such as Icy or Fiji to create Image Processing tasks together with the tools they have available.

The following subsections provide essential instructions to use JDLL in Jyhton scripts:
- [0. Setting Up JDLL](https://github.com/bioimage-io/JDLL#0-setting-up-jdll-1)
- [1. Downloading a model (optional)](https://github.com/bioimage-io/JDLL#1-getting-a-model-optional-1)
- [2. Installing DL engines](https://github.com/bioimage-io/JDLL#2-installing-dl-engines-1)
- [3. Creating the tensors](https://github.com/bioimage-io/JDLL#3-creating-the-tensors-1)
- [4. Loading the model](https://github.com/bioimage-io/JDLL#4-loading-the-model-1)
- [5. Running the model](https://github.com/bioimage-io/JDLL#5-running-the-model-1)
- [6. Closing the model and the tensors](https://github.com/bioimage-io/JDLL#6-closing-the-model-and-the-tensors-1)

Note that the [JDLL Wiki](https://github.com/bioimage-io/JDLL/wiki) contains an extensive documentation of the JDLL API. The API methods can be used with Jython. Consider going over the API to get a more in-depth idea of the scripting possibilities of JDLL.

## 0. Setting up JDLL

Download the dependency and include it in your project. Depending on the application, dependencies are installed differently. For example, in Icy, click on the `Plugins` tab, then `Setup>Online plugin`. In the window that opens look for the name `JDLL` within the plugin list, click on it and then click on `Install` on the middle right of the window.

In Fiji, the JDLL library and its dependencies are shipped from the deepImageJ update site. In order to install JDLL in Fiji, [add the deepImageJ update site to the Fiji updater](https://imagej.net/list-of-update-sites/) and update. The URL for the deepImageJ update site is: https://sites.imagej.net/DeepImageJ/

Once installed, JDLL should be ready to be used. 

If the software application JDLL wants to be used in does not support automatic installation of libraries, download JDLL and its dependencies and locate them in the corresponding directory.
The download links are:
- [JDLL](https://maven.scijava.org/service/local/repositories/releases/content/io/bioimage/dl-modelrunner/0.4.0/dl-modelrunner-0.4.0.jar)
- [GSON](https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar)
- [ImgLib2](https://repo1.maven.org/maven2/net/imglib2/imglib2/6.1.0/imglib2-6.1.0.jar)
- [SnakeYAML](https://repo1.maven.org/maven2/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar)
- [jackson-dataformat-msgpack](https://repo1.maven.org/maven2/org/msgpack/jackson-dataformat-msgpack/0.9.5/jackson-dataformat-msgpack-0.9.5.jar)
- [msgpack-core](https://repo1.maven.org/maven2/org/msgpack/msgpack-core/0.9.5/msgpack-core-0.9.5.jar)
- [jackson-databind](https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.14.2/jackson-databind-2.14.2.jar)
- [jackson-core](https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.14.2/jackson-core-2.14.2.jar)
- [jackson-annotations](https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.14.2/jackson-annotations-2.14.2.jar)

   
## 1. Downloading a model (optional)
If a model from the supported by JDLL is already available you can skip this step. Note that for Tensorflow the models need to be saved in the [`SavedModel`](https://www.tensorflow.org/guide/saved_model) format.

If no model is available, a good starting point is downloading any of the models of the [Bioimage.io respository](https://bioimage.io/#/). The download can be done manually or using JDLL. Bioimag.io is seamlessly integrated into JDLL, offering multiple methods to effortlessly mange and use its models. An example Jython script to download a Bioimage.io model can be found [here](https://github.com/bioimage-io/JDLL/blob/main/scripts/example-download-bmz-model.py).

Here is an emaple of how JDLL can be used to download any Bioimage.io model, in this case the [`B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet`](https://bioimage.io/#/?tags=placid-llama&id=10.5281%2Fzenodo.7261974) model.
```python
from io.bioimage.modelrunner.bioimageio import BioimageioRepo

# Name of the model of interest, note that each model is unique. The names are case sensitive.
modelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"
# Directory where the model folder will be downloaded
modelsDir = "/path/to/wanted/model/directory"

# First create an instance of the Bioimage.io repository
br = BioimageioRepo.connect()
try:
	br.downloadByName(modelName, modelsDir)
	print("Great success!")
except Exception as e:
	# If the download is interrumpted or any of the model files cannot be downloaded
	# and exception will be thrown
	print("Error downloading the model: ", str(e))
```
Output:
```
Great success!
```
More information on how to download  Bioimage.io models can be found [here](https://github.com/bioimage-io/JDLL/wiki/Engine-Installation-(EngineInstall)). Note that all the methods decribed in JAva can be used in a pythonic way with Jytho. The static ones do not need the instantiation of a class and the non-static ones do require it.



## 2. Installing DL engines
JDLL is installed empty. Several models might require different Deep Learning framework versions, each of them consuming considerable amounts of disk space. In order to make JDLL setup light and fast JDLL is installed without default DL engines. The user can then get the Dl engines that they want depending on their needs.

JDLL provides the needed methods to install the wanted engines in an easy manner. Following the above example, find below some code that can be used to install a DL engine. As it can be observed the model that was downloaded [supports Tensorflow 2 and Keras weights](https://github.com/bioimage-io/collection-bioimage-io/blob/19ea59e662410c3ee49b7da184730919336d7568/rdfs/10.5281/zenodo.7261974/7782776/rdf.yaml#L146). Keras is not supported so in order to load and run the model, Tensorflow weights need to be installed.

```python
from io.bioimage.modelrunner.engine.installation import EngineInstall

framework = "tensorflow"
version = "2.11.0"
cpu = True
gpu = True

enginesDir = "/path/to/wanted/engines/dir"
installed = EngineInstall.installEngineWithArgsInDir(framework, version, cpu, gpu, enginesDir)
if (installed):
	print("Great success!")
else:
	print("Error installing")
```
Output:
```
Great success!
```

As previously mentioned, JDLL integrates deeply with Bioimage.io models. An easier way to install the engines needed for Bioimage.io models is shown in the code below.

In the example it is shown how simply providing the name of the model of interest, JDLL will know which engines to install.
```python
from io.bioimage.modelrunner.engine.installation import EngineInstall

modelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"
modelsDir = "/path/to/wanted/models/dir"
installed =  EngineInstall.installEnginesForModelByNameinDir(modelName, modelsDir)
if (installed):
	print("Great success!")
else:
	print("Error installing")
```
Output:
```
Great success!
```
The Wiki covers extensively engine installation ([here](https://github.com/bioimage-io/JDLL/wiki/Understanding-engine-installation) and [here](https://github.com/bioimage-io/JDLL/wiki/Engine-Installation-(EngineInstall))). In addtion JDLL also includes methods to manage the engines and know: [the information about each engine](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-I-(DeepLearningVersion)), [which engines are supported](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-II-(AvailableEngines)) and [which engines have already been installed](https://github.com/bioimage-io/JDLL/wiki/Engine-Management-III-(InstalledEngines)) Again, not that the methods explained in the Wiki can easily be used in Jython scripts.

In addition, a more detailed Jython script with an example on how to install an engine can be found [here](https://github.com/bioimage-io/JDLL/blob/main/scripts/example-download-engine.py).



## 3. Creating the tensors
Once the model and the engine are already installed it is the moment to start the process of running the model on the tensors. In this section, creation of the tensors will be explained.

JDLL tensors are agnostic to the DL framework to be used, they are always creted in the same way. JDLL manages internally the conversion of the agnostic tensor into the framework specific tensor once the model is going to be run. The unified method of creating tensors facilitates the integration of every supported DL framework into any software.

JDLL tensors use ImgLib2 to store the tensor information. In practice, JDLL tensors are just wrappers of ImgLib2 `RandomAccessibleIntervals` that contain all the data needed to convert them back and forth into the framework specific tensors.

The example below will show how to create the input and output tensors required to run the [example model](https://bioimage.io/#/?tags=placid-llama&id=10.5281%2Fzenodo.7261974). As per its [rdf.yaml file](https://github.com/bioimage-io/collection-bioimage-io/blob/19ea59e662410c3ee49b7da184730919336d7568/rdfs/10.5281/zenodo.7261974/7782776/rdf.yaml), the model has one input named `input_1`, with `bxyc` axes ([explanation here](https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md)) and a required shape of[1, 512, 512, 1]. The ouptut of the model is named `conv2d_19` with with `bxyc` axes and fixed shape [1, 512, 512, 3].
```python
from io.bioimage.modelrunner.tensor import Tensor

from net.imglib2.img.array import ArrayImgFactory
from net.imglib2.type.numeric.real import FloatType

imgFactory = ArrayImgFactory( FloatType() )
img1 = imgFactory.create( [1, 512, 512, 1] )
# Create the input tensor with the nameand axes given by the rdf.yaml file
# and add it to the list of input tensors
inpTensor = Tensor.build("input_1", "bxyc", img1)

# Ouput tensors can be created empty, if the output shape is not known.
# Note that this method does not preallocate memory for the output tensor
outputEmptyTensor = Tensor.buildEmptyTensor("conv2d_19", "bxyc")

# Or ouptut tensors can also be built blank, to pre-allocate memory
# if the shape and data type are known.
outputBlankTensor = Tensor.buildBlankTensor("conv2d_19",
			"bxyc", [1, 512, 512, 3], FloatType())

```

More information about tensors can be found in the [JDLL wiki](https://github.com/bioimage-io/JDLL/wiki/JDLL-tensors-I-(Tensor)).

Another option is to create the tensors directly from images are open on the software. The next example shows how an `ImagePlus` opened by Fiji can be converted into a tensor.
```python
import os
from io.bioimage.modelrunner.tensor import Tensor
from ij import IJ
from net.imglib2.img.display.imagej import ImageJFunctions
from net.imglib2.view import Views

# Path to the model whose sample image is going to be displayed
model_name = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"
models_dir = "/path/to/wanted/models/dir"
model_of_interest_path = os.path.join(models_dir, model_name)

#Open the image and show it
imp = IJ.openImage(os.path.join(model_of_interest_path, "sample_input_0.tif"))
imp.show()

# Convert the image into a float32 ImgLib2 image.
# Note that as a 2D image the dimensions are just "xy"
wrapImg = ImageJFunctions.convertFloat(imp)
# Convert to the required axes order "bxyc"
# Permute from "xy" to "yx"
wrapImg = Views.permute(wrapImg, 0, 1)
# Add one dimension to "yxb", from (512, 512) to (512, 512, 1)
wrapImg = Views.addDimension(wrapImg, 0, 0
# Permute from "yxb" and (512, 512, 1) to "bxy" and (1, 512, 512)
wrapImg = Views.permute(wrapImg, 0, 2)
# Add one dimension to get "bxyc", from (1, 512, 512) to (1, 512, 512, 1)
wrapImg = Views.addDimension(wrapImg, 0, 0)

# Build the corresponding tensor
inputTensor = Tensor.build("input_1", "bxyc", wrapImg)
```

In order to get the orginal image back from the tensor after all the permutations and dimensions added:
```python
from net.imglib2.img.display.imagej import ImageJFunctions
from net.imglib2.view import Views

# Convert from "bxyc" (1, 512, 512, 1) into "xy" (512, 512)
wrapImg = Views.dropSingletonDimensions(wrapImg)
ImageJFunctions.show(wrapImg)
```

A complete example that contains tensor creation from Fiji ImagePlus images can be found [here](https://github.com/bioimage-io/JDLL/blob/c037809a3695632487cc024bcefe93b529cc89fe/scripts/example-run-model-in-fiji.py#L67).


## 4. Loading the model
Before making inference with a model, it needs to be loaded. Similar to [tensor creation](https://github.com/bioimage-io/JDLL#3-creating-the-tensors) JDLL provides an unified way to load models from any DL framework.

Loading a model implies first defining which engine is going to be used for the model and then loading the model. The engine is defined as an instance of the class `io.bioimage.modelrunner.engine.EngineInfo`. The engine used to load a model can be either the exact same version wanted, or a [compatible](https://github.com/bioimage-io/JDLL/wiki/Potential-errors#compatible-versions) one. Using the exact same version guarantees that the model loading and inference are going to be smooth but implies that many more engines will have to be installed.

An example of defining the `EngineInfo` instance needed to load a model is shown below. The engine required in this code is required to be the exact engine wanted.

Note that `enginesDir` is the directory where the wanted engines have been installed. [Click here and look at the example in the redirected section](https://github.com/bioimage-io/JDLL#2-installing-dl-engines).
```python
from io.bioimage.modelrunner.engine import EngineInfo

framework = "tensorflow"
version = "2.11.0"
cpu = True
gpu = True
enginesDir = "/path/to/wanted/engines/dir"

enigneInfoExact = EngineInfo.defineDLEngine(framework, version, cpu, gpu, enginesDir)
```

In order to require a compatible engine, not the exact one:

```python
from io.bioimage.modelrunner.engine import EngineInfo

framework = "tensorflow"
version = "2.11.0"
cpu = True
gpu = True
enginesDir = "/path/to/wanted/engines/dir"

enigneInfoCompat = EngineInfo.defineCompatibleDLEngine(framework, version, cpu, gpu, enginesDir)
```

The developer acknowledges that the class `io.bioimage.modelrunner.engine.EngineInfo` can be difficult to understand. This is why the [wiki contains a detailed sections trying to explain it in an understandable manner with several examples](https://github.com/bioimage-io/JDLL/wiki/Load-and-run-models-I-(EngineInfo)).

Once the `EngineInfo` instance has been created, loading the model is easy. The only parameters needed now are the path to the `model folder` and the path to the `source file`.
The model folder is the folder that contains the `.pb` file in Tensorflow, the `.pt` file in Pytorch and the `.onnx` file in Onnx. The source file is not needed for Tensorflow, is the path to the `.pt` file in Pytorch and the path to the `.onnx` file in Onnx.

Then with the arguments `modelFolder`, `modelSource` and the previously created `enigneInfoExact` or `enigneInfoCompat` an instance of `io.bioimage.modelrunner.model.Model` can be created. Tha object then can be loaded and run inference.


```python
from io.bioimage.modelrunner.model import Model

# Path to the example model folder
modelFolder = "path/to/models/dir/B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"
modelSource = None # Not needed in Tensorflow

model = Model.create(modelFolder, modelSource, enigneInfoCompat)
model.loadModel()

print("Great sucess!")
```

Output:
```
Great sucess!
```

JDLL tight integration with Bioimage.io models makes loading them easier. To load a Bioimage.io model it is not necessary to create the `EngineInfo` object.

For Bioimage.io models, loading is reduced to the next example:
```python
from io.bioimage.modelrunner.model import Model

# Path to the Bioimage.io model folder
modelFolder = "path/to/models/dir/B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"
enginesDir = "/path/to/wanted/engines/dir"

bioimageioModel = Model.createBioimageioModel(modelFolder, enginesDir)
bioimageioModel.load()

print("Great sucess!")
```

Output:
```
Great sucess!
```

More information about loading models and models in general can be found in [this Wiki page](https://github.com/bioimage-io/JDLL/wiki/Load-and-run-models-II-(Model)).

In addition, a complete example showing how to load a Bioimage.io model can be found [here](https://github.com/bioimage-io/JDLL/blob/c037809a3695632487cc024bcefe93b529cc89fe/scripts/example-run-model-in-fiji.py#L87).



## 5. Running the model
Once the model has been loaded and the input and output tensors have been created. Running the model is simple. The input tensors should be added to a `List<?>` in the same order the model expects. Same for the ouptuts in another `List`.
```python
inputList = []
outputputList = []

inputList.append(inputTensor)
outputputList.append(outputEmptyTensor)
print("Ouptut tensor is empty: ",  outputEmptyTensor.isEmpty())

model.runModel(inputList, outputputList)
print("Ouptut tensor after inference is empty: ", outputEmptyTensor.isEmpty())
```
Output:
```
Ouptut tensor is empty: true
Ouptut tensor after inference is empty: false
```


## 6. Closing the model and the tensors
Models and tensors need to be closed to be released and free the memory that they were using
```python
model.close()
inputTensor.close()
outputBlankTensor.close()
outputEmptyTensor.close()
```
      
   
# Examples

* [ExampleLoadAndRunModel](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleLoadAndRunModel.java) (PyTorch)
* [ExampleLoadTensorflow1Tensorflow2](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleLoadTensorflow1Tensorflow2.java)
* [ExampleDownloadEngine](https://github.com/bioimage-io/JDLL/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleDownloadEngine.java)
* [ExampleDownloadModel](https://github.com/bioimage-io/JDLL/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleDownloadModel.java)
   
# Scripting examples

* [example-run-model-in-fiji](https://github.com/bioimage-io/JDLL/blob/main/scripts/example-run-model-in-fiji.py) (Requires to be executed from Fiji, as it uses the application to display the result image)
* [example-download-bmz-model](https://github.com/bioimage-io/JDLL/blob/main/scripts/example-download-bmz-model.py)
* [example-download-engine](https://github.com/bioimage-io/JDLL/blob/main/scripts/example-download-engine.py)


# Acknowledgements.

This library was initially developed by the Icy team following the development of the DeepImageJ project. 
We acknowledge the financial support of France-Bioimaging.
We also acknowledge the AI for Life consortium that supported a hackathon in Milan (February 2023) during which we received feedback and advices from an excellent community.

# References
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
