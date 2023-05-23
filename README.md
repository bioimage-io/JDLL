[![Build Status](https://github.com/bioimage-io/model-runner-java/actions/workflows/build.yml/badge.svg)](https://github.com/bioimage-io/model-runner-java/actions/workflows/build.yml)

# JDLL (Java Deep Learning Library)

This project provides a Java library for running Deep Learning (DL) models agnostically, enabling communication between Java software and various Deep Learning frameworks (engines). It also allows the use of multiple DL frameworks in the same session, manages the different DL frameworks and brings the models from the 
[Bioimage.io](https://bioimage.io/#/) repository to Java.

This library is intended for developers and was originally built by the DeepIcy team as the backend of the DeepIcy plugin.

[JDLL](<https://github.com/bioimage-io/model-runner-java/tree/main>) is able to load models and make inference, create tensors, download Bioiamge.io models and manage the supported DL frameworks. The library is designed in a modular way, allowing the main software to avoid dealing with the various objects and structures required by different DL frameworks. Instead the Java model runner provides interfaces for models and tensors that handle internally their creation and inference in the differnet Java engines. The main software only needs to interact with the Java model runner and does not need to worry whether the model is in PyTorch, Tensorflow or other framework.

**FOR A MORE COMPREHENSIVE AND COMPLETE EXPLANATION OF JDLL, PLEASE VISIT THE [WIKI](https://github.com/bioimage-io/JDLL/wiki).**

## Setting Up the Model Runner

1. Download the dependency and include it in your project

   In order to benefit from the library, include the dependency in your code. The dependency can be added manually or using a dependency manager such as Maven. If you are using Maven, add the following dependency to the project pom file:

   ```xml
   <dependency>
     <groupId>io.bioimage</groupId>
     <artifactId>dl-modelrunner</artifactId>
     <version>0.3.6</version>
   </dependency>
   ```

   and add to `<repositories>` the following:

   ```xml
   <repository>
     <id>scijava.public</id>
     <url>https://maven.scijava.org/content/groups/public</url>
   </repository>
   ```

## Manage the DL engines

   Certain pairs of DL frameworks cannot be loaded in the same classloader due to incompatible classes with the same names. For example, the Java APIs of Tensorflow 1 and Tensorflow 2 are incompatible, which has slowed the adoption of newer versions of Tensorflow in Java softwares, disrupting the connection with the latest deep learning developments.

   To address this issue, the library is designed in a modular way that creates a separate classloader for each DL framework once it is called, avoiding conflicts between the frameworks.

   To load frameworks in separate classloaders, the library requires that the executable JAR files be stored in identifiable folders, with all DL frameworks stored in the same directory. An example of this is shown in the images below:

   ![alt text](https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/wiki/engines_folders.png)

   All engines should be stored in the same directory (in the example, `C:\Users\carlos\icy\engines`), with each engine following the naming convention:

   ```
   <DL_framework_name>-<python_version>-<java_api_version>-<os>-<architecture>-<cpu_if_it_runs_in_cpu>-<gpu_if_it_runs_in_gpu>
   ```

   For example, the folder `Pytorch-1.11.0-1.11.0-windows-x86_64-cpu-gpu` contains a PyTorch engine, Python version 1.11.0, same as the Java version, for Windows operating system, architecture x86_64 (64 bits), that runs in CPU and GPU.

   Another example: the folder `Tensorflow-1.12.0-1.12.0-windows-x86_64-cpu` contains a Tensorflow engine, Python version 1.12.0, same as the Java version, for Windows operating system, architecture x86_64 (64 bits), that runs only in CPU.

   The script `download-engines.py` can be used to download desired engines for your platform (OS and architecture):
   ```
   python download-engines.py <engine-filter> <version-filter>
   ```

   For example:
   ```
   python download-engines.py pytorch 1.13.1
   ```

   Running the script with no arguments downloads all available engines for your platform.
   
   ### Manage the engines with JDLL
   
   The installation of the engines for JDLL to work can be done manually. before starting the software. But to enhance usability JDLL provides the methods to install the required engines at runtime. In addition, once an engine is installed it can be used directly, without needing to re-start the JVM.
   
   There are several ways to install an engine with code:
   * If we know the different parameters of the engine we want to install we can call:
      ```
      String framework = "tensorflow";
      String version = "2.7.0";
      boolean cpu = true;
      boolean gpu = true;
      String dir = "/path/to/wanted/engines/dir";
      EngineManagement.installEngineWithArgsInDir(framework, version, cpu, gpu, dir);
      ```
      
   * If we have a certain [Bioimage.io](https://bioimage.io/#/) model and we want to install the engines compatible with it:
      ```
      String modelName = "Neuron Segmentation in EM (Membrane Prediction)";
      String dir = "/path/to/wanted/engines/dir";
      EngineManagement.installEnginesForModelByNameinDir(modelName, dir);
      ```
      we can also use the model ID:
      ```
      String modelID = "10.5281/zenodo.5874741/5874742";
      String dir = "/path/to/wanted/engines/dir";
      EngineManagement.installEnginesForModelByNameinDir(modelID, dir);
      ```
      
      or directly provide the model folder and install the engines required for that model. Note that this only works for [Bioimage.io](https://bioimage.io/#/) models that contain the [rdf.yaml specs file](https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md) in the folder:
      ```
      String modelFolder = "/path/to/model/folder";
      String dir = "/path/to/wanted/engines/dir";
      EngineManagement.installEnginesinDirForModelInFolder(modelFolder, dir);
      ```
      
      In the explained examples, the engines installed will be specified by the weights in the rdf.yaml files of the selected models. However regard that only the [engines supported by JDLL](https://github.com/bioimage-io/JDLL/blob/main/README.md#supported-engines) will be installed. These are torchscript (pytorch), tensorflow_saved_model_bundled (tensorflow) and onnx.
      
      For more information, please read the wiki page about [engine management and installation](TODO add link)

### Supported engines

Currently, the following `engine_name-interface` engines exist:

| Engine                          | Source code                                                    |
|---------------------------------|----------------------------------------------------------------|
| PyTorch                         | https://github.com/bioimage-io/pytorch-java-interface          |
| Tensorflow 1                    | https://github.com/bioimage-io/tensorflow-1-java-interface     |
| Tensorflow 2 API 0.2.0          | https://github.com/bioimage-io/tensorflow-2-java-interface-0.2 |
| Tensorflow 2 all APIs but 0.2.0 | https://github.com/bioimage-io/tensorflow-2-java-interface     |
| Onnx                            | https://github.com/bioimage-io/onnx-java-interface             |

The information about the engines supported currently by the model runner, for which OS and architectures and which JAR files are required for each of the engines is stored in [this json file](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/resources/availableDLVersions.json).

Note that the model runner will be in **constant development** and that it is open to community collaboration, so **pull requests** to the official repository of the model runner to improve functionality or to add new engines are **very welcomed**.

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

## Downloading and running Bioimage.io models

   JDLL also facilitates the use of [Bioimage.io](https://bioimage.io/#/) models. It integrates methods to download the models and to load them directly using the information from the rdf.yaml file.
   
   ### 1. Download
   JDLL can connect to the web repository of the Bioimage.io and retreive information about every model that exists there.
   
      // Create an instance of the BioimageRepo object
		BioimageioRepo br = BioimageioRepo.connect();
      boolean verbose = false;
      // Retrieve a map where the key corresponds to the online URL to the rdf.yaml 
      // specs file of a model and the value corresponds to the information contained in the file
		Map<Path, ModelDescriptor> models = br.listAllModels(verbose);
   
   The rdf.yaml file contains some cualitative data such as a short descrption of the model, its name, the model ID or a couple of references and citations, but also contains technical information that enables loading the model.
   
   Once a model has been selected, it can be downloaded by its name with:
   
      String name = "Neuron Segmentation in EM (Membrane Prediction)";
      String modelsDirectory = "/path/to/models/dir";
      br.downloadByName(name, modelsDirectory);
   
   It also can be downloaded by its ID:
   
      String modelID = "10.5281/zenodo.5874741/5874742";
      String modelsDirectory = "/path/to/models/dir";
      br.downloadModelByID(modelID, modelsDirectory);
      
   By the URL of its rdf.yaml specs file (the key of the `models` map:
   
      String rdfSource = "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5874741/5874742/rdf.yaml";
      String modelsDirectory = "/path/to/models/dir";
      br.downloadByRdfSource(rdfSource, modelsDirectory);
      
   Or by the JDLL Java object `ModelDescriptor`, which contains the info of the model. This object is the value in the `models` map:
   
      String rdfSource = "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5874741/5874742/rdf.yaml";
      ModelDescriptor descriptor = models.get(rdfSource);
      String modelsDirectory = "/path/to/models/dir";
      br.downloadModel(descriptor, modelsDirectory);
      
   ### 2. Load and run Bioimage.io models
   
   JDLL facilitates the use of Bioimage.io models making easier to use them. This is possible due to the [rdf.yaml specs file](https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md), as it contains all the technical info needed to load a model.
   
   Loading a Bioimage.io model is very easy.
   
   ```
   String modelPath = "/path/to/model";
   // Note that this is not the path to the actual engine that we want to use, but the
   // path to the directory where all the engine folders are located.
   // Using the example from section [Manage DL engines](https://github.com/bioimage-io/JDLL/edit/main/README.md#manage-the-dl-engines)
   // enginesDir would be C:\Users\carlos\icy\engines
   String enginesDir = "/path/to/engines";
   Model bioiamgeioModel = Model.createBioimageioModel(modelPath, enginesDir);
   ```
   
   Regard that `Model.createBioimageioModel(modelPath, enginesDir)` loads the model only if there is a compatible engine installed in the computer.
   Compatible means that the Deep Learning framework has to be the same and that the major version is the same (for example Pytorch 1.13.1 and 1.11.0 are compatible but Tensorflow 1.15.0 and Tensorflow 2.7.0 are NOT compatible).
   
   I not compatible engine is found, an exception will be thrown asking to install a compatible engine.
   
   In order to find if a compatible engine exists:
   
   ```
   String enginesDir = "/path/to/engines/dir";
   String engine = "tensorflow";
   String version = "2.7.0";
   InstalledEngines manager = InstalledEngines.buildEnginesFinder(enginesDir);
   String compatibleVersion = manager.getMostCompatibleVersionForEngine(engine, version);
   ```
   
   where `compatibleVersion` will be the most compatible version installed of that engine. If the same engine wanted is installed, `version`and `compatibleVersion` will be the same. However, if no comptible version is found, `compatibleVersion` will be `null`.
   
   If we want to load the model using the exact versions of the engines specified in the rdf.yaml file:
   
   ```
   String modelPath = "/path/to/model";
   // Note that this is not the path to the actual engine that we want to use, but the
   // path to the directory where all the engine folders are located.
   // Using the example from section [Manage DL engines](https://github.com/bioimage-io/JDLL/edit/main/README.md#manage-the-dl-engines)
   // enginesDir would be C:\Users\carlos\icy\engines
   String enginesDir = "/path/to/engines";
   Model bioiamgeioModel = Model.createBioimageioModelWithExactWeigths(modelPath, enginesDir);
   ```
   
   If the exact engine version is not installed, the method will throw an exception asking to install it.
   
   Once the model is loaded we can continue the steps explained above [here](https://github.com/bioimage-io/JDLL#2-creating-agnostic-tensors) and [here](https://github.com/bioimage-io/JDLL#3-making-inference).

## Examples

* [ExampleLoadAndRunModel](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleLoadAndRunModel.java) (PyTorch)
* [ExampleLoadTensorflow1Tensorflow2](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleLoadTensorflow1Tensorflow2.java)
* [ExampleDownloadEngine](https://github.com/bioimage-io/JDLL/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleDownloadEngine.java)
* [ExampleDownloadModel](https://github.com/bioimage-io/JDLL/blob/main/src/main/java/io/bioimage/modelrunner/example/ExampleDownloadModel.java)


## Acknowledgements.

This library was initially developed by the Icy team following the development of the DeepImageJ project. 
We acknowledge the financial support of France-Bioimaging.
We also acknowledge the AI for Life consortium that supported a hackathon in Milan (February 2023) during which we received feedback and advices from an excellent community.
