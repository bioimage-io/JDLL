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

The information about the engines supported currently by JDLL, for which OS and architectures and which JAR files are required for each of the engines is stored in [this json file](https://github.com/bioimage-io/model-runner-java/blob/main/src/main/resources/availableDLVersions.json) and can be found [here](https://github.com/bioimage-io/JDLL/wiki/List-of-supported-engines).

Note that JDLL will be in **constant development** and that it is open to community collaboration, so **pull requests** to the official repository of JDLL to improve functionality or to add new engines are **very welcomed**.


# Quickstart
This section will give the basic notions on how to use JDLL:
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
     <artifactId>dl-runner</artifactId>
     <version>0.3.12</version>
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
ouptutBlankTensor.close();
ouptutEmptyTensor.close();
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
