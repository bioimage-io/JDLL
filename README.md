# Java Library for Deep Learning

This project provides a Java library for running Deep Learning (DL) models agnostically, enabling communication between Java software and various Deep Learning frameworks. It also allows the use of multiple DL frameworks in the same session. 

This library is intended for developers and was originally built by the DeepIcy team as the backend of the DeepIcy plugin.

The [Java model runner](<https://github.com/bioimage-io/model-runner-java/tree/main>) is able to load models, create tensors and make inference. The library is designed in a modular way, allowing the main software to avoid dealing with the various objects and structures required by different DL frameworks. Instead the Java model runner provides interfaces for models and tensors that handle internally their creation and inference in the differnet Java DL framworks. The main software only needs to interact with the Java model runner and does not need to worry whether the model is in Pytorch or in Tensorflow

## Setting Up the Model Runner

1. Download the dependency and include it in your project

   In order to benefit from the library, include the dependency in your code. The dependency can be added manually or using a dependency manager such as Maven. If you are using Maven add the following dependeny to the project pom file:
   
   ```
   <dependency>
     <groupId>org.bioimageanalysis.icy</groupId>
     <artifactId>dl-model-runner</artifactId>
     <version>1.0.0</version>
   </dependency>
   ```
2. Prepare the environment

   Certain pairs of DL frameworks cannot be loaded in the same classloader due to incompatible classes with the same names. For example, the Java APIs of Tensorflow 1 and Tensorflow 2 are incompatible, which has slowed the adoption of newer versions of Tensorflow in software such as deepImageJ and disrupted the connection between the latest deep learning developments and the bioimage analysis and life sciences community.
   To address this issue, the library is designed in a modular way that creates a separate classloader for each DL framework when it is called, avoiding conflicts between the frameworks. To load frameworks in separate classloaders, the library requires that the executable JAR files be stored in identifiable folders, with all DL frameworks stored in the same directory. An example of this is shown in the images below:
   All engines should be stored in the same directory (in the example, C:\Users\carlos\icy\engines), with each engine following the naming convention:
   `<DL_framework_name>.<python_version>.<java_api_version>.<os>.<architecture>.<cpu_if_it_runs_in_cpu>.<gpu_if_it_runs_in_gpu>.`
   For example, the folder `Pytorch-1.11.0-1.11.0-windows-x86_64-cpu-gpu` contains a Pytorch engine, Python version 1.11.0, same as the Java version, for Windows operating system, architecture x86_64 (64 bits), that runs in CPU and GPU.
   Another example: the folder `Tensorflow-1.12.0-1.12.0-windows-x86_64-cpu` contains a Tensorflow engine, Python version 1.12.0, same as the Java version, for Windows operating system, architecture x86_64 (64 bits), that runs only in CPU.
   When examining the JAR files provided inside a particular engine folder, you will notice that they correspond to the standard dependencies of the framework, plus a JAR file containing the keyword `DlEngine`. This file needs to be created for each DL framework and enables the connection to the main model runner library. It needs to implement an interface
