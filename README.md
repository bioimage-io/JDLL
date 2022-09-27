# Deep Learning

Deep Learning plugin.
Responsible of managing Deep Learning framework (TensorFlow, PyTorch..)


## Overview of java libraries

- the main library that implements the functionality for running bioimage.io models in java is this one ([model-runner-java](https://github.com/bioimage-io/model-runner-java))
- for running a model with a given weight format additional interface libraries are needed:
  - [tensorflow-2-java.interface](https://github.com/bioimage-io/tensorflow-2-java.interface) for models with tensorflow 2 weights
  - [tensorflow-1-java-interface](https://github.com/bioimage-io/tensorflow-1-java-interface) for models with tensorflow 1 weights
  - [pytorch-java-interface](https://github.com/bioimage-io/pytorch-java-interface) for models with pytorch weights
