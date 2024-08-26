# LocatorGraph: Fault Localization in Sequence-Based Models using Graph Neural Networks

## Table of Contents
- [Abstract](#abstract)
- [Experimental Details](#experimental-details)
- [Setup](#setup)
- [Training](#training)
  - [Binary Classification](#binary-classification)
  - [Node Detection](#node-detection)
  - [Feature Inspection](#feature-inspection)
- [Fault Injector](#fault-injector)
- [Citation](#citation)
- [License](#license)

## Abstract
Deep learning models, particularly Sequence-based
models (SBMs) like Recurrent Neural Networks (RNNs), LSTM,
GRU, Transformer-based, and patch-based architectures play a
pivotal role in intelligent software systems. However, like any
software application, SBMs applications are susceptible to bugs.
Bug-fix patterns in SBMs differ from traditional techniques,
primarily due to their inherent black-box nature, data-driven
nature, and sequential dependencies. Moreover, current methods,
although tailored for generic deep neural network (DNN) struc-
tures, are time-consuming, require specialized expertise, and are
not directly applicable to the unique structure and requirements
of SBMs. To address these issues, we propose LocatorGraph, a
novel graph neural network-based tool to identify the root causes
of faults in SBMs.
We convert SBM code to trace graphs which are then fed
to the proposed framework LocatorGraph, a data-driven tool,
that employs a graph neural network model to trace graphs
for SBMs, enabling it to effectively detect and localize bugs
in real-world SBMs. Beyond mere identification, LocatorGraph
locates faulty nodes in the generated tracegraphs after which
the bug-causing faulty features are located for repair. Through
rigorous evaluation of 160 diverse models including generated
buggy models, LocatorGraph outperforms existing methods in
fault localization, showing robustness in identifying potential
problems with an AUC of 79.39% and F1 score of 80.08%. On
Node Detection, LG gives an accuracy of 75.84% and an F1-score
of 76.26%. In the final phase, LG outperforms all baselines to
locate bug-causing features in the nodes with an accuracy of
74.32% and F1 score 73.21%. We also provide details of the
improvement observed in the models after fixing bugs using
LocatorGraph based approach

## Experimental Details
We use the following libraries to perform our experiments:
- **sklearn**
- **transformers**
- **keras**
- **torch-geometric**

## Setup
Experiments have been performed in notebooks on Google Colab and the High-Performance Cluster at Oakland University.

To explore our experimentation, please check out the notebooks provided in this repository.

## Training
Our model has been trained for three tasks:

### Binary Classification
The task of binary classification is focused on detecting whether a sequence-based model is buggy or not. You can explore the details of this task in the following notebook:
- [Binary Classification Notebook](./locator_graph/binary_notebook.ipynb)

### Node Detection
This task aims to identify faulty nodes within the trace graphs of SBMs. The following notebook contains details and results of this task:
- [Node Detection Notebook](./locator_graph/node_classification.ipynb)

### Feature Inspection
In this final task, the goal is to locate bug-causing features within the identified faulty nodes. More details can be found in the following notebook:
- [Feature Inspection Notebook](./locator_graph/feature_inspector.ipynb)

## Fault Injector
Based on **DeepCrime**, we developed a **Fault Injector** specifically designed for Sequence-Based Models. This tool enables the introduction of bugs into SBMs to test the effectiveness of fault localization techniques.

To explore this component, please check out the corresponding materials and experiments.

## Citation
If you use this tool in your research, please cite our work as follows:

```
@article{YourLastName2024LocatorGraph,
  title={LocatorGraph: Fault Localization in Sequence-Based Models using Graph Neural Networks},
  author={YourFirstName, YourLastName and CollaboratorFirstName, CollaboratorLastName and OtherAuthorFirstName, OtherAuthorLastName},
}
```