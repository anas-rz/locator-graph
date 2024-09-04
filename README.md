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
- `sklearn`
- `transformers`
- `keras`
- `torch-geometric`

## Setup
Before you begin, ensure that you have the following software installed on your system:

- Conda: This project uses Conda to manage the Python environment and dependencies. Conda simplifies package management and deployment, making it easier to create isolated environments for different projects.
- Python 3.8 or later: The project is developed and tested using Python 3.8. While it may work with other versions, using the specified version is highly recommended to avoid compatibility issues.

### Create a Conda Environment

To ensure that all dependencies are properly managed and isolated, we will create a new Conda environment for the project. This environment will include Python and all necessary libraries required to run the code.

Run the following command to create a Conda environment:

`conda create --name yourenvname python=3.8`

Replace yourenvname with a name of your choice for the environment. You can use a name that reflects the project or purpose, such as `fault_localization_env`.

Once the environment is created, activate it using the following command:

```
conda activate yourenvname
```

With the Conda environment activated, you can now install the project's dependencies. We use pip for package management, and all required packages are listed in the requirements.txt file.

To install the dependencies, run:

```
pip install -r requirements.txt
```

To verify package installation use:
```
pip list
```

With everything set up, you are now ready to run the project. Depending on the project structure, you may need to run a specific script or start a service.

## Training
Our model has been trained for three tasks:

### Bug Identification
The task of binary classification is focused on detecting whether a sequence-based model is buggy or not. With everything set up, you are now ready to run the binary classification task. 
```
python locator_graph/bug_identification.py --help
python locator_graph/bug_identification.py --dataset_path /path_to/dataset.csv --base_dir /base/dir --epochs 200 --test_size 0.25 --val_size 0.1
```

### Node Detection
This task aims to identify faulty nodes within the trace graphs of SBMs. The following notebook contains details and results of this task:
```
python locator_graph/node_classification.py --help
python locator_graph/node_classification.py --input_csv /path_to/dataset.csv --base_dir /base/dir
```

### Feature Inspection
In this final task, the goal is to locate bug-causing features within the identified faulty nodes. More details can be found in the following notebook:
- [Feature Inspection Notebook](./locator_graph/feature_inspector.ipynb)

## Fault Injector
Based on [**DeepCrime**](https://github.com/dlfaults/deepcrime), we developed a [**Fault Injector**](./locator_graph/fault_injector/) specifically designed for Sequence-Based Models. This tool enables the introduction of bugs into SBMs to test the effectiveness of fault localization techniques.

To explore this component, please check out the corresponding materials and experiments.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

