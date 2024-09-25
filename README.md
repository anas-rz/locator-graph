# LocatorGraph: Fault Localization in Sequence-Based Models using Graph Neural Networks

## Table of Contents
- [Experimental Details](#experimental-details)
- [Setup](#setup)
- [Training](#training)
  - [Binary Classification](#binary-classification)
  - [Node Detection](#node-detection)
  - [Feature Inspection](#feature-inspection)
- [Fault Injector](#fault-injector)
- [Citation](#citation)
- [License](#license)


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

### Graph-Based Fault Inspector (GBFI)
This task aims to identify faulty features within the trace graphs of SBMs. The following script contains details and results of this task:
```
python locator_graph/gbfi.py
python locator_graph/gbfi.py --input_csv /path_to/dataset.csv --base_dir /base/dir
```

### Node-Based Fault Inspector (NBFI)
In this final task, the goal is to locate bug-causing features within the identified faulty nodes. More details can be found in the following notebook:
```
python locator_graph/nbfi.py --help
python locator_graph/nbfi.py --input_file /path_to/dataset.csv
```
The input file for feature inspector has been manually generated using a script to collect all features from the faulty layers in the model code files.

## Fault Injector
Based on [**DeepCrime**](https://github.com/dlfaults/deepcrime), we developed a [**Fault Injector**](./locator_graph/fault_injector/) specifically designed for Sequence-Based Models. This tool enables the introduction of bugs into SBMs to test the effectiveness of fault localization techniques.

To explore this component, please check out the corresponding materials and experiments.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

