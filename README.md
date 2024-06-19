# One-Class Multi-Instance Support Vector Machine (OCMISVM)
This repository contains a collection of Python scripts implementing the One-Class Multi-Instance Support Vector Machine (OCMISVM) model and its experiments on synthetic and Chest X-Ray multi-instance datasets. It covers data creation, building baseline models, calculating classification performance metrics, and visualizing the results.

## Project Description

- `run.py`: Imports baseline models from [kenchi](https://github.com/Y-oHr-N/kenchi), [pyod](https://github.com/yzhao062/pyod), [sklearn](https://scikit-learn.org/stable/), [deepod](https://pypi.org/project/deepod/), and our OCMISVM model. Configures parameters, creates the one-class multi-instance dataset, splits the dataset, conducts anomaly detection, and reports results in terms of accuracy, precision, recall, F1 score, and balanced accuracy.
- `utils.py`: Contains utility functions for experiments, processes raw chest X-ray data, creates the multi-instance dataset for anomaly detection, and includes visualization code for plotting important instances identified by the OC classification model.
- `MISVM.py`: Implements the OCMISVM model, featuring the class OCMISVM with three key methods: __init__, fit, and predict. The __init__ method defines the hyperparameters of OCMISVM. The fit method takes the multi-instance training data X and its associated grouping information y (with ungrouped bags marked as 'unlabeled'). The predict method also takes multi-instance data X and their grouping information, outputting decisions where 1 and -1 indicates the inlier (normal) and outlier (abnormal) data for the bags in X.
- `other_models.py`: Includes baseline outlier detection models for comparison. Contains wrappers for sklearn, kenchi, pyod, and deepod models, with each wrapper defining the fit and predict methods to consistently output 1 for inlier and -1 for outlier data.

## Getting Started
### Dependencies

- Python 3.9.16
- Python packages listed in requirements.txt

### Installation
To set up the project, start by cloning the repository to your local machine. It is highly recommended to create an isolated Python environment via conda. For example:
```
conda create -n "OCMISVM" python=3.9
```

Then, activate the created environment:
```
conda activate "OCMISVM"
```

Next, install the required packages:
```
pip install requirements.py
```

## Running the tests

After installing the packages, you can conduct the experiments by running:

```
python3 run.py
```
A folder will be created under the 'output_path' in `run.py`, containing all the experimental results.

## Data

The necessary data for executing the framework's scripts is readily accessible in the [Chest X-ray dataset repository](https://github.com/ieee8023/covid-chestxray-dataset). Clone this repository to your local machine and specify the path to the cloned dataset in `run.py`.

## Contributing

We welcome contributions to improve the framework and extend its functionalities. Please feel free to fork the repository, make your changes, and submit a pull request.
