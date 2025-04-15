# HierarQML

This tutorial guides on using HierarQcal to create quantum circuit models for classification tasks. It uses the modern stack of Pytorch Lightning and PennyLane for classical and quantum machine learning. This project is designed to be easily reusable. You can also copy the entire `qml_template` directory to create a new, independent project based on this codebase.

## Setup

Create a Python environment and install the required packages using

```
pip install -r requirements.text
```

Set the Python Path using `export PYTHONPATH=.` before executing any commands later on, or you will get errors like ModuleNotFoundError: No module named ''.

## Project Structure

To go through the main project structure, let's train a QCNN on GTZAN data.

### Directory Structure

```
├── hierarqml
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── gtzan.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── models
│   │   ├── base.py
│   │   ├── __init__.py
│   │   └── qcnn.py
│   └── utils.py
├── notebooks
│   └── music_genre_classification.ipynb
├── readme.md
├── requirements.txt
└── training
 ├── ansatz.py
 ├── __init__.py
 ├── run_expt.py
 └── util.py
```

The main code is `hierarqml` and `training`. `training` is used to train and experiment with the models. Within `hierarqml` we have `data` and `models`. `notebooks` is used for exploration and demonstration purposes.

### Data

`base_data_module` which inherits from `pl.LightningDataModule` is a simple base class to avoid writing the same boilerplate for the data sources. `gtzan` inherits from `base_data_module` and is the source for accessing the GTZAN dataset.

### Models

We use PyTorch-Lightning for training, which defines the `LightningModule` interface that handles not only everything that a Model handles but also specifies the details of the learning algorithm: what loss should be computed from the output of the model and the ground truth, which optimizer should be used, with what learning rate, etc. This is handled by the `base` file.

Users can define their quantum models by inheriting from `torch.nn.Module`. An example `qcnn` is defined in this way.

## Training

`training/run_expt.py` is a script that handles the training and has many command line arguments. Below is an example command:

```
python training/run_expt.py \
--data_class GTZAN \
--model_class QCNN \
--embedding AngleEmbedding \
--rotation X \
--data_path ./Data/features_30_sec.csv \
--pca 8 \
--batch_size 16 --max_epochs 2 --lr 0.01
```

Currently, the HierarQcal motif is defined int `training/ansatz` in the `get_motif` function. One can replace the code inside the function for custom motifs or can create separate Python files with `get_motif` defined in them. That file path needs to be passed as a command line argument `--ansatz {file_path}`.

The data embedding is handled by the function `setup_data_and_model_from_args` inside `training/util`. Specifically, it calls the `get_circuit` method to obtain a `QNode` which will be passed to the Quantum PyTorch Model. Currently, only angle and amplitude embedding are supported.