Getting Started
===============

Installation
------------

HierarQcal is hosted on [pypi](https://pypi.org/project/hierarqcal/) and can be installed via pip:

Based on the quantum computing framework you use, choose one of:

.. code-block:: console

    (.venv) $ pip install hierarqcal[cirq]

or

.. code-block:: console

    (.venv) $ pip install hierarqcal[qiskit]

or

.. code-block:: console

    (.venv) $ pip install hierarqcal[pennylane]

or if you only want to use hierarQcal core functionality

.. code-block:: console

    (.venv) $ pip install hierarqcal

The package is quantum computing framework independent, there are helper functions for Cirq, Qiskit and Pennylane to represent the circuits in their respective frameworks.

Basic usage
------------
There are tutorials for each quantum computing framework that cover most of the package's functionality:

.. toctree::
    :maxdepth: 2
    :caption: Tutorials
    
    examples/examples_cirq
    examples/examples_qiskit
    examples/examples_pennylane
