Getting Started
===============

Installation
------------

HierarQcal will be published soon! For the time being you can install it as follows:

Clone the project and run the following commands (on the `develop` branch):

.. code-block:: console

    (.venv) $ cd path/to/project/
    (.venv) $ cd pip install -r requirements_core.txt

Then based on the quantum computing framework you use, choose one of:

.. code-block:: console

    (.venv) $ pip install .[cirq]

or

.. code-block:: console

    (.venv) $ pip install .[qiskit]

or

.. code-block:: console

    (.venv) $ pip install .[pennylane]

or if you only want to use hierarQcal core functionality

.. code-block:: console

    (.venv) $ pip install .

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
