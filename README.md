# HierarQcal 

<img src="https://github.com/matt-lourens/hierarqcal/blob/master/img/dalle_img.png?raw=true" alt="dalle image" height="150" style="padding-right:10px" align="left"/>

<p style="height:150px">
<b>HierarQcal</b> is a quantum circuit builder that simplifies circuit design, composition, generation, scaling and parameter management. It provides an intuitive and dynamic data structure for constructing computation graphs hierarchically. This enables the generation of complex quantum circuit architectures, which is particularly useful for Neural Architecture Search (NAS), where an algorithm can determine the most efficient circuit architecture for a specific task and hardware. HierarQcal also facilitates the creation of hierarchical quantum circuits, such as those resembling tensor tree networks or MERA, with a single line of code. The package is open source and framework-agnostic, it includes tutorials for Qiskit, PennyLane, and Cirq. Built to address the unique challenges of applying NAS to Quantum Computing, HierarQcal offers a novel approach to explore and optimize quantum circuit architectures. 
</p>
<br/>

*A robot building itself with artificial intelligence, pencil drawing -  generated with* [Dall E 2](https://openai.com/dall-e-2/)
___


[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


## Quick example

#### Building a [Quantum Convolutional Neural Network (QCNN)](https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html) with one line of code:

```python
from hierarqcal import Qinit, Qcycle, Qmask
hierq = Qinit(8) + (Qcycle(mapping=u) + Qmask("right", mapping=v))*3
```

<img src="https://github.com/matt-lourens/hierarqcal/blob/master/img/rbt_right.png?raw=true" style="border:solid 2px black;">

#### Modular and hierarchical circuit building:
```python
### Reverse binary tree
from hierarqcal import Qinit, Qcycle, Qmask
# motif: level 1
m1_1 = Qcycle(stride=2)
m1_2 = Qmask(pattern="left")
# motif: level 2
m2_1 = m1_1 + m1_2
# motif: level 3
m3_1 = Qinit(8) + m2_1 * 3
```
$m^3_1\rightarrow \text{QCNN}:$

<img src="https://github.com/matt-lourens/hierarqcal/blob/master/img/rbt_left.png?raw=true" style="border:solid 2px black;">

```python
# extending follows naturally, repeating the above circuit 5 times is just:
m3_1 * 5
```
## Installation
[![PyPI version](https://badge.fury.io/py/hierarqcal.svg)](https://badge.fury.io/py/hierarqcal)

<code>HierarQcal</code> is hosted on [pypi](https://pypi.org/project/hierarqcal/) and can be installed via pip:

```bash
# Based on the quantum computing framework you use, choose one of:
pip install hierarqcal[cirq]
# or
pip install hierarqcal[qiskit]
# or
pip install hierarqcal[pennylane]
```

The package is quantum computing framework independent, there are helper functions for Cirq, Qiskit and Pennylane to represent the circuits in their respective frameworks. You can also use the the package independent of any framework, if you want to do this just run:
```bash
pip install hierarqcal
```

## Tutorial and Documentation
There are quickstart tutorials for each major Quantum computing framework: 
 - [HierarQcal Cirq Tutorial](https://github.com/matt-lourens/hierarqcal/blob/master/examples/examples_cirq.ipynb)
 - [HierarQcal Qiskit Tutorial](https://github.com/matt-lourens/hierarqcal/blob/master/examples/examples_qiskit.ipynb) 
 - [HierarQcal Pennylane Tutorial](https://github.com/matt-lourens/hierarqcal/blob/master/examples/examples_pennylane.ipynb). 
 
 For more detailed usage see the [documentation](https://matt-lourens.github.io/hierarqcal/index.html).

## Contributing
We welcome contributions to the project. Please see the [contribution guidelines](https://github.com/matt-lourens/hierarqcal/blob/master/CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.

## License
BSD 3-Clause "New" or "Revised" License, see [LICENSE](https://github.com/matt-lourens/hierarqcal/blob/master/LICENSE.txt) for more information.

## Citation
```latex
@article{lourens2023hierarchical,
      title={Hierarchical quantum circuit representations for neural architecture search},
      url = {https://arxiv.org/abs/2210.15073},
      author={Matt Lourens and Ilya Sinayskiy and Daniel K. Park and Carsten Blank and Francesco Petruccione},
      year={2023},
      eprint={2210.15073},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```