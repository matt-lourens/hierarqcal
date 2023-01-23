# HierarQcal

<img src="https://github.com/matt-lourens/hierarqcal/blob/master/img/dalle_img.png?raw=true" alt="dalle image" height="150" padding-right="10" align="left"/>

<p style="height:150px">
**HierarQcal** is an Open-Source Python Package for Building Custom Quantum Circuits for Machine Learning. The package simplifies the process of creating general quantum convolutional neural networks (QCNN) by enabling an hierarchical design process. With HierarQcal, automatic generation of QCNN circuits is made easy, and it facilitates QCNN search space design for neural architecture search (NAS). The package includes primitives such as <code>Qconv, Qpool, Qdense, Qfree </code> that can be stacked together hierarchically to form complex QCNN circuit architectures.
</p>
<br/>

*A robot building itself with artifical intelligence, pencil drawing -  generated with* [Dall E 2](https://openai.com/dall-e-2/)


## Quick example
```python
from hierarqcal import Qconv, Qpool, Qfree
qcnn = Qfree(8) + (Qconv(stride=1) + Qpool(filter="right")) * 3
```
$\text{QCNN:}$

<img src="https://github.com/matt-lourens/hierarqcal/blob/master/img/rbt_right.png?raw=true" style="border:solid 2px black;">

```python
### Reverse binary tree
from hierarqcal import Qconv, Qpool, Qfree
# motif: level 1
m1_1 = Qconv(stride=2)
m1_2 = Qpool(filter="left")
# motif: level 2
m2_1 = m1_1 + m1_2
# motif: level 3
m3_1 = Qfree(8) + m2_1 * 3
```
$m^3_1\rightarrow \text{QCNN}:$

<img src="https://github.com/matt-lourens/hierarqcal/blob/master/img/rbt_left.png?raw=true" style="border:solid 2px black;">

```python
# extending follows naturally, repeating the above circuit 5 times is just:
m3_1 * 5
```
## Installation
<code>HierarQcal</code> will be published soon! For the time being you can install it as follows:

Clone the project and run the following commands (on the `develop` branch):
```bash
cd path/to/project/
pip install -r requirements_core.txt
# Based on the quantum computing framework you use, choose one of:
pip install .[cirq]
# or
pip install .[qiskit]
# or
pip install .[pennylane]
```
The package is quantum computing framework independent, there are helper functions for Cirq, Qiskit and Pennylane to represent the circuits in their respective frameworks. You can also use the the package independent of any framework, to do this install it with:
```bash
pip install .
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
@article{lourensArchitectureRepresentationsQuantum2022,
  doi = {10.48550/ARXIV.2210.15073},
  url = {https://arxiv.org/abs/2210.15073},
  author = {Lourens, Matt and Sinayskiy, Ilya and Park, Daniel K. and Blank, Carsten and Petruccione,   Francesco},
  keywords = {Quantum Physics (quant-ph), Artificial Intelligence (cs.AI)},
  title = {Architecture representations for quantum convolutional neural networks},
  publisher = {arXiv},
  journal = {arXiv:2210.15073[quant-ph]},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```