# Dynamic QCNNs

<img src="./img/dalle_img.png" alt="dalle image" style="height:150px;padding-right:10px" align="left"/>

<p style="height:150px">
<code>dynamic_qcnn</code> is an open-source python package for the dynamic generation of QCNN circuits by system or hand. It includes primitives: <code>Qconv, Qpool, Qdense, Qfree </code> that can be stacked together hierarchically to form QCNN circuit architectures. 
</p>
<br/>

*A robot building itself with artifical intelligence, pencil drawing -  generated with* [Dall E 2](https://openai.com/dall-e-2/)


# Example usage
```python
from dynamic_cnn import Qconv, Qpool, Qfree
qcnn = Qfree(8) + (Qconv(stride=1) + Qpool(filter="right")) * 3
```
$\text{QCNN:}$

<img src="./img/rbt_right.png" style="border:solid 2px black;">

```python
### Reverse binary tree
from dynamic_cnn import Qconv, Qpool, Qfree
# motif: level 1
m1_1 = Qconv(stride=2)
m1_2 = Qpool(filter="left")
# motif: level 2
m2_1 = m1_1 + m1_2
# motif: level 3
m3_1 = Qfree(8) + m2_1 * 3
```
$m^3_1\rightarrow \text{QCNN}:$

<img src="./img/rbt_left.png" style="border:solid 2px black;">

```python
# extending follows naturally, repeating the above circuit 5 times is just:
m3_1 * 5
```
## Installation
The package is still under development, to use it for the time being you can clone the project and install it as follow (on the `develop` branch):
```bash
cd path/to/project/
pip install -r requirements_core.txt
pip install .[cirq]
``` 
You only need numpy to use the core functionality, the other packages that gets installed are for visualization of the graphs (matplotlib and networkx) and the circuits (google's Cirq). The latter will be removed as a requirement soon.

## Usage
See `examples/examples_cirq.ipynb` for some basic usage.

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