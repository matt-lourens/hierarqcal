# Dynamic QCNNs

<img src="./img/dalle_img.png" alt="dalle image" style="height:150px;padding-right:10px" align="left"/>

<p style="height:150px">
<code>dynamic_qcnn</code> is an open-source python package for the dynamic creation of QCNNs by system or hand. It includes primitives: <code>Qconv, Qpool, Qdense, Qfree </code> that can be stacked together hierarchically to form QCNN circuit architectures. 

*A robot building itself with artifical intelligence, pencil drawing -  generated with* [Dall E 2](https://openai.com/dall-e-2/)
</p>

# Example usage
```python
from dynamic_cnn import Qconv, Qpool, Qfree
qcnn = Qfree(8) + (Qconv(stride=1) + Qpool(filter="right")) * 3
```
<img src="./img/rbt_right.svg" style="background:white;border:solid 2px black">

