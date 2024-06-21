<p align="center">
  <p align="center">
   <h1 align="center">Vectorized Conditional Neural Field (VCNeF)</h1> 
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://jhagnberger.github.io/"><strong>Jan Hagnberger</strong></a>,
    <a target="_blank" href="https://kmario23.github.io/"><strong>Marimuthu Kalimuthu</strong></a>,
    <a target="_blank" href="https://www.ki.uni-stuttgart.de/institute/team/Musekamp-00001/"><strong>Daniel Musekamp</strong></a>,
    <a target="_blank" href="https://matlog.net/"><strong>Mathias Niepert</strong></a>
  </p>

 <p align="center">
  <a href='https://icml.cc/virtual/2024/poster/32919'><img src="https://img.shields.io/badge/ICML'24%20Main-Conference-red?style=flat&logoSize=auto&labelColor=darkgreen" alt="ICML Conference"></a>
  <a href='https://iclr.cc/virtual/2024/21341'><img src="https://img.shields.io/badge/AI4DifferentialEquations%20in%20Science%40ICLR'24-Conference-red?style=flat&logoSize=auto&labelColor=darkgreen" alt="ICLR Conference"></a>
 </p>

#
 
This repository contains the official PyTorch implementation of the VCNeF model from the ICML'24 paper,  
"[**Vectorized Conditional Neural Fields: A Framework for Solving Time-dependent Parametric Partial Differential Equations**](https://arxiv.org/abs/2406.03919)".

## Requirements

The VCNeF model requires and is tested with the following packages.
- [PyTorch](https://pytorch.org) in version 2.2.0
- [NumPy](https://numpy.org) in version 1.26.4
- [Einops](https://einops.rocks) in version 0.7.0


Please also see the [``requirements.txt``](./requirements.txt) file which contains all packages to run the provided examples.


## Usage

The following example shows how to use the VCNeF model.

```python
import torch
from vcnef.vcnef_1d import VCNeFModel as VCNeF1DModel
from vcnef.vcnef_2d import VCNeFModel as VCNeF2DModel
from vcnef.vcnef_3d import VCNeFModel as VCNeF3DModel

model = VCNeF2DModel(num_channels=4,
                     condition_on_pde_param=True,
                     pde_param_dim=2,
                     d_model=256,
                     n_heads=8,
                     n_transformer_blocks=1,
                     n_modulation_blocks=6)

# Random data with shape b, s_x, s_y, c
x = torch.rand(4, 64, 64, 4)
grid = torch.rand(4, 64, 64, 2)
pde_param = torch.rand(4, 2)
t = torch.arange(1, 21).repeat(4, 1) / 20

y_hat = model(x, grid, pde_param, t)
```

## Files
Below is a listing of the directory structure of VCNeF.

``examples.py``: Contains lightweight examples of how to use VCNeF. \
``examples_pde_bench.py``: Contains examples of how to use VCNeF with PDEBench data and the PDEBench training loop. \
``📂 vcnef``: Contains the code for the VCNeF model. \
``📂 utils``: Contains utils for the PDEBench example.


## Dataset for PDEBench Example

To use the PDEBench example ``examples_pde_bench.py``, you have to download the [PDEBench datasets](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986). An overview of the avaiable data and how to download it can be found in the [PDEBench repository](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download). To use the downloaded datasets in the example, you have to adapt the path in ``base_path`` and the file name(s) in ``file_names``.


## VCNeF Architecture
The following illustation shows the architecture of the VCNeF model for solving 2D time-dependent PDEs (e.g., Navier-Stokes equations).

![VCNeF Architecrture](img/vcnef_architecture.svg)


## Acknowledgements

The code of VCNeF is based on the code of [Linear Transformers](https://github.com/idiap/fast-transformers) and [PDEBench](https://github.com/pdebench/PDEBench). We would like to thank the authors of Linear Transformers and PDEBench for their work, which made our method possible.


## License

MIT licensed, except where otherwise stated. Please see [`LICENSE`](./LICENSE) file.


## Citation
If you find our project useful, please consider citing it.

```bibtex
@inproceedings{vcnef2024,
author = {Hagnberger, Jan and Kalimuthu, Marimuthu and Musekamp, Daniel and Niepert, Mathias},
title = {{Vectorized Conditional Neural Fields: A Framework for Solving Time-dependent Parametric Partial Differential Equations}},
year = {2024},
booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML 2024)}
}
```
