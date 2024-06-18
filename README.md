# Code for "[Fast post-process Bayesian inference with Variational Sparse Bayesian Quadrature](https://arxiv.org/abs/2303.05263)"

Variational Sparse Bayesian Quadrature (VSBQ) is a fast post-process Bayesian inference method for (potentially expensive) Bayesian models. It operates by recycling existing likelihood/density evaluations (e.g., from maximum-a-posteriori (MAP) optimization runs), fitting a regression surrogate (a sparse Gaussian process), and conducting variational inference to get a posterior approximation. Our current implementation is based on [PyVBMC](https://github.com/acerbilab/pyvbmc). `benchflow` is a toolkit for running the benchmark experiments in the paper.

## Installation
```bash
conda create -n vsbq python=3.9
conda activate vsbq
pip install -e ./benchflow
pip install -e ./pyvbmc
# Install the kernel for Jupyter
python -m ipykernel install --user --name vsbq 
```

## Example

See the [example notebook](./example.ipynb) for a simple example of using VSBQ.

## Citation
Please cite our paper if you find this work useful:
```
@misc{liFastPostprocessBayesian2024,
  title = {Fast Post-Process {{Bayesian}} Inference with {{Variational Sparse Bayesian Quadrature}}},
  author = {Li, Chengkun and Clart{\'e}, Gr{\'e}goire and Jørgensen, Martin and Acerbi, Luigi},
  year = {2024},
  number = {arXiv:2303.05263},
  eprint = {2303.05263},
  primaryclass = {cs, stat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2303.05263},
  archiveprefix = {arxiv}
}
```

