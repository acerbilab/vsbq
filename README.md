# Code for "Fast post-process Bayesian inference with Variational Sparse Bayesian Quadrature"

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
