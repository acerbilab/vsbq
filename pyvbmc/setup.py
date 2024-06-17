from setuptools import find_packages, setup

setup(
    name="pyvbmc",
    version="0.1.0",
    description="VSBQ (based on pyvbmc implementations)",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pytest",
        "sphinx",
        "numpydoc",
        "cma",
        "corner",
        "imageio",
        "kaleido",
        "myst_nb",
        "numpydoc",
        "sphinx",
        "sphinx-book-theme",
        "pylint",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-rerunfailures",
        "plum-dispatch",
        "scikit-learn",
        "tqdm",
        "ipywidgets",
        "jaxgp @ git+ssh://git@github.com/pipme/JaxGP.git",
    ],
)
