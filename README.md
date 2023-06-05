# Multi-fidelity Multi-disciplinary Modelling Language (M3L)

<!---
[![Python](https://img.shields.io/pypi/pyversions/lsdo_project_template)](https://img.shields.io/pypi/pyversions/lsdo_project_template)
[![Pypi](https://img.shields.io/pypi/v/lsdo_project_template)](https://pypi.org/project/lsdo_project_template/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

[![GitHub Actions Test Badge](https://github.com/LSDOlab/m3l/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOLab/m3l/actions)
[![Forks](https://img.shields.io/github/forks/LSDOlab/m3l.svg)](https://github.com/LSDOlab/m3l/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/m3l.svg)](https://github.com/LSDOlab/m3l/issues)

A package for modularly specifying model data transfer. For docs, see https://m3l.readthedocs.io/en/latest/index.html.

# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/m3l.git
```
If you want users to install a specific branch, run
```sh
pip install git+https://github.com/LSDOlab/m3l.git@branch
```

**Enabled by**: `packages=find_packages()` in the `setup.py` file.

## Installation instructions for developers
To install `m3l`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/m3l.git
pip install -e ./m3l
```

# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.
