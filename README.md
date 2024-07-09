# GBMPurity

## Overview
GBMPurity is a machine learning-based tool designed to estimate the tumour purity of Glioblastoma Multiforme (GBM) samples from single-cell RNA sequencing data..

We have made this tool available as an easy-to-use web application at https://gbmdeconvoluter.leeds.ac.uk/.

However, if you would like to run GBMPurity locally or access the PyTorch model, following the installation instructions below.

## Table of Contents
- [GBMPurity](#gbmpurity)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Inference on bulk samples](#inference-on-bulk-samples)
    - [Access GBMPurity model](#access-gbmpurity-model)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/GBMPurity.git
    cd GBMPurity
    ```

2. Navigate to the cloned repository:
    ```bash
    cd GBMPurity
    ```

3. Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    ```


## Usage
### Data Preparation
Ensure that your input data meets the required format as described.

### Inference on bulk samples
To run inference on bulk samples organised in a matrix as described above, run the following command (replace the -d argument with the path to your data):
```bash
python src/GBMPurity.py -d /path/to/your/data.csv
```
### Access GBMPurity model
You can access the PyTorch model in the directory ```GBMPurity/model/GBMPurity.pt```
