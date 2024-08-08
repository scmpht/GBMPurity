# GBMPurity

## Table of Contents
- [GBMPurity](#gbmpurity)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Inference on bulk samples](#inference-on-bulk-samples)
    - [Access GBMPurity model](#access-gbmpurity-model)
  - [Development and Support](#development-and-support)
  - [Disclaimers](#disclaimers)
  - [Citation](#citation)

## Overview
GBMPurity is a machine learning model designed to estimate the purity of bulk RNA-seq primary IDH-wildtype Glioblastoma (GBM) samples. In this context, purity refers to the proportion of malignant cells within a tumour sample.

Our model is a multi-layer perceptron trained on simulated pseudobulk tumours of known purity. These simulations were created using the GBmap single-cell resource ([Ruiz-Moreno et al., 2022](https://doi.org/10.1101/2022.08.27.505439)).

For a detailed description of GBMPurity's architecture, training methodology, and performance metrics, please refer to our [preprint]([https://www.biorxiv.org/](https://www.biorxiv.org/content/10.1101/2024.07.11.602650v1)).

<image src="./img/GBMPurity.png" width="100%"/>

We have made this tool available as an easy-to-use web application at https://gbmdeconvoluter.leeds.ac.uk/.

However, if you would like to run GBMPurity locally or access the PyTorch model, following the installation instructions below.




## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/GBMPurity.git
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

## Development and Support

GBMPurity is a product of the Glioma Genomics group at the University of Leeds.

For inquiries, support, and feedback, please contact us at: scmpht@leeds.ac.uk

## Disclaimers

- GBMPurity is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement.
- This tool is intended for research purposes only and should not be used for clinical decision making.
- While we strive for accuracy, users should be aware that no computational model is perfect. Always interpret results in the context of our reported methods and performance.
- GBMPurity was built using primary IDH-wildtype Glioblastoma, and thus is not suitable for inference on any other tissue type.
- We are not responsible for any actions taken based on the output of this tool.

## Citation

This tool is free to use, but we kindly request that you cite our paper in any published work that uses GBMPurity:

> Thomas, M., <i>et al.</i> (2024). GBMPurity: A Machine Learning Tool for Estimating Glioblastoma Tumour Purity from Bulk RNA-seq Data. <i>BioRxiv</i> [Preprint]. DOI: doi:10.1101/2024.07.11.602650

For use in bibliographic management software, you can use the following BibTeX entry:

    @article{Thomas2024GBMPurity,
      title={GBMPurity: A Machine Learning Tool for Estimating Glioblastoma Tumour Purity from Bulk RNA-seq Data},
      author={Thomas, M. and [Other Authors]},
      journal={[Journal Name]},
      volume={[Volume]},
      number={[Issue]},
      pages={[Page numbers]},
      year={2024},
      publisher={[Publisher]},
      doi={[DOI number]}
    }
