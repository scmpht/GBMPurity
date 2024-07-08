"""
GBMPurity.py

This script performs inference using the GBMPurity model on given input data.
The input data path is specified via command line arguments.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import sys
sys.path.insert(1, './src/modelling')
from torch_models import MLP2h
from utils import tpm

def check_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Perform data checks on the input DataFrame.

    Parameters:
        df (pd.DataFrame): Input data.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Processed data and a list of warnings.
    
    Raises:
        ValueError: If any errors are found in the data.
    """
    warnings = []

    # Import required genes
    gene_lengths = pd.read_csv("./model/input-genes-lengths.csv")
    genes = gene_lengths['feature_name']

    # Check Dimensions
    if df.shape[0] < 1:
        raise ValueError("We didn't detect any genes.")
    if df.shape[1] <= 1:
        raise ValueError("We didn't detect any samples.")

    # Missing values
    if df.isnull().values.any():
        warnings.append(f"We found {df.isnull().values.sum()} missing value(s). These will be converted to 0.")
        df = df.fillna(0)

    # Check for duplicate genes
    input_genes = df.iloc[:, 0]
    duplicate_genes = input_genes[input_genes.duplicated()].unique()
    if len(duplicate_genes) > 0:
        warnings.append(f'We found {len(duplicate_genes)} duplicate gene(s). Counts for these genes will be summed for each sample.')

    data = df.set_index(df.columns[0])
    data = data.groupby(data.index).sum()

    # Check appropriate genes
    overlap = set(input_genes).intersection(set(genes))
    if len(overlap) == 0:
        raise ValueError("We didn't find any required genes. Are the provided genes in the HGNC format e.g. CD47?")
    else:
        p_overlap = len(overlap) / len(genes)
        if p_overlap < 0.8:
            raise ValueError(f"We found {int(p_overlap * 100)}% of the required genes. Purity estimates will be unreliable under 80%.")
        elif p_overlap < 0.99:
            warnings.append(f"We found {int(p_overlap * 100)}% of the required genes. Note that GBMPurity tends to underestimate the tumor purity with more missing genes.")

    # Check correct data
    # Non-numeric
    non_numeric = data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all()
    if not non_numeric:
        raise ValueError("All gene expression values must be numeric.")

    # Negative values
    if (data.values < 0).any():
        raise ValueError("Gene expression values must be non-negative. Data should be uploaded as raw counts (without batch correction).")

    # Process data to return
    data = data.T
    data = data.reindex(columns=genes, fill_value=0)

    return data, warnings

def gbm_purity_inference(data_path: str) -> pd.DataFrame:
    """
    Perform inference to predict GBM purity using the GBMPurity model.

    Parameters:
        data_path (str): Path to the input data CSV file.

    Returns:
        pd.DataFrame: DataFrame with sample names and predicted purities.
    """
    # Import gene lengths
    gene_lengths = pd.read_csv("./model/input-genes-lengths.csv")
    lengths = gene_lengths['feature_length'].values

    # Load and transform input data
    data = pd.read_csv(data_path)
    data, warnings = check_data(data)

    for warning in warnings:
        print(f"Warning: {warning}")
    if warnings:
        proceed = input("Warnings detected. Do you want to continue? (y/n): ")
        if proceed.lower() != 'y':
            raise Exception("Inference process aborted by user due to warnings.")

    X = np.log2(tpm(data.values, lengths) + 1)

    # Import model
    model = torch.load("./model/GBMPurity.pt")
    model.eval()

    # Input to GBMPurity model
    y_pred = model(torch.tensor(X).float()).detach().numpy().flatten().clip(0, 1)

    samples = data.index.values
    results = pd.DataFrame({'Sample': samples, 'Purity': y_pred})
    return results

def main() -> None:
    """
    Main function to parse arguments and run the inference.
    """
    parser = argparse.ArgumentParser(description="Perform inference using the GBMPurity model.")
    parser.add_argument('-d', '--data_path', type=str, required=True, help="Path to the inference data CSV file.")
    args = parser.parse_args()

    purities = gbm_purity_inference(args.data_path)
    file_name = args.data_path.split("/")[-1]
    print(purities)
    save_path = f"./results/GBMPurity_estimates({file_name})"
    print(f"Results saved to {save_path}")
    purities.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()

