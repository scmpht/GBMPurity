"""
trainGBMPurity.py

This script trains GBMPurity using single-cell RNA-sequencing data from GBMap resource (Ruiz-Moreno et al., 2022).
It includes data preprocessing, model definition, and training with early stopping.


Author:
    Morgan Thomas <scmpht@leeds.ac.uk>
    Date: 2024-07-08
"""

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from purity_dataset import singleCellData
from torch_models import MLP2h
from ..utils import tpm, accuracy, ccc

np.random.seed(80085)
torch.manual_seed(42)

def load_data(data_path, lengths_path):
    """
    Load and preprocess the data.
    
    Parameters:
        data_path (str): Path to the single-cell data file.
        lengths_path (str): Path to the gene lengths CSV file.
    
    Returns:
        AnnData: Preprocessed single-cell data.
    """
    # Single cell data
    train_data = ad.read_h5ad(data_path)

    # TPM Normalisation
    gene_lengths = pd.read_csv(lengths_path)['feature_length'].values
    train_data.layers['TPM'] = tpm(train_data.X, gene_lengths)
    train_data.layers['logTPM'] = np.log2(train_data.layers['TPM'] + 1)

    return train_data

def train_model(train_data, model, loss_fn, optimizer, batch_size=64, patience_lim=200, avg_val_window=25, max_batches=5000, model_save_path="../model/GBMPurity.pt"):
    """
    Train the model with early stopping.
    
    Parameters:
        train_data (AnnData): Preprocessed training data.
        model (nn.Module): The PyTorch model to be trained.
        loss_fn: Loss function.
        optimizer: Optimizer.
        batch_size (int, optional): Batch size for training. Default is 64.
        patience_lim (int, optional): Patience limit for early stopping. Default is 200.
        avg_val_window (int, optional): Window size for average validation loss. Default is 25.
        max_batches (int, optional): Maximum number of batches to train. Default is 5000.
        model_save_path (str, optional): Path to save the optimal model. Default is "../model/GBMPurity.pt".
    
    Returns:
        None
    """
    train_dataset = singleCellData(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # Tracking for plot
    training_samples = []
    loss_values = []
    avg_loss_values = []
    total_training_samples = 0

    # Adding early stopping
    patience = 0
    prev_loss = float('inf')
    stopped_decreasing = False

    for i, (inputs, labels) in enumerate(train_dataloader):
        if stopped_decreasing or i >= max_batches:
            break

        # Model training
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = loss_fn(outputs, labels.view(-1, 1).float())

        training_samples.append(total_training_samples)
        loss_values.append(loss.item())

        # Calculate average validation loss over the last few batches
        if i < avg_val_window:
            avg_loss = sum(loss_values) / (i + 1)
        else:
            avg_loss = sum(loss_values[-avg_val_window:]) / avg_val_window

        avg_loss_values.append(avg_loss)

        if avg_loss >= prev_loss:  # worse loss if greater MAE
            patience += 1
        else:
            patience = 0
            torch.save(model, model_save_path)
            optimal_model_sample_no = total_training_samples
            prev_loss = avg_loss

        if patience >= patience_lim:
            stopped_decreasing = True
            print("Stopping early due to training loss no longer decreasing.")

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Batch {i} loss: {loss.item()}")

        total_training_samples += batch_size

def main():
    # Paths to data
    data_path = '../../data/GBMap-data-filtered.h5ad'
    lengths_path = "../../model/input-genes-lengths.csv"

    # Load and preprocess data
    train_data = load_data(data_path, lengths_path)

    # Model parameters
    input_size = len(train_data.var)  # Assuming genes are stored in `var`
    h1 = (32, 16)
    dropout = 0.4
    lr = 3e-5
    wd = 1e-5

    # Model definition
    model = GBMPurity(input_size=input_size, h1=h1[0], h2=h1[1], p_dropout=dropout).float()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Train model
    train_model(train_data, model, loss_fn, optimizer)

if __name__ == "__main__":
    main()
