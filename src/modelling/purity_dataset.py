"""
purity_dataset.py

This module defines the SingleCellData class for handling the single-cell RNA sequencing data used to train GBMPurity.
It includes functionalities for filtering samples with low cell counts, normalizing gene expression data,
and simulating bulk RNA-seq data from single-cell data. The class is designed to be used with PyTorch's 
DataLoader for training machine learning models.

Classes:
    SingleCellData: A dataset class for single-cell RNA sequencing data.

Functions:
    tpm(counts, lengths): Static method to perform TPM normalization.
    simulate_bulk(self, n_cells, split, idx): Simulate bulk RNA-seq data from single-cell data.
    __len__(self): Return the length of the dataset.
    __getitem__(self, idx): Get a sample from the dataset.
    filter_low_cells(self, adata): Filter samples with low cell counts.
"""

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SingleCellData(Dataset):
    def __init__(self, raw_adata):
        """
        Initialize the SingleCellData dataset.

        Parameters:
            raw_adata (AnnData): Raw single-cell data.
        """
        self.adata = self.filter_low_cells(raw_adata)
        self.X = self.adata.X
        self.labels = np.array(self.adata.obs['cell_label'])
        self.genes = self.adata.var_names
        self.tumor_idx = np.where(self.labels == 'Tumor')[0]
        self.normal_idx = np.where(self.labels == 'Normal')[0]
        
        self.gene_lengths = pd.read_csv("../../model/input-genes-lengths.csv")
        if np.array_equal(self.gene_lengths['feature_name'].values, self.adata.var.index.values):
            self.gene_lengths = np.array(self.gene_lengths['feature_length'])
        
        self.sampleCells = []
        for sample in self.adata.obs['sample'].unique():
            sample_idx = np.where(self.adata.obs['sample'] == sample)[0]
            tumor_intersect = np.intersect1d(self.tumor_idx, sample_idx)
            normal_intersect = np.intersect1d(self.normal_idx, sample_idx)
            self.sampleCells.append([tumor_intersect, normal_intersect])
        
        self.length = len(self.adata.obs['sample'].unique())

    @staticmethod
    def tpm(counts, lengths):
        """
        Transcripts per million (TPM) normalization.

        Parameters:
            counts (np.array): Gene expression counts.
            lengths (np.array): Gene lengths.

        Returns:
            np.array: TPM normalized gene expression.
        """
        X = counts / lengths
        X = X / X.sum(axis=0) * 1e4  # axis=0 for 1D array, axis=1 for 2D array
        return X

    def simulate_bulk(self, n_cells, split, idx):
        """
        Simulate bulk RNA-seq data from single-cell data.

        Parameters:
            n_cells (int): Number of cells to simulate bulk data from.
            split (float): Ratio of tumor cells in the simulated bulk data.
            idx (int): Index for sampleCells to select cells from.

        Returns:
            tuple: Simulated bulk RNA-seq data and purity.
        """
        tumor_count = int(n_cells * split)
        normal_count = n_cells - tumor_count
        purity = tumor_count / (tumor_count + normal_count)
        cells = np.concatenate([
            np.random.choice(self.sampleCells[idx][0], tumor_count),
            np.random.choice(self.sampleCells[idx][1], normal_count)
        ])
        bulk = self.X[cells].sum(axis=0).A1
        bulk = np.log2(self.tpm(bulk, self.gene_lengths) + 1)
        return bulk, purity

    def __len__(self):
        """
        Length of the dataset.

        Returns:
            int: An arbitrarily large number as we simulate samples.
        """
        return int(1e10)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
            idx (int): Index for the sample. Not used as samples are randomly generated.

        Returns:
            tuple: Pseudobulk RNA-seq data and purity.
        """
        pseudobulk, purity = self.simulate_bulk(
            n_cells=np.random.randint(200, 4001),
            split=np.random.random(),
            idx=np.random.randint(0, self.length)
        )
        return pseudobulk, purity
    
    def filter_low_cells(self, adata):
        """
        Filter samples with low cell counts.

        Parameters:
            adata (AnnData): Annotated data matrix.

        Returns:
            AnnData: Filtered data.
        """
        sample_counts = adata.obs.groupby('sample', observed=False)['cell_label'].value_counts()
        limit = 5
        low_cell_samples = list(sample_counts[sample_counts < limit].index.get_level_values('sample').unique())
        print(f"{len(low_cell_samples)} Samples with < {limit} of Tumor or Normal cell count won't be included:\n{low_cell_samples}")
        sample_data = adata[~adata.obs['sample'].isin(low_cell_samples)]
        return sample_data
