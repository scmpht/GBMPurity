"""
utils.py

Utility functions for the project, including functions for TPM normalization,
accuracy calculation, and concordance correlation coefficient (CCC) calculation.

Functions:
    bulk_tpm(counts: np.array, lengths: np.array) -> np.array
    accuracy(y: np.array, y_pred: np.array) -> float
    ccc(x: np.array, y: np.array) -> float

Author:
    Morgan Thomas <scmpht@leeds.ac.uk>
    Date: 2024-07-08
"""


def tpm(X: np.ndarray, lengths: np.ndarray) -> np.array:
    """
    Perform TPM normalization on the given counts using the provided gene lengths.

    Args:
        counts (np.array): Array of gene counts.
        lengths (np.array): Array of gene lengths.

    Returns:
        np.array: TPM normalized values.

    Raises:
        ValueError: If TPM values do not sum to 1.
    """
    
    if X.shape[1] != lengths.shape[0]:
        raise ValueError("The number of rows in X must match the length of lengths")
    
    # Calculate RPK (Reads Per Kilobase)
    rpk = np.divide(X, lengths)
    
    # Calculate the scaling factor
    scaling_factor = np.nansum(rpk, axis=1).reshape(-1, 1)
    
    # Calculate TPM
    tpm = (rpk / scaling_factor) * 1e6
    
    return tpm



def accuracy(y: np.array, y_pred: np.array) -> float:
    """
    Calculate accuracy of predictions based on a threshold of 0.5.

    Args:
        y (np.array): True values.
        y_pred (np.array): Predicted values.

    Returns:
        float: Accuracy score.
    """
    return np.mean((y > 0.5) == (y_pred > 0.5))



def ccc(x: np.array, y: np.array) -> float:
    """
    Calculate the Concordance Correlation Coefficient (CCC) between two arrays.

    Args:
        x (np.array): First array.
        y (np.array): Second array.

    Returns:
        float: CCC value.
    """
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    rhoc = 2 * sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)
    return rhoc



