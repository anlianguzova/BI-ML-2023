import numpy as np
import pandas as pd


def binary_classification_metrics(y_pred: np.array, y_true: np.array):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    confusion_matrix = pd.crosstab(y_true, y_pred)

    tp = confusion_matrix.iloc[0, 0]
    fp = confusion_matrix.iloc[0, 1]
    fn = confusion_matrix.iloc[1, 0]
    tn = confusion_matrix.iloc[1, 1]

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + fn + fn + tn) if (tp + fn + fn + tn) != 0 else 0

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred: np.array, y_true: np.array):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    confusion_matrix = pd.crosstab(y_true, y_pred)
    trues = np.diag(confusion_matrix)
    all_values = len(y_pred)

    return sum(trues)/all_values




def r_squared(y_pred: np.array, y_true: np.array) -> int:
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    sst = np.sum(np.square(y_pred - y_true))
    ssr = np.sum(np.square(y_true - np.mean(y_pred)))

    return 1 - (sst / ssr)


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return 1/len(y_pred) * np.sum((np.square(y_true - y_pred)))


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return 1/len(y_pred) * sum(abs(y_true - y_pred))
