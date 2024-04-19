"""
CMPUT 466/566 - Machine Learning, Winter 2024, Assignment 1
B. Chan

TODO: You will need to implement the following functions:
- train_nb: ndarray, ndarray, int -> Params
- predict_nb: Params, ndarray -> ndarray

Implementation description will be provided under each function.

For the following:
- N: Number of samples.
- D: Dimension of input features.
- C: Number of classes (labels). We assume the class starts from 0.
"""

import numpy as np


class Params:
    def __init__(self, means, covariances, priors, num_features, num_classes):
        """ This class represents the parameters of the Naive Bayes model,
            where the generative model is modeled as a Gaussian.
        NOTE: We assume lables are 0 to K - 1, where K is number of classes.

        We have three parameters to keep track of:
        - self.means (ndarray (shape: (K, D))): Mean for each of K Gaussian likelihoods.
        - self.covariances (ndarray (shape: (K, D, D))): Covariance for each of K Gaussian likelihoods.
        - self.priors (shape: (K, 1))): Prior probabilty of drawing samples from each of K class.

        Args:
        - num_features (int): The number of features in the input vector
        - num_classes (int): The number of classes in the task.
        """

        self.D = num_features
        self.C = num_classes

        # Shape: K x D
        self.means = means

        # Shape: K x D x D
        self.covariances = covariances

        # Shape: K x 1
        self.priors = priors

        assert self.means.shape == (self.C, self.D), f"means shape mismatch. Expected: {(self.C, self.D)}. Got: {self.means.shape}"
        assert self.covariances.shape == (self.C, self.D, self.D), f"covariances shape mismatch. Expected: {(self.C, self.D, self.D)}. Got: {self.covariances.shape}"
        assert self.priors.shape == (self.C, 1), f"priors shape mismatch. Expected: {(self.C, 1)}. Got: {self.priors.shape}"


def train_nb(train_X, train_y, num_classes, **kwargs):
    """ This trains the parameters of the NB model, given training data.

    Args:
    - train_X (ndarray (shape: (N, D))): NxD matrix storing N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): Column vector with N scalar training outputs (labels).

    Output:
    - params (Params): The parameters of the NB model.
    """
    assert len(train_X.shape) == len(train_y.shape) == 2, f"Input/output pairs must be 2D-arrays. train_X: {train_X.shape}, train_y: {train_y.shape}"
    (N, D) = train_X.shape
    assert train_y.shape[1] == 1, f"train_Y must be a column vector. Got: {train_y.shape}"

    # Shape: C x D
    means = np.zeros((num_classes, D))

    # Shape: C x D x D
    covariances = np.tile(np.eye(D), reps = (num_classes, 1, 1))

    # Shape: C x 1
    priors = np.ones(shape=(num_classes, 1)) / num_classes

    # ====================================================
    for c in range(num_classes):
        # Select the data points of class c
        class_indices = train_y == c
        X_c = train_X[class_indices.ravel()]

        # Compute the mean and variance for each feature for class c
        means[c, :] = np.mean(X_c, axis=0)
        # Add a small value to variance to avoid division by zero
        covariances[c, :] = np.diag((np.var(X_c, axis=0)))

        # Compute the prior probability for class c
        priors[c] = float(np.sum(class_indices)) / N

    # Convert variances to a diagonal covariance matrix
    # ====================================================

    params = Params(means, covariances, priors, D, num_classes)
    return params


def predict_nb(params, X):
    """ This function predicts the probability of labels given X.

    Args:
    - params (Params): The parameters of the NB model.
    - X (ndarray (shape: (N, D))): NxD matrix with N D-dimensional inputs.

    Output:
    - probs (ndarray (shape: (N, K))): NxK matrix storing N K-vectors (i.e. the K class probabilities)
    """
    assert len(X.shape) == 2, f"Input/output pairs must be 2D-arrays. X: {X.shape}"
    (N, D) = X.shape
    C = params.C
    probs = np.zeros((N, params.C))
    unnormalized_probs = np.zeros((N, params.C))
    # ====================================================
    for c in range(params.C):
        mean = params.means[c]
        # Extract the diagonal elements of the covariance matrix as variances
        variance = np.diag(params.covariances[c]).copy()  # Use .copy() to ensure a writable array
        # Replace zeros with a small number to avoid division by zero
        variance[variance == 0] = 1e-10
        # No need for Cholesky decomposition since we're using diagonal covariance matrices
        # Calculate the log probability density function of the Gaussian distribution
        log_likelihood = -0.5 * np.sum(((X - mean) ** 2) / variance, axis=1)
        log_likelihood -= 0.5 * np.sum(np.log(2. * np.pi * variance))
        log_likelihood += np.log(params.priors[c])

        # Store unnormalized log probabilities
        unnormalized_probs[:, c] = log_likelihood

    # Normalize the probabilities so that they sum to 1 for each sample
    max_log_prob = np.max(unnormalized_probs, axis=1, keepdims=True)
    probs = np.exp(unnormalized_probs - max_log_prob)
    probs /= np.sum(probs, axis=1, keepdims=True)
    
    # ====================================================

    return probs
