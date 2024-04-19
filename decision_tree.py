"""
CMPUT 466/566 - Machine Learning, Winter 2024, Assignment 1
B. Chan

TODO: You will need to implement the following functions:
- entropy(y, num_categories): ndarray, int -> float
- optimal_split(X, y, H_data, split_dim, num_classes, debug): ndarray, ndarray, float, int, int, bool -> (float, float)

Implementation description will be provided under each function.

For the following:
- N: Number of samples.
- D: Dimension of input features.
- C: Number of classes (labels). We assume the class starts from 0.

Use Node to represent the decision tree, built using the train function.
The root of the tree is at level = 0.
You can access its child(ren) using node.left and/or node.right.
If the node is a leaf node, the is_leaf flag is set to True.
"""


import numpy as np


class Node:
    def __init__(self,
                 num_classes,
                 split_dim=None,
                 split_value=None,
                 left=None,
                 right=None,
                 is_leaf=False,
                 probs=0.):
        """ This class corresponds to a node for the Decision Tree classifier.
        
        Args:
        - split_dim (int): The split dimension of the input features.
        - split_value (float): The value used to determine the left and right splits.
        - left (Node): The left sub-tree.
        - right (Node): The right sub-tree.
        - is_leaf (bool): Whether the node is a leaf node.
        - probs (ndarray (shape: (C, 1))): The C-column vector consisting the probabilities of classifying each class.
        """
        assert num_classes > 1, "num_classes must be at least 2, got: {}".format(num_classes)

        self.num_classes = num_classes
        self.is_leaf = is_leaf
        if self.is_leaf:
            assert len(probs.shape) == 2 and probs.shape[1] == 1, f"probs needs to be a column vector. Got: {probs.shape}"
            self.probs = probs
        else:
            self.split_dim = split_dim
            self.split_value = split_value
            self.left = left
            self.right = right


def entropy(samples, num_categories):
    """ This function computes the entropy of a categorical distribution given samples.
        
    Args:
    - samples (ndarray (shape: (N, 1))): A N-column vector consisting N samples.
    - num_categories (int): The number of categories. Note: 2 <= num_categories

    Output:
    - ent (float): The ent of a categorical distribution given samples.
    """
    # Get the number of data points per class.
    #TO TEST THE FUNCTION
    (counts, _) = np.histogram(samples,
                               bins=np.arange(num_categories + 1))
    if len(samples) == 0:
        return 0
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    ent = -np.sum(probabilities * np.log2(probabilities))
    return ent


    # ====================================================
    # TODO: Implement your solution within the box
    # Set the entropy of the unnormalized categorical distribution counts
    # Make sure the case where p_i = 0 is handeled appropriately.

    # ====================================================

    


def optimal_split(X, y, H_data, split_dim, num_classes, debug=False):
    """ This function finds the optimal split over a random split dimension.

    Args:
    - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
    - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
    - H_data (float): The entropy of the data before the split.
    - split_dim (int): The dimension to find split on.
    - num_classes (int): The number of class labels. Note: 2 <= num_classes.
    - debug (bool): Whether or not to print debug messages
    
    Outputs:
    - split_value (float): The value used to determine the left and right splits.
    - maximum_information_gain (float): The maximum information gain from all possible choices of a split value.
    """
    (N, D) = X.shape

    # Sort data based on column at split dimension
    sort_idx = np.argsort(X[:, split_dim])
    X = X[sort_idx]
    y = y[sort_idx]

    # This returns the unique values and their first indicies.
    # Since X is already sorted, we can split by looking at first_idxes.
    (unique_values, first_idxes) = np.unique(X[:, split_dim], return_index=True)
    current_split_index = None
    current_split_value = None
    H_left = None
    H_right = None
    current_information_gain = None

    # Initialize variables
    maximum_information_gain = 0
    split_value =unique_values[0] - 1  

    for i in range(1, len(unique_values)):
        current_value = unique_values[i]
        current_split_value = (unique_values[i] + unique_values[i - 1]) / 2 
        # Split y into two parts based on the split value
        y_left = y[:first_idxes[i]]
        y_right = y[first_idxes[i]:]
        
        # Calculate entropies
        H_left = entropy(y_left, num_classes)
        H_right = entropy(y_right, num_classes)
        
        # Calculate the weights for left and right nodes
        weight_left = len(y_left) / N
        weight_right = len(y_right) / N
        
        # Calculate the information gain
        current_information_gain = H_data - (weight_left * H_left + weight_right * H_right)
        
        # Update the best split if the current split is better
        if current_information_gain > maximum_information_gain:
            maximum_information_gain = current_information_gain
            split_value = current_split_value

        if debug:
            print(f"Split at {current_value} gives information gain {current_information_gain}")

          

    return split_value, maximum_information_gain


def build_leaf(y, H_data, level, num_classes, debug=False, debug_message=""):
    """ This function builds a leaf node.
        
    Args:
    - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
    - H_data (float): The entropy of the y.
    - level (int): The current level (depth) of the tree. NOTE: 0 <= level.
    - num_classes (int): The number of class labels. Note: 2 <= num_classes.
    - debug (bool): Whether or not to print debug messages
    - debug_message (str): The message indicates why build leaf is called.

    Output:
    - current_node (Node): The leaf node.
    """
    
    # Count the number of labels per class and compute the probabilities.
    # counts: (D,)
    # NOTE: The + 1 is required since for last class.
    (counts, _) = np.histogram(y, bins=np.arange(num_classes + 1))
    probs = counts[:, None] / len(y)
    current_node = Node(num_classes=num_classes, is_leaf=True, probs=probs)

    if debug:
        print("Building leaf node: Num Samples: {}, Entropy: {}, Depth: {}, Probs: {} - {}".format(
            len(y),
            H_data,
            level,
            probs.T,
            debug_message,
        ))

    return current_node


def build_tree(seed,
               X,
               y,
               level,
               max_depth,
               min_leaf_data,
               min_entropy,
               num_split_retries,
               num_classes,
               debug):
    """ This function builds the decision tree from a specified level recursively.
    
    Args:
    - seed (int): The seed to for randomly choosing the splits.
    - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
    - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
    - level (int): The current level (depth) of the tree. NOTE: 0 <= level.
    - max_depth (int): The maximum depth of the decision tree. Note: 0 <= max_depth
    - min_leaf_data (int): The minimum number of data required to split. Note: 1 <= min_leaf_data
    - min_entropy (float): The minimum entropy required to determine a leaf node.
    - num_split_retries (int): The number of retries if the split fails
                                (i.e. split has 0 information gain). Note: 0 <= num_split_retries.
    - num_classes (int): The number of class labels. Note: 2 <= num_classes.
    - debug (bool): Debug mode. This will provide more debugging information.
    
    Output:
    - current_node (Node): The node at the specified level.
    
    NOTE: The Node class is the defined with the following attributes:
    - is_leaf
        - is_leaf == True -> probs
        - is_leaf == False -> split_dim, split_value, left, right
    """
    (N, D) = X.shape
    assert N > 0, "There should be at least one data point."

    rng = np.random.RandomState(seed)
    (left_seed, right_seed) = rng.randint(0, 2 ** 32 - 1, size=(2,))

    H_data = entropy(y, num_classes)

    # Determine whether we have enough data or the data is pure enough for a split
    if N < min_leaf_data or H_data < min_entropy or max_depth <= level:
        return build_leaf(y, H_data, level, num_classes, debug, "Hyperparameter constraints")

    # Find the optimal split. Repeat if information gain is 0.
    # Got to try at least once.
    best_information_gain = -1.0
    best_split_dim = None
    best_split_value = None

    split_dims = rng.randint(0, D, num_split_retries + 1)
    for split_dim in split_dims:
        split_value, maximum_information_gain = optimal_split(X,
                                                              y,
                                                              H_data,
                                                              split_dim,
                                                              num_classes,
                                                              debug)
        assert maximum_information_gain >= 0, f"Information gain must be non-negative. Got: {maximum_information_gain}"

        if maximum_information_gain > best_information_gain:
            best_information_gain = maximum_information_gain
            best_split_dim = split_dim
            best_split_value = split_value

    # Find indicies for left and right splits
    left_split = X[:, best_split_dim] <= best_split_value
    right_split = X[:, best_split_dim] > best_split_value
    num_left = left_split.sum()
    num_right = right_split.sum()
    assert num_left + num_right == N, f"The sum of splits ({num_left + num_right}) should add up to number of samples ({N})"

    if num_left == 0 or num_right == 0 or best_information_gain == 0.0:
        return build_leaf(y, H_data, level, num_classes, debug, "No split dimension with positive information gain")

    if debug:
        print("Creating new level: Information gain: {}, Split Dimension: {}, Split Sizes: ({}, {}), Depth: {}".format(
            maximum_information_gain,
            best_split_dim,
            num_left,
            num_right,
            level,
        ))

    # Build left and right sub-trees
    left_child = build_tree(left_seed,
                            X[left_split],
                            y[left_split],
                            level + 1,
                            max_depth,
                            min_leaf_data,
                            min_entropy,
                            num_split_retries,
                            num_classes,
                            debug)
    right_child = build_tree(right_seed,
                             X[right_split],
                             y[right_split],
                             level + 1,
                             max_depth,
                             min_leaf_data,
                             min_entropy,
                             num_split_retries,
                             num_classes,
                             debug)

    current_node = Node(num_classes=num_classes,
                        split_dim=best_split_dim,
                        split_value=best_split_value,
                        left=left_child,
                        right=right_child,
                        is_leaf=False)
    return current_node


def train_dt(train_X,
             train_y,
             seed,
             max_depth,
             min_leaf_data,
             min_entropy,
             num_split_retries,
             num_classes,
             debug):
    """ Builds the decision tree from root level.
    
    Args:
    - train_X (ndarray (shape: (N, D))): NxD matrix storing N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): Column vector with N scalar training outputs (labels).
    - seed (int): The seed to for randomly choosing the splits.
    - max_depth (int): The maximum depth of the decision tree. Note: 0 <= max_depth
    - min_leaf_data (int): The minimum number of data required to split. Note: 1 <= min_leaf_data
    - min_entropy (float): The minimum entropy required to determine a leaf node.
    - num_split_retries (int): The number of retries if the split fails
                                (i.e. split has 0 information gain). Note: 0 <= num_split_retries.
    - num_classes (int): The number of class labels. Note: 2 <= num_classes.
    - debug (bool): Debug mode. This will provide more debugging information.

    Output:
    - tree (Node): The root node of the decision tree.
    """
    assert len(train_X.shape) == 2, f"train_X should be a matrix. Got: {train_X.shape} tensor."
    assert train_X.shape[0] == train_y.shape[0], f"X and y should have same number of data (train_X: {train_X.shape[0]}, train_y: {train_y.shape[0]})."
    assert train_y.shape[1] == 1, f"train_y should be a column-vector. Got: {train_y.shape}."

    tree = build_tree(seed,
                      train_X,
                      train_y,
                      0,
                      max_depth,
                      min_leaf_data,
                      min_entropy,
                      num_split_retries,
                      num_classes,
                      debug)
    assert isinstance(tree, Node)

    return tree


def predict_dt(node, X):
    """ This function predicts the probability of labels given X from a specified node recursively.
        
    Args:
    - node (Node): The starting node to determine the probability of labels.
    - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
    
    Output:
    - probs_data (ndarray (shape: (N, C))): A NxC matrix consisting N C-dimensional probabilities for each input.
    """
    assert len(X.shape) == 2, f"X should be a matrix. Got: {X.shape} tensor."
    (N, D) = X.shape
    if N == 0:
        return np.empty(shape=(0, node.num_classes))

    if node.is_leaf:
        # node.probs is shape (C, 1)
        return np.repeat(node.probs.T, repeats=N, axis=0)

    left_split = X[:, node.split_dim] <= node.split_value
    right_split = X[:, node.split_dim] > node.split_value
    num_left = left_split.sum()
    num_right = right_split.sum()
    assert num_left + num_right == N, f"The sum of splits ({num_left + num_right}) should add up to number of samples ({N})"

    # Compute the probabilities following the left and right sub-trees
    probs_left = predict_dt(node.left, X[left_split])
    probs_right = predict_dt(node.right, X[right_split])

    # Combine the probabilities returned from left and right sub-trees
    probs_data = np.zeros(shape=(N, node.num_classes))
    probs_data[left_split] = probs_left
    probs_data[right_split] = probs_right
    return probs_data
