"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    float_arr = np.asarray(x,dtype= float)
    mean = np.mean(float_arr,axis=0)
    std = np.std(float_arr,axis=0)
    safe_std = np.where(std ==0,1.0,std)

    normaliztion = (float_arr-mean)/safe_std
    return normaliztion

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    w = np.zeros(n_features,dtype= float)
    b=0.0
    dic ={'w':w,'b':b}
    return dic

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    w =params['w']
    b=params['b']
    scores = x @ w + b
    return scores

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    scores = np.asarray(scores)
    predictions = np.where(scores>0,1,-1)
    return predictions

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    loss = max(0.0,1.0-score * y)
    #hingelosss
    return loss

# Step 6 - svm_objective (not yet solved)
# TODO: implement

# Step 7 - compute_gradients (not yet solved)
# TODO: implement

# Step 8 - apply_update (not yet solved)
# TODO: implement

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

