import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a
    range of 0 <= y[i] < C

    ///////  N = number of example
    //////   C = class (emotion)

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    nom_first_term = logits
    nom_second_term = np.max(logits, axis=1, keepdims= True)
    nominator = np.exp(nom_first_term - nom_second_term)

    denominator = np.sum(nominator,axis=1, keepdims = True)

    probability_vector = nominator/denominator
   
    smooth_factor = 1e-14

    example_size = logits.shape[0]

    loss = -np.sum(np.log(probability_vector[np.arange(example_size),y]+smooth_factor))/example_size
   


    dlogits = probability_vector.copy()
    dlogits[np.arange(example_size), y] -= 1
    dlogits /= example_size

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
