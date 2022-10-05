import numpy as np

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.power(W, 2))
    grad = reg_strength * 2 * W

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    return loss, grad

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    true_probs = np.zeros(probs.shape)
    if probs.shape == (len(probs), ):
        true_probs[target_index] = 1
        value = - np.sum(true_probs * np.log(probs))
    else:
        # print(probs)
        np.put_along_axis(true_probs, target_index.reshape(-1,1), 1, axis=1)
        value = - np.sum(true_probs * np.log(probs))

        # print("----")
        # print(true_probs)
        # print("---")
        # value = value.mean()
    return value

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    # print(f"Shape of predictions {predictions.shape} and it was \n {predictions}")
    pred_copy = predictions.copy()
    pred_copy -= np.max(pred_copy)
    if len(predictions.shape) == 1:

        result = np.exp(pred_copy)/ np.sum(np.exp(pred_copy))
    else:
        pred_copy -= np.max(pred_copy, axis=1, keepdims=True)
        sum_exp = np.sum(np.exp(pred_copy), axis=1, keepdims=True)
        result = np.exp(pred_copy)/ sum_exp

    return result


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    # TODO implement softmax with cross-entropy

    preds = predictions.copy()
    probs = softmax(preds)
    loss_value = cross_entropy_loss(probs, target_index)
    mask = np.zeros_like(predictions)

    # mask and dprediction for (N) shape predictions
    if predictions.shape == (len(predictions), ):
        mask[target_index] = 1
        dprediction = - (mask - probs)

    # mask and dprediction for (batch_size, N) shape predictions
    else:
        np.put_along_axis(mask, target_index.reshape(-1,1), 1, axis=1)

        # mask[np.arange(len(mask)), target_index] = 1
        dprediction = - (mask - probs)
    # print(f"gradient {dprediction}")

    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    return loss_value, dprediction

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.mask = np.zeros_like(X)
        self.mask[X > 0] = 1

        return np.maximum(X, np.zeros_like(X))

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.zeros_like(d_out)
        d_result[self.mask > 0] = 1

        return d_result * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        return np.dot(X, self.W.value) + self.B.value


    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        X = self.X.value
        W = self.W.value
        self.W.grad = np.dot(X.T, d_out)
        self.B.grad = np.dot(np.ones(shape=(1, X.shape[0])), d_out)
        d_input = np.dot(d_out, W.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
