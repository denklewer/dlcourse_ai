import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(np.power(W, 2))
    grad = reg_strength * 2 * W

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
        np.put_along_axis(true_probs, target_index.reshape(-1,1), 1, axis=1)
        value = - np.sum(true_probs * np.log(probs))
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
    '''
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
    '''

    preds = predictions.copy()
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    mask = np.zeros_like(predictions)
    # mask and dprediction for (N) shape predictions
    if predictions.shape == (len(predictions), ):
        mask[target_index] = 1
        dprediction = - (mask - probs)
    # mask and dprediction for (batch_size, N) shape predictions
    else:
        np.put_along_axis(mask, target_index.reshape(-1,1), 1, axis=1)
        dprediction = - (mask - probs)

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.mask = np.zeros_like(X)
        self.mask[X > 0] = 1

        return np.maximum(X, np.zeros_like(X))

    def backward(self, d_out):
        d_result = np.zeros_like(d_out)
        d_result[self.mask > 0] = 1

        return d_result * d_out

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = Param(X)
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        X = self.X.value
        W = self.W.value
        self.W.grad = np.dot(X.T, d_out)
        self.B.grad = np.dot(np.ones(shape=(1, X.shape[0])), d_out)
        d_input = np.dot(d_out, W.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        filter_size = self.filter_size
        padding = self.padding

        padded_X = np.zeros(shape=(batch_size, height + padding * 2, width + padding * 2, channels))
        padded_X[:,padding:height-padding, padding:width-padding, :] = X.copy()


        for y in range(out_height):
            for x in range(out_width):
                
                left_bound_h = y
                right_bound_h = max(height, y+filter_size)

                left_bound_w = x
                right_bound_w = max(height, x+filter_size)

                target_x = X[: , left_bound_h: y+filter_size, x-filter_size:x+filter_size, :]
                print(target_x.shape)

                # TODO: Implement forward pass for specific location
                pass
        raise Exception("Not implemented!")


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                pass

        raise Exception("Not implemented!")

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
