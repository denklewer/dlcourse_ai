import numpy as np


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
    # Your final implementation shouldn't have any loops
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
    loss = reg_strength * np.sum(np.power(W, 2))
    grad = reg_strength * 2 * W

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dL = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T,dL)



    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            loss = 0
            for batch_ind in batches_indices:
                batch = X[batch_ind]
                target_index = y[batch_ind]
                batch_loss, dW = linear_softmax(batch, self.W, target_index)
                reg_los, dW_r = l2_regularization(self.W, reg)
                loss = loss + batch_loss + reg_los
                step = (dW + dW_r) * learning_rate
                self.W -= step

            loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis=1)


        return y_pred



                
                                                          

            

                

#%%
