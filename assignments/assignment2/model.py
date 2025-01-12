import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.f1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.f2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.grad = 0


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        for name, param in params.items():
            param.grad = np.zeros_like(param.grad)

        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        f1_result = self.f1.forward(X)

        relu_result = self.relu.forward(f1_result)
        f2_result = self.f2.forward(relu_result)

        loss, d_preds = softmax_with_cross_entropy(f2_result, y)

        # print(f"loss1 was \n {loss} \n d_preds was \n {d_preds}")



        d_f2 = self.f2.backward(d_preds)
        d_relu = self.relu.backward(d_f2)
        d_f1 = self.f1.backward(d_relu)



        # batch_loss, dW = linear_softmax(batch, self.W, target_index)
        # reg_los, dW_r = l2_regularization(self.W, reg)
        # loss = loss + batch_loss + reg_los
        # step = (dW + dW_r) * learning_rate
        # self.W -= step
        
        # After that, implement l2 regularization on all params

        # Hint: self.params() is useful again!

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {
            "f1.W": self.f1.W,
            "f2.W": self.f2.W,
            "f1.B": self.f1.B,
            "f2.B": self.f2.B
        }

        # TODO Implement aggregating all of the params


        return result
