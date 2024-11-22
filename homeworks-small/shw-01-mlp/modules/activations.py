import numpy as np
from .base import Module

from scipy.special import softmax, expit, log_softmax

class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """

        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        
        return (input > 0).astype(int) * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """

        self.expit = expit(input)

        return self.expit

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """

        return grad_output * self.expit * (1 - self.expit)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """

        self.softmax = softmax(input, axis=1)

        return self.softmax

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """

        grad_input = np.zeros_like(grad_output)

        # for i in range(input.shape[0]):
        #     softmax_for_x = self.softmax[i]
        #     jacobian = np.diagflat(softmax_for_x) - np.outer(softmax_for_x, softmax_for_x)
        #     grad_input[i] = grad_output[i] @ jacobian

        grad_input = grad_output * self.softmax  - self.softmax \
              * np.sum(grad_output * self.softmax , axis=1, keepdims=True)

        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """

        return log_softmax(input, axis=1)
    
    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """

        """"
        J = 1 - softmax(i) if i = j else - softmax(i)

        Let x be a vector row, then:

        dL/dx = x @ J.T

        dL/dX = X @ (Diag - vstack(softmax))

        dL/dX = X - row_sum(X) * softmax
        """

        grad_input = grad_output - softmax(input, axis=1) * grad_output.sum(axis=1, keepdims=True)

        return grad_input