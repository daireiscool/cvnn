"""
Module which has all the Complex functions and classes.
This may be split laters into different modules for:
    Activations
    Losses
"""


class BasicActivation():
    """
    A basic activation function.
    """
    
    def __init__(self):
        """
        ::param input: (float) A single value 
        ::output: (float)
        """
        pass

    def activate(self, inputs):
        """
        Activation function.
        
        ::param inputs: (list|array)
        """
        return inputs
    
    def diff(self, inputs):
        """
        Differential of the activation function.
        
        ::param inputs: (list|array)
        """
        return 1



def backpropogation(
    X_train,
    y_train,
    layers,
    activations,
    loss,
    validation_data,
    validation_size,
    alpha,
    early_stopping,
    epoch,
    n_iter,
):
    """
    Function to perform back propogation on real numbers.

    ::param X_train: (pandas dataframe|ndarray)
    ::param y_train: (pandas series|array)
    ::param layers: (list[array])
    ::param activations: (list[class])
    ::param loss: (function)
    ::param validation_data: (list[ndarray, array]) Data to validate the model on
    ::param validation_size: (float) Splits the training data if > 0 and validation_data is empty
    ::param alpha: (float) Learning step size
    ::param early_stopping: (float) Stops training if difference is less than value
    ::param epoch: (int) Epoch size for training
    ::param n_iter: (int) Max number of training iterations
    
    ::output: (list[array])
    """