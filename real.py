"""
Module which has all the Real functions and classes.
This may be split laters into different modules for:
    Activations
    Losses
"""
from tqdm import tqdm

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



def back_propogation(
    info
):
    """
    Function to perform back propogation on real numbers.
    Input is a dictionary of metadata of the neural network.
        Output is also a neural network.

    ::param info: (dict)
    ::output: (dict)
    """