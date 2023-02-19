"""
A custom made Neural Network template.
This is used to compare ability for Neural Networks using
    * Real weights
    * Complex weights
    * Hamiltonian wights
    * Other
    
Complex Neural Networks should be better at data with rotations.

@author: Dáire Campbell <daireiscool@gmail.com>

To do:
    1. Build basic model
    2. Create varied layer sizes
    3. Custom inputs such as Activation functions
    4. Have the ability to manually input the weights (for GA initiation)
    5. Back propogations
    6. Change the weight data types (real, imaginary, complex)
"""
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def BasicActivation(input):
    """
    A basic activation function.
    
    ::param input: (float) A single value 
    ::output: (float)
    """
    return input


class NN():
    """
    Class implication of a generic Neural Network whose
        weights can be Real, Complex or Other.
    """
    
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 1,
        activation_functions: list = [],
        layers: list = [8,],
        verbose: bool = True,
        random_seed: int = 0,
        ):
        """
        Initialisation of the NN class.
        
        The inputs are:
        ::param input_size: (int)
        ::param output_size: (int)
        ::param activation_functions: (list[functions]) Activation for each layer
        ::param layers: (list[int]) List of neurons per layer.
        ::param verbose: (bool) To print outputs/logs.
        ::param random_seed: (int) Random seed.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_functions = activation_functions
        self.layers = layers
        self.verbose = verbose
        random.seed(random_seed)
        self.random_seed = random_seed
        
        self.weights = self._initialise_weights()


    def _initialise_weights(self):
        """
        Function to initialise weights for the layers
            with random weights.
        
        ::output: (list[arrays])
        """
        weights = []
        for num, layer in enumerate(self.layers + [self.output_size]):
            if num == 0:
                weights = weights + [
                    np.array([[random.uniform(-10, 10) for val in range(self.input_size+1)]
                              for r in range(layer)])
                ]    
            else:
                weights = weights + [
                    np.array([[random.uniform(-10, 10) for val in range(self.layers[num-1]+1)]
                              for r in range(layer)])
                ]    
        return weights            


    def _log(self, args):
        """
        Simple print of a log function.
        """
        if self.verbose:
            print(*args)


    def fit(
        self,
        X_train,
        y_train,
        loss,
        validation_data: list = (),
        validation_size: float = 0,
        alpha: float= 1e3,
        early_stopping: float = 1e-5,
        epoch: int = 1,
        n_iter: int = 10_000
    ):
        """
        Function to train a Neural Network on inputed data.
        
        ::param X_train: (pandas dataframe|ndarray)
        ::param y_train: (pandas series|array)
        ::param loss: (function)
        ::param validation_data: (list[ndarray, array]) Data to validate the model on
        ::param validation_size: (float) Splits the training data if > 0 and validation_data is empty
        ::param alpha: (float) Learning step size
        ::param early_stopping: (float) Stops training if difference is less than value
        ::param epoch: (int) Epoch size for training
        ::param n_iter: (int) Max number of training iterations
        """
        
        if validation_data | validation_size:
            self.validation = True
            self._log("Using Validation Data")
            if validation_data:
                X_train, y_train = shuffle(X_train, y_train, self.random_seed)
                X_val, y_val = shuffle(
                    validation_data[0], validation_data[1], self.random_seed)
            else:
                X_train, y_train, X_val, y_val = train_test_split(
                    X_train, y_train, test_size=validation_size, random_state=self.random_seed)
        else:
            self.validation = False        
        
        self.X_train = X_train
        self.y_train = y_train
        self.loss = loss
        self.X_val = X_val
        self.y_val = y_val
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.epoch = epoch
        self.n_iter = n_iter

        self.loss_train = np.array([])
        self.loss_val = np.array([])


    def predict(self, X):
        """
        Function to predict based on given weights.

        ::param X: (pandas dataframe|ndarray) Data to evaluate/ predict model against
        """
        y_pred = np.array([])

        for i in X:
            z = i
            for weight in self.weights:
                z = np.append(z, 1)
                z = z*weight
                z = np.sum(z, axis = 1)
            y_pred  = np.append(y_pred, z)
        
        return y_pred


if __name__ == "__main__":
    nn = NN()
    print(nn.weights)
