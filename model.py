import numpy as np
import matplotlib.pyplot as plt

input_vector = np.array([3, 1.5], [2, 1], [4, 1.5], [3, 4], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1



neural_network = NeuralNetwork(learning_rate)
neural_network.predict(input_vector)

class NeuralNetwork:
    def __init__(self, learing_rate):
        self.weights = np.array([np.randn(), np.random,randn()])
        self.bias = np.random.randn()
        self.learning_rate = learing_rate

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_deriv(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.__sigmoid(layer_1)
        prediction = layer_2
        return prediction
    
    def __compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.__sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self.__sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )

        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)
    
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range (iterations):
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            derror_dbias, derror_dweights = self.__compute_gradients(input_vector, target)

            self._update_parameters(derror_dbias, derror_dweights)

            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)
            
            return cumulative_errors