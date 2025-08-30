import numpy as np


def scaled_output(probabilities):
    probabilities = np.array(probabilities)  # Convert to numpy array
    return (probabilities * 4) - 2  # Scale to range -2 to 2


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / (input_dim + output_dim))  # Xavier initialization
        self.bias = np.zeros(output_dim)
    
    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        output = np.tanh(z)
        return output
    
    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.bias])

    def set_params(self, params):
        weight_size = self.weights.size
        self.weights = params[:weight_size].reshape(self.weights.shape)
        self.bias = params[weight_size:weight_size + self.bias.size]


class HiddenLayer:
    def __init__(self, input_dim, hidden_dim):
        self.weights = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.bias = np.zeros(hidden_dim)
    
    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return np.tanh(z)
    
    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.bias])

    def set_params(self, params):
        weight_size = self.weights.size
        self.weights = params[:weight_size].reshape(self.weights.shape)
        self.bias = params[weight_size:weight_size + self.bias.size]


class FinalLayer:
    def __init__(self, hidden_dim, output_dim):
        self.weights = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.bias = np.zeros(output_dim)
    
    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        output = np.tanh(z)
        return output
    
    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.bias])

    def set_params(self, params):
        weight_size = self.weights.size
        self.weights = params[:weight_size].reshape(self.weights.shape)
        self.bias = params[weight_size:weight_size + self.bias.size]


def dot_product(neurons_taken, neurons_lost):
    result = np.multiply(neurons_taken, neurons_lost)
    return result


def process_neurons(probabilities_taken, probabilities_lost, classical_params):
    # Initialize the fully connected layers
    fc_layer_taken = FullyConnectedLayer(input_dim=4, output_dim=4)
    hidden_layer_lost = HiddenLayer(input_dim=4, hidden_dim=16)
    final_layer_lost = FinalLayer(hidden_dim=16, output_dim=4)

    # Set the classical parameters
    fc_layer_taken.set_params(classical_params[:fc_layer_taken.get_params().size])
    hidden_layer_lost.set_params(classical_params[fc_layer_taken.get_params().size:fc_layer_taken.get_params().size + hidden_layer_lost.get_params().size])
    final_layer_lost.set_params(classical_params[-final_layer_lost.get_params().size:])

    # Scale the probabilities
    scaled_values_taken = scaled_output(probabilities_taken)
    scaled_values_lost = scaled_output(probabilities_lost)

    # Process the scaled values through their respective layers
    output_neurons_taken = fc_layer_taken.forward(scaled_values_taken)
    hidden_output = hidden_layer_lost.forward(scaled_values_lost)
    output_neurons_lost = final_layer_lost.forward(hidden_output)

    # Combine the outputs from the two sets of neurons
    combined_output = dot_product(output_neurons_taken, output_neurons_lost)

    return combined_output