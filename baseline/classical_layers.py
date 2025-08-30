import numpy as np


def scaled_output(probabilities):
    probabilities = np.array(probabilities)  # Convert to numpy array
    #print(f"Scaled Output Dimensions: {probabilities.shape}")
    return (probabilities * 4) - 2  # Scale to range -2 to 2

# %%
class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / (input_dim + output_dim))  # Xavier initialization
        self.bias = np.zeros(output_dim)
        #print(f"Initialized Weights Dimensions: {self.weights.shape}")
        #print(f"Initialized Bias Dimensions: {self.bias.shape}")
    
    def forward(self, inputs):
        #print(f"Input Dimensions: {inputs.shape}")
        z = np.dot(self.weights, inputs) + self.bias
        #print(f"Z (Weighted Sum) Dimensions: {z.shape}")
        output = np.tanh(z)
        #print(f"Output Dimensions: {output.shape}")
        return output
    
    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.bias])

    def set_params(self, params):
        weight_size = self.weights.size
        self.weights = params[:weight_size].reshape(self.weights.shape)
        self.bias = params[weight_size:weight_size + self.bias.size]


# class HiddenLayer:
#     def __init__(self, input_dim, hidden_dim):
#         self.weights = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
#         self.bias = np.zeros(hidden_dim)
#         #print(f"Initialized Hidden Layer Weights Dimensions: {self.weights.shape}")
#         #print(f"Initialized Hidden Layer Bias Dimensions: {self.bias.shape}")
    
#     def forward(self, inputs):
#         #print(f"Hidden Layer Input Dimensions: {inputs.shape}")
#         z = np.dot(self.weights, inputs) + self.bias
#         #print(f"Hidden Layer Output Dimensions: {z.shape}")
#         return np.tanh(z)
    
#     def get_params(self):
#         return np.concatenate([self.weights.flatten(), self.bias])

#     def set_params(self, params):
#         weight_size = self.weights.size
#         self.weights = params[:weight_size].reshape(self.weights.shape)
#         self.bias = params[weight_size:weight_size + self.bias.size]


# class FinalLayer:
#     def __init__(self, hidden_dim, output_dim):
#         self.weights = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
#         self.bias = np.zeros(output_dim)
#         #print(f"Initialized Final Layer Weights Dimensions: {self.weights.shape}")
#         #print(f"Initialized Final Layer Bias Dimensions: {self.bias.shape}")
    
#     def forward(self, inputs):
#         #print(f"Final Layer Input Dimensions: {inputs.shape}")
#         z = np.dot(self.weights, inputs) + self.bias
#         #print(f"Final Layer Z (Weighted Sum) Dimensions: {z.shape}")
#         output = np.tanh(z)
#         #print(f"Final Layer Output Dimensions: {output.shape}")
#         return output
    
#     def get_params(self):
#         return np.concatenate([self.weights.flatten(), self.bias])

#     def set_params(self, params):
#         weight_size = self.weights.size
#         self.weights = params[:weight_size].reshape(self.weights.shape)
#         self.bias = params[weight_size:weight_size + self.bias.size]





# def dot_product(neurons_taken, neurons_lost):
#     # print(f"Dot Product Input Neurons Taken Dimensions: {neurons_taken.shape}")
#     # print(f"Dot Product Input Neurons Lost Dimensions: {neurons_lost.shape}")
#     result = np.multiply(neurons_taken, neurons_lost)
#     # print(f"Dot Product Output Dimensions: {result.shape}")
#     return result




def process_neurons(probabilities_taken, classical_params):
    # Initialize the fully connected layers
    fc_layer_taken = FullyConnectedLayer(input_dim=4, output_dim=4)
    # multi_hidden_layer_lost = MultiHiddenLayer(input_dim=4, hidden_dim=64, num_layers=num_hidden_layers)  # Updated line
    # final_layer_lost = FinalLayer(hidden_dim=64, output_dim=6)

    # Set the classical parameters
    fc_layer_taken.set_params(classical_params[:fc_layer_taken.get_params().size])
    # start_hidden = fc_layer_taken.get_params().size
    # end_hidden = start_hidden + multi_hidden_layer_lost.get_params().size
    # multi_hidden_layer_lost.set_params(classical_params[start_hidden:end_hidden])
    # final_layer_lost.set_params(classical_params[end_hidden:])

    # Scale the probabilities
    scaled_values_taken = scaled_output(probabilities_taken)
    # scaled_values_lost = scaled_output(probabilities_lost)

    # Process the scaled values through their respective layers
    output_neurons_taken = fc_layer_taken.forward(scaled_values_taken)
    # hidden_output = multi_hidden_layer_lost.forward(scaled_values_lost)
    # output_neurons_lost = final_layer_lost.forward(hidden_output)

    # Combine the outputs from the two sets of neurons
    # combined_output = dot_product(output_neurons_taken, output_neurons_lost)
    combined_output =  output_neurons_taken

    return combined_output