from benchmarking import Benchmarking

# Example usage
Unitaries = ['U_SU4']
U_num_params = [15]
Encodings = ['autoencoder8']
dataset = 'OrganAMNIST'
classes = [0, 3, 7, 8] 

# Run benchmarking and plot loss, print train/test accuracy, precision, recall, F1, and confusion matrices
results = Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings)