"""
Main execution script for the QCNN package.
"""

if __name__ == "__main__":
    from benchmarking import Benchmarking

    # Configuration for the benchmark run
    Unitaries = ['U_SU4']
    U_num_params = [15]
    Encodings = ['autoencoder8']
    dataset = 'OrganAMNIST'
    classes = [0, 3, 7, 8] 

    # Run the benchmarking
    results = Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings)