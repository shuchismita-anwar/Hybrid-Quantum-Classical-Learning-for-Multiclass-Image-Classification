if __name__ == "__main__":
    from benchmarking import Benchmarking

    # ========== MAIN CONFIGURATION SECTION ==========
    # MODIFY THESE PARAMETERS TO CUSTOMIZE YOUR EXPERIMENT
    
    # Quantum circuit configuration
    Unitaries = ['U_SU4']  # Available: ['U_SU4'] - quantum unitary ansatz
    U_num_params = [15]    # Number of parameters for U_SU4 (don't change unless modifying circuit)
    
    # Data encoding methods - MODIFY HERE to change feature reduction/encoding
    # Available options:
    # - 'resize256': Simple resize to 256 features
    # - 'pca8', 'pca12', 'pca16', 'pca30', 'pca32': PCA with different dimensions
    # - 'autoencoder8', 'autoencoder12', 'autoencoder16', 'autoencoder30', 'autoencoder32'
    Encodings = ['autoencoder8']
    
    # Dataset configuration - MODIFY HERE to change dataset and target classes
    dataset = 'OrganAMNIST'  # Options: 'fashion_mnist', 'mnist', 'OrganAMNIST'
    classes = [0, 3, 7, 8]   # Class indices to include in training - MODIFY HERE for different classes
    
    # NOTE: To change taken/lost qubits, modify quantum_circuits.py in the QCNN function
    # NOTE: To change training parameters (steps, batch_size, learning_rate), modify config.py
    
    # Run the benchmarking
    results = Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings)