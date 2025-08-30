"""
Benchmarking and evaluation functions for the QCNN package.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
try:
    from .data_processing import data_load_and_process
    from .training import circuit_training, accuracy_test, square_loss
    from .quantum_circuits import QCNN
    from .classical_layers import process_neurons
    from .visualization import plot_confusion_matrix
    from .config import pca32, autoencoder32, pca30, autoencoder30, pca16, autoencoder16, pca12, autoencoder12
except ImportError:
    from data_processing import data_load_and_process
    from training import circuit_training, accuracy_test, square_loss
    from quantum_circuits import QCNN
    from classical_layers import process_neurons
    from visualization import plot_confusion_matrix
    from config import pca32, autoencoder32, pca30, autoencoder30, pca16, autoencoder16, pca12, autoencoder12


def calculate_metrics(predictions, labels):
    # Convert predictions and labels to the appropriate format
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)

    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    return precision, recall, f1


def Encoding_to_Embedding(Encoding):
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'

    # Amplitude Hybrid Embedding
    # 4 qubit block
    elif Encoding == 'pca32-1':
        Embedding = 'Amplitude-Hybrid4-1'
    elif Encoding == 'autoencoder32-1':
        Embedding = 'Amplitude-Hybrid4-1'

    elif Encoding == 'pca32-2':
        Embedding = 'Amplitude-Hybrid4-2'
    elif Encoding == 'autoencoder32-2':
        Embedding = 'Amplitude-Hybrid4-2'

    elif Encoding == 'pca32-3':
        Embedding = 'Amplitude-Hybrid4-3'
    elif Encoding == 'autoencoder32-3':
        Embedding = 'Amplitude-Hybrid4-3'

    elif Encoding == 'pca32-4':
        Embedding = 'Amplitude-Hybrid4-4'
    elif Encoding == 'autoencoder32-4':
        Embedding = 'Amplitude-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca16-1':
        Embedding = 'Amplitude-Hybrid2-1'
    elif Encoding == 'autoencoder16-1':
        Embedding = 'Amplitude-Hybrid2-1'

    elif Encoding == 'pca16-2':
        Embedding = 'Amplitude-Hybrid2-2'
    elif Encoding == 'autoencoder16-2':
        Embedding = 'Amplitude-Hybrid2-2'

    elif Encoding == 'pca16-3':
        Embedding = 'Amplitude-Hybrid2-3'
    elif Encoding == 'autoencoder16-3':
        Embedding = 'Amplitude-Hybrid2-3'

    elif Encoding == 'pca16-4':
        Embedding = 'Amplitude-Hybrid2-4'
    elif Encoding == 'autoencoder16-4':
        Embedding = 'Amplitude-Hybrid2-4'

    # Angular HybridEmbedding
    # 4 qubit block
    elif Encoding == 'pca30-1':
        Embedding = 'Angular-Hybrid4-1'
    elif Encoding == 'autoencoder30-1':
        Embedding = 'Angular-Hybrid4-1'

    elif Encoding == 'pca30-2':
        Embedding = 'Angular-Hybrid4-2'
    elif Encoding == 'autoencoder30-2':
        Embedding = 'Angular-Hybrid4-2'

    elif Encoding == 'pca30-3':
        Embedding = 'Angular-Hybrid4-3'
    elif Encoding == 'autoencoder30-3':
        Embedding = 'Angular-Hybrid4-3'

    elif Encoding == 'pca30-4':
        Embedding = 'Angular-Hybrid4-4'
    elif Encoding == 'autoencoder30-4':
        Embedding = 'Angular-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca12-1':
        Embedding = 'Angular-Hybrid2-1'
    elif Encoding == 'autoencoder12-1':
        Embedding = 'Angular-Hybrid2-1'

    elif Encoding == 'pca12-2':
        Embedding = 'Angular-Hybrid2-2'
    elif Encoding == 'autoencoder12-2':
        Embedding = 'Angular-Hybrid2-2'

    elif Encoding == 'pca12-3':
        Embedding = 'Angular-Hybrid2-3'
    elif Encoding == 'autoencoder12-3':
        Embedding = 'Angular-Hybrid2-3'

    elif Encoding == 'pca12-4':
        Embedding = 'Angular-Hybrid2-4'
    elif Encoding == 'autoencoder12-4':
        Embedding = 'Angular-Hybrid2-4'

    # Two Gates Compact Encoding
    elif Encoding == 'pca16-compact':
        Embedding = 'Angle-compact'
    elif Encoding == 'autoencoder16-compact':
        Embedding = 'Angle-compact'
    return Embedding


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings):
    for U, U_params in zip(Unitaries, U_num_params):
        for Encoding in Encodings:
            X_train, X_test, Y_train, Y_test = data_load_and_process(dataset, classes=classes, feature_reduction=Encoding)

            print(f"Training {U} with {Encoding} encoding")

            # Unpack all four return values: loss history, train/test accuracy history, and trained parameters
            loss_history, train_accuracy_history, test_accuracy_history, trained_params = circuit_training(
                X_train, Y_train, X_test, Y_test, U, U_params, Encoding_to_Embedding(Encoding), 'QCNN', square_loss
            )

            total_conv_params = U_params * 6  # 6 convolutional layers
            total_pooling_params = 2 * 2  # 2 pooling layers
            total_quantum_params = total_conv_params + total_pooling_params

            quantum_params = trained_params[:total_quantum_params]
            classical_params = trained_params[total_quantum_params:]

            # Calculate Train Metrics
            train_predictions = []
            for x in X_train:
                probabilities_taken, probabilities_lost = QCNN(x, quantum_params, U, U_params, Encoding_to_Embedding(Encoding))
                combined_output = process_neurons(probabilities_taken, probabilities_lost, classical_params)
                train_predictions.append(combined_output)

            train_accuracy = accuracy_test(train_predictions, Y_train)
            train_precision, train_recall, train_f1 = calculate_metrics(train_predictions, Y_train)
            print(f"Train Accuracy: {train_accuracy}")
            print(f"Train Precision: {train_precision}, Train Recall: {train_recall}, Train F1: {train_f1}")

            # Calculate Test Metrics
            test_predictions = []
            for x in X_test:
                probabilities_taken, probabilities_lost = QCNN(x, quantum_params, U, U_params, Encoding_to_Embedding(Encoding))
                combined_output = process_neurons(probabilities_taken, probabilities_lost, classical_params)
                test_predictions.append(combined_output)
                
            plot_confusion_matrix(test_predictions, Y_test)
            test_accuracy = accuracy_test(test_predictions, Y_test)
            test_precision, test_recall, test_f1 = calculate_metrics(test_predictions, Y_test)
            print(f"Test Accuracy: {test_accuracy}")
            print(f"Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1: {test_f1}")