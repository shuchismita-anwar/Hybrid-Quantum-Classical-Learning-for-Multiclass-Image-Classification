import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from data_processing import data_load_and_process
from training import circuit_training, accuracy_test
from quantum_circuits import QCNN
from classical_layers import process_neurons
from visualization import plot_confusion_matrix
from embeddings import Encoding_to_Embedding


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings):
    results = []
    for U, U_params in zip(Unitaries, U_num_params):
        for Encoding in Encodings:
            # Load and process the dataset
            X_train, X_test, Y_train, Y_test = data_load_and_process(dataset, classes=classes, feature_reduction=Encoding)

            print(f"Training {U} with {Encoding} encoding")

            # Train the quantum-classical model, now tracking train and test accuracy history
            loss_history, train_accuracy_history, test_accuracy_history, trained_params = circuit_training(
                X_train, Y_train, X_test, Y_test, U, U_params, Encoding_to_Embedding(Encoding), 'QCNN', 'mse'
            )

            # Plot the loss history


            # Split the trained parameters into quantum and classical
            total_conv_params = U_params * 6  # 6 convolutional layers
            total_pooling_params = 2 * 2  # 2 pooling layers
            total_quantum_params = total_conv_params + total_pooling_params

            quantum_params = trained_params[:total_quantum_params]
            classical_params = trained_params[total_quantum_params:]

            # Calculate final predictions on the train set
            train_predictions = []
            for x in X_train:
                probabilities_taken = QCNN(x, quantum_params, U, U_params, Encoding_to_Embedding(Encoding))
                combined_output = process_neurons(probabilities_taken, classical_params)
                train_predictions.append(combined_output)

            # Calculate final predictions on the test set
            test_predictions = []
            for x in X_test:
                probabilities_taken = QCNN(x, quantum_params, U, U_params, Encoding_to_Embedding(Encoding))
                combined_output = process_neurons(probabilities_taken, classical_params)
                test_predictions.append(combined_output)

            # Convert predictions to class labels (argmax)
            train_pred_labels = np.argmax(train_predictions, axis=1)
            test_pred_labels = np.argmax(test_predictions, axis=1)

            train_true_labels = np.argmax(Y_train, axis=1)
            test_true_labels = np.argmax(Y_test, axis=1)

            # Compute final accuracies
            train_accuracy = accuracy_test(train_predictions, Y_train)
            test_accuracy = accuracy_test(test_predictions, Y_test)

            # Compute precision, recall, and F1 score for train and test sets
            train_precision = precision_score(train_true_labels, train_pred_labels, average='weighted')
            test_precision = precision_score(test_true_labels, test_pred_labels, average='weighted')

            train_recall = recall_score(train_true_labels, train_pred_labels, average='weighted')
            test_recall = recall_score(test_true_labels, test_pred_labels, average='weighted')

            train_f1 = f1_score(train_true_labels, train_pred_labels, average='weighted')
            test_f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')

            # Print results
            print(f"Train Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}")
            print(f"Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")
            
            # Plot confusion matrix for train set
            plot_confusion_matrix(train_true_labels, train_pred_labels, title="Train Confusion Matrix")

            # Plot confusion matrix for test set
            plot_confusion_matrix(test_true_labels, test_pred_labels, title="Test Confusion Matrix")

            # Append results for logging or further analysis
            results.append({
                'Unitary': U,
                'Encoding': Encoding,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Train Precision': train_precision,
                'Test Precision': test_precision,
                'Train Recall': train_recall,
                'Test Recall': test_recall,
                'Train F1': train_f1,
                'Test F1': test_f1,
                'Loss History': loss_history,
                'Train Accuracy History': train_accuracy_history,
                'Test Accuracy History': test_accuracy_history
            })
    
    return results