# %%
"""Training utilities and optimization functions."""

import numpy as np
from pennylane import RMSPropOptimizer
from quantum_circuits import QCNN
from classical_layers import FullyConnectedLayer, process_neurons
from visualization import plot_training_loss_history, plot_accuracy_same_graph
from config import steps, learning_rate, batch_size


def accuracy_test(predictions, labels):
    # print("accuracy_test function called.")
    # print(f"Predictions: {predictions}")
    # print(f"Labels: {labels}")
    # print(f"Predictions Dimensions: {np.array(predictions).shape}")
    # print(f"Labels Dimensions: {np.array(labels).shape}")
    
    correct = 0
    for i, (p, l) in enumerate(zip(predictions, labels)):
        # print(f"Iteration {i}: Prediction: {p}, Label: {l}")
        if np.argmax(p) == np.argmax(l):
            correct += 1
    return correct / len(labels)

def square_loss(labels, predictions):
    labels = np.array(labels)  # Convert labels to numpy array
    predictions = np.array(predictions)  # Convert predictions to numpy array
    loss = np.mean((labels - predictions) ** 2)
    return loss


def cross_entropy(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)
    return -np.sum(labels * np.log(predictions + 1e-9)) / len(labels)


def circuit_training(X_train, Y_train, X_test, Y_test, U, U_params, embedding_type, circuit, cost_fn):
    total_conv_params = U_params * 6  # 6 convolutional layers
    total_pooling_params = 2 * 2  # 2 pooling layers, each with 2 parameters
    total_quantum_params = total_conv_params + total_pooling_params  # Total quantum params = 94

    # Adjust classical parameter initialization
    total_classical_params = (
        FullyConnectedLayer(4, 4).get_params().size 
    )

    # Initialize all parameters
    params = np.random.randn(total_quantum_params + total_classical_params)

    optimizer = RMSPropOptimizer(stepsize=0.01)
    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), batch_size)
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = optimizer.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit), params)
        loss_history.append(cost_new)

        # Calculate train accuracy on the full train set
        if it % 50 == 0 and it != 0:
            train_predictions = []
            for x in X_train:
                probabilities_taken = QCNN(x, params[:total_quantum_params], U, U_params, embedding_type)
                combined_output = process_neurons(probabilities_taken, params[total_quantum_params:])
                train_predictions.append(combined_output)

            train_accuracy = accuracy_test(train_predictions, Y_train)
            train_accuracy_history.append(train_accuracy)

            # Calculate test accuracy on the test set
            test_predictions = []
            for x in X_test:
                probabilities_taken = QCNN(x, params[:total_quantum_params], U, U_params, embedding_type)
                combined_output = process_neurons(probabilities_taken, params[total_quantum_params:])
                test_predictions.append(combined_output)

            test_accuracy = accuracy_test(test_predictions, Y_test)
            test_accuracy_history.append(test_accuracy)

        if it % 10 == 0:
            print(f"Iteration: {it}, Cost: {cost_new}")

    plot_training_loss_history(loss_history, interval=10)

            
    plot_accuracy_same_graph(train_accuracy_history, test_accuracy_history, interval=50)

    return loss_history, train_accuracy_history, test_accuracy_history, params


def cost(params, X_batch, Y_batch, U, U_params, embedding_type, circuit):
    total_conv_params = U_params * 6
    total_pooling_params = 2 * 2
    quantum_params = params[:total_conv_params + total_pooling_params]
    classical_params = params[total_conv_params + total_pooling_params:]

    predictions = []
    for x in X_batch:
        probabilities_taken= QCNN(x, quantum_params, U, U_params, embedding_type)
        combined_output = process_neurons(probabilities_taken, classical_params)
        predictions.append(combined_output)

    return square_loss(Y_batch, predictions)