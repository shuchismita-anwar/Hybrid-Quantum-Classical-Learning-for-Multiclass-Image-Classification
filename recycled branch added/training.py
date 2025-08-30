import numpy as np
from pennylane import RMSPropOptimizer
try:
    from .quantum_circuits import QCNN
    from .classical_layers import FullyConnectedLayer, HiddenLayer, FinalLayer, process_neurons
    from .model_utils import save_best_model
    from .visualization import plot_training_loss_history, plot_accuracy_same_graph
    from .config import steps, batch_size
except ImportError:
    from quantum_circuits import QCNN
    from classical_layers import FullyConnectedLayer, HiddenLayer, FinalLayer, process_neurons
    from model_utils import save_best_model
    from visualization import plot_training_loss_history, plot_accuracy_same_graph
    from config import steps, batch_size


def accuracy_test(predictions, labels):
    correct = 0
    for i, (p, l) in enumerate(zip(predictions, labels)):
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


def cost(params, X_batch, Y_batch, U, U_params, embedding_type, circuit):
    total_conv_params = U_params * 6
    total_pooling_params = 2 * 2
    quantum_params = params[:total_conv_params + total_pooling_params]
    classical_params = params[total_conv_params + total_pooling_params:]

    predictions = []
    for x in X_batch:
        probabilities_taken, probabilities_lost = QCNN(x, quantum_params, U, U_params, embedding_type)
        combined_output = process_neurons(probabilities_taken, probabilities_lost, classical_params)
        predictions.append(combined_output)

    return square_loss(Y_batch, predictions)


def circuit_training(X_train, Y_train, X_test, Y_test, U, U_params, embedding_type, circuit, cost_fn, hidden_dim=16):
    total_conv_params = U_params * 6  # 6 convolutional layers
    total_pooling_params = 2 * 2  # 2 pooling layers, each with 2 parameters
    total_quantum_params = total_conv_params + total_pooling_params  # Total quantum params

    total_classical_params = (
        FullyConnectedLayer(4, 4).get_params().size +
        HiddenLayer(4, 16).get_params().size +
        FinalLayer(16, 4).get_params().size
    )

    # Initialize all parameters
    params = np.random.randn(total_quantum_params + total_classical_params)
    
    optimizer = RMSPropOptimizer(stepsize=0.001)
    loss_history = []
    train_accuracy_history = []  # List to store training accuracy values
    test_accuracy_history = []   # List to store test accuracy values

    best_accuracy = 0  # Track the best test accuracy
    best_params = None  # Store the best model's parameters

    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), batch_size)
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        # Optimizer step and cost calculation
        params, cost_new = optimizer.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit), params)
        
        # Record loss every 10 steps
        if it % 10 == 0:
            loss_history.append(cost_new)
            print(f"Iteration: {it}, Loss: {cost_new}")

        # Calculate accuracy every 50 steps for both training and test sets
        if it % 50 == 0 and it != 0:
            # Calculate training accuracy
            train_predictions = []
            for x in X_batch:
                probabilities_taken, probabilities_lost = QCNN(x, params[:total_quantum_params], U, U_params, embedding_type)
                combined_output = process_neurons(probabilities_taken, probabilities_lost, params[total_quantum_params:])
                train_predictions.append(combined_output)
            
            train_accuracy = accuracy_test(train_predictions, Y_batch)
            train_accuracy_history.append(train_accuracy)  # Append training accuracy to history
            print(f"Iteration: {it}, Training Accuracy: {train_accuracy}")

            # Calculate test accuracy
            test_predictions = []
            for x in X_test:
                probabilities_taken, probabilities_lost = QCNN(x, params[:total_quantum_params], U, U_params, embedding_type)
                combined_output = process_neurons(probabilities_taken, probabilities_lost, params[total_quantum_params:])
                test_predictions.append(combined_output)
            
            test_accuracy = accuracy_test(test_predictions, Y_test)
            test_accuracy_history.append(test_accuracy)  # Append test accuracy to history
            print(f"Iteration: {it}, Test Accuracy: {test_accuracy}")

            # If the current test accuracy is the best we've seen, save the model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = params.copy()  # Store the best parameters
                save_best_model(best_params)  # Save the best parameters to a file

    # Plot training loss, training accuracy, and test accuracy separately
    plot_training_loss_history(loss_history, interval=10)
    plot_accuracy_same_graph(train_accuracy_history, test_accuracy_history, interval=50)
    
    return loss_history, train_accuracy_history, test_accuracy_history, best_params