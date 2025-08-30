import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_loss_history(loss_history, interval=10):
    iterations = [i * interval for i in range(len(loss_history))]
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    # Set x-ticks every 100 iterations (adjust as needed):
    plt.xticks(np.arange(0))
    plt.show()

    
def plot_accuracy_same_graph(train_accuracy_history, test_accuracy_history, interval=50):
    iterations = [i * interval for i in range(len(train_accuracy_history))]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_accuracy_history, label='Training Accuracy', marker='.', color='blue')
    plt.plot(iterations, test_accuracy_history, label='Test Accuracy', marker='.', color='green')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylim(top=1.05)
    plt.show()


def plot_confusion_matrix(predictions, labels):
    # Convert probabilities to class predictions
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()