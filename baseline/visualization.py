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
    plt.plot(iterations, train_accuracy_history, label='Training Accuracy', color='blue')
    plt.plot(iterations, test_accuracy_history, label='Test Accuracy', color='green')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylim(top=1.05)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix with seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()