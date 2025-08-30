import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.utils import to_categorical
from medmnist import OrganAMNIST
import matplotlib.pyplot as plt
try:
    from .config import pca32, autoencoder32, pca30, autoencoder30, pca16, autoencoder16, pca12, autoencoder12
except ImportError:
    from config import pca32, autoencoder32, pca30, autoencoder30, pca16, autoencoder16, pca12, autoencoder12


def data_load_and_process(dataset='fashion_mnist', classes=[0, 3, 7, 8], feature_reduction='resize256'):
    """
    Load and process dataset for QCNN training.
    
    CONFIGURATION PARAMETERS:
    - dataset: Choose from 'fashion_mnist', 'mnist', or 'OrganAMNIST' - MODIFY HERE to change dataset
    - classes: List of class indices to include in training - MODIFY HERE to change target classes
    - feature_reduction: Method for dimensionality reduction - MODIFY HERE to change preprocessing
    
    Available datasets:
    - 'fashion_mnist': Fashion-MNIST dataset (clothing items)
    - 'mnist': MNIST handwritten digits
    - 'OrganAMNIST': Medical imaging dataset
    
    Available feature_reduction methods:
    - 'resize256': Resize images to 256 features
    - 'pca8', 'pca12', 'pca16', 'pca30', 'pca32': PCA with different dimensions
    - 'autoencoder8', 'autoencoder12', 'autoencoder16', 'autoencoder30', 'autoencoder32': Autoencoder compression
    """
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'OrganAMNIST':
        pneumonia_train = OrganAMNIST(split="train", download=True)
        x_train, y_train = pneumonia_train.imgs, pneumonia_train.labels
        pneumonia_test = OrganAMNIST(split="test", download=True)
        x_test, y_test = pneumonia_test.imgs, pneumonia_test.labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    # Filter the dataset for the specified classes
    train_filter = np.isin(y_train, classes)
    test_filter = np.isin(y_test, classes)
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # Map the labels to a new range (0 to number of classes - 1)
    label_map = {c: i for i, c in enumerate(classes)}
    y_train = np.vectorize(label_map.get)(y_train)
    y_test = np.vectorize(label_map.get)(y_test)

    # One-hot encode the labels
    # CONFIGURATION: Change num_classes if you modify the number of classes
    num_classes = len(classes)  # Automatically set based on classes parameter
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    if feature_reduction == 'resize256':
        x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
        x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
        x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()
        
        plt.figure(figsize=(10, 2))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(x_train[i].reshape((16, 16)), cmap='gray')  # Reshape to view as an image
            plt.title(f"Label: {y_train[i]}")
            plt.axis('off')
        plt.show()

        return x_train, x_test, y_train, y_test

    elif feature_reduction == 'pca8' or feature_reduction in pca32 \
            or feature_reduction in pca30 or feature_reduction in pca16 or feature_reduction in pca12:

        x_train = tf.image.resize(x_train[:], (784, 1)).numpy()
        x_test = tf.image.resize(x_test[:], (784, 1)).numpy()
        x_train, x_test = tf.squeeze(x_train), tf.squeeze(x_test)

        if feature_reduction == 'pca8':
            pca = PCA(8)
        elif feature_reduction in pca32:
            pca = PCA(32)
        elif feature_reduction in pca30:
            pca = PCA(30)
        elif feature_reduction in pca16:
            pca = PCA(16)
        elif feature_reduction in pca12:
            pca = PCA(12)

        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        # Rescale for angle embedding
        if feature_reduction == 'pca8' or feature_reduction == 'pca16-compact' or \
                feature_reduction in pca30 or feature_reduction in pca12:
            x_train, x_test = (x_train - x_train.min()) * (np.pi / (x_train.max() - x_train.min())),\
                              (x_test - x_test.min()) * (np.pi / (x_test.max() - x_test.min()))
        return x_train, x_test, y_train, y_test

    elif feature_reduction == 'autoencoder8' or feature_reduction in autoencoder32 \
            or feature_reduction in autoencoder30 or feature_reduction in autoencoder16 or feature_reduction in autoencoder12:
        if feature_reduction == 'autoencoder8':
            latent_dim = 8
        elif feature_reduction in autoencoder32:
            latent_dim = 32
        elif feature_reduction in autoencoder30:
            latent_dim = 30
        elif feature_reduction in autoencoder16:
            latent_dim = 16
        elif feature_reduction in autoencoder12:
            latent_dim = 12

        class Autoencoder(Model):
            def __init__(self, latent_dim):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim
                self.encoder = tf.keras.Sequential([
                    layers.Flatten(),
                    layers.Dense(latent_dim, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Dense(784, activation='sigmoid'),
                    layers.Reshape((28, 28))
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        autoencoder = Autoencoder(latent_dim)

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder.fit(x_train, x_train,
                        epochs=10,
                        shuffle=True,
                        validation_data=(x_test, x_test))

        x_train, x_test = autoencoder.encoder(x_train).numpy(), autoencoder.encoder(x_test).numpy()

        plt.figure(figsize=(10, 2))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            try:
                # Try reshaping to 16x16 (for example, if the feature_reduction is 'resize256')
                plt.imshow(x_train[i].reshape((16, 16)), cmap='gray')
            except ValueError:
                # If reshaping fails, fall back to another valid shape for smaller arrays
                plt.imshow(x_train[i].reshape((int(np.sqrt(x_train[i].size)), -1)), cmap='gray')
            plt.title(f"Label: {y_train[i]}")
            plt.axis('off')
        plt.show()

        # Rescale for Angle Embedding
        # Note this is not a rigorous rescaling method
        if feature_reduction == 'autoencoder8' or feature_reduction == 'autoencoder16-compact' or\
                feature_reduction in autoencoder30 or feature_reduction in autoencoder12:
            x_train, x_test = (x_train - x_train.min()) * (np.pi / (x_train.max() - x_train.min())), \
                              (x_test - x_test.min()) * (np.pi / (x_test.max() - x_test.min()))

        return x_train, x_test, y_train, y_test