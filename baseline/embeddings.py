import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding


def data_embedding(X, embedding_type='Amplitude'):
    """
    Embed classical data into quantum states using different embedding strategies.
    
    Args:
        X: Input data to embed
        embedding_type: Type of embedding ('Amplitude', 'Angle', 'Angle-compact')
    """
    if embedding_type == 'Amplitude':
        qml.templates.AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        qml.templates.AngleEmbedding(X, wires=range(8), rotation='Y')
    elif embedding_type == 'Angle-compact':
        qml.templates.AngleEmbedding(X[:8], wires=range(8), rotation='X')
        qml.templates.AngleEmbedding(X[8:16], wires=range(8), rotation='Y')


def Encoding_to_Embedding(Encoding):
    """
    Map encoding types to embedding types.
    
    Args:
        Encoding: Feature reduction encoding type
        
    Returns:
        Corresponding embedding type for quantum circuits
    """
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'

    return Embedding