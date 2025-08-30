# Hybrid Quantum-Classical Learning for Multiclass Image Classification

[![arXiv](https://img.shields.io/badge/arXiv-2508.18161-b31b1b.svg)](https://arxiv.org/abs/2508.18161)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35.1-green.svg)](https://pennylane.ai/)

This repository contains the implementation of the hybrid quantum-classical architecture described in the paper ["Hybrid Quantum-Classical Learning for Multiclass Image Classification"](https://arxiv.org/abs/2508.18161) by Anwar et al. The work proposes a novel approach that leverages discarded qubit states from quantum pooling operations to improve classification performance.

## 🎯 Key Novelties

Traditional Quantum Convolutional Neural Networks (QCNNs) discard qubit states after pooling operations, losing valuable quantum information. Our hybrid architecture **recycles these discarded qubits** and processes them through dedicated classical layers, achieving significant performance improvements:

- **MNIST**: 68.51% → 93.55% accuracy (+25.04%)
- **Fashion-MNIST**: 81.40% → 96.45% accuracy (+15.05%)
- **OrganAMNIST**: 69.13% → 88.50% accuracy (+19.37%)

## 🏗️ Architecture Overview

The hybrid quantum-classical architecture consists of three main components:

### 1. Quantum Processing Pipeline

- **Data Embedding**: Amplitude or Angle embedding based on feature dimensionality
- **Quantum Convolution**: 6 convolutional layers using SU(4) unitary ansatz (15 parameters each)
- **Quantum Pooling**: 2 pooling layers that traditionally discard half the qubits

### 2. Novel Qubit Recycling Mechanism

- **Retained Qubits**: Continue through the quantum circuit to final measurement
- **Discarded Qubits**: Measured after first pooling layer and fed to classical processing
- **Scaling Layer**: Maps quantum probabilities from [0,1] to [-2,2] for optimal classical processing

### 3. Classical Integration

- **Retained Branch**: Single fully connected layer (4→4) with tanh activation
- **Discarded Branch**: Three-layer network (4→16→4) for feature expansion and contraction
- **Fusion**: Element-wise Hadamard product of both branches
- **Classification**: Direct optimization with cross-entropy loss

## 📁 Repository Structure

```
├── baseline/                    # Baseline QCNN implementation (without recycling)
│   ├── main.py                 # Entry point for baseline experiments
│   ├── quantum_circuits.py     # Quantum circuit definitions
│   ├── classical_layers.py     # Classical processing layers
│   ├── training.py            # Training loop and optimization
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── embeddings.py          # Quantum state embedding methods
│   ├── benchmarking.py        # Evaluation and metrics
│   ├── visualization.py       # Plotting and visualization
│   └── config.py              # Hyperparameters and configuration
├── recycled branch added/      # Proposed hybrid architecture (with recycling)
│   ├── main.py                # Entry point for hybrid experiments
│   ├── quantum_circuits.py    # Modified quantum circuits with dual outputs
│   ├── classical_layers.py    # Enhanced classical layers for fusion
│   ├── training.py           # Training with dual-branch processing
│   └── [other files...]      # Similar structure to baseline
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/shuchismita-anwar/Hybrid-Quantum-Classical-Learning-for-Multiclass-Image-Classification.git
cd hybrid-quantum-classical-learning
```

2. **Create virtual environment** (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Running Experiments

#### Baseline QCNN (without recycling)

```bash
cd baseline
python main.py
```

#### Hybrid Architecture (with recycling)

```bash
cd "recycled branch added"
python main.py
```

### Configuration

Modify the main configuration in `main.py`:

```python
# Dataset and classes
dataset = 'OrganAMNIST'  # Options: 'mnist', 'fashion_mnist', 'OrganAMNIST'
classes = [0, 3, 7, 8]   # Target classes for classification

# Feature reduction method
Encodings = ['autoencoder8']  # Options: 'resize256', 'pca8', 'autoencoder8', etc.

# Quantum circuit configuration
Unitaries = ['U_SU4']    # Quantum unitary ansatz
U_num_params = [15]      # Parameters per unitary gate
```

Training hyperparameters in `config.py`:

```python
steps = 2001           # Training iterations
learning_rate = 0.001  # Optimizer learning rate
batch_size = 25        # Mini-batch size
```

## 🔬 Technical Details

### Quantum Circuit Architecture

The quantum backbone follows a (conv, conv, pool) pattern:

1. **Block 1**: Conv → Conv → Pool (8→4 qubits)
2. **Block 2**: Conv → Conv → Pool (4→2 qubits)
3. **Block 3**: Conv → Conv (2 qubits final)

Each convolutional layer uses the SU(4) ansatz with 15 trainable parameters:

```python
def U_SU4(params, wires):
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    # ... additional gates
```

### Scaling Mechanism

The key innovation is the scaling layer that maps quantum probabilities:

```python
def scaled_output(probabilities):
    return (probabilities * 4) - 2  # [0,1] → [-2,2]
```

This transformation:

- Centers outputs at origin for optimal tanh activation
- Preserves gradient flow during backpropagation
- Maintains relative probability ordering

### Classical Fusion

The Hadamard product fusion captures quantum-classical correlations:

```python
def dot_product(neurons_taken, neurons_lost):
    return np.multiply(neurons_taken, neurons_lost)
```

This multiplicative fusion acts as a soft logical AND, requiring agreement between both quantum branches for confident predictions.

## 📊 Experimental Results

### Performance Comparison

| Dataset                 | Baseline | Hybrid | Improvement |
| ----------------------- | -------- | ------ | ----------- |
| MNIST (0,1,2,3)         | 70.03%   | 93.55% | +23.52%     |
| MNIST (3,4,5,6)         | 68.51%   | 88.52% | +20.01%     |
| Fashion-MNIST (0,1,2,3) | 74.30%   | 86.55% | +12.25%     |
| Fashion-MNIST (1,2,8,9) | 81.40%   | 96.45% | +15.05%     |
| OrganAMNIST (0,3,7,8)   | 69.13%   | 88.50% | +19.37%     |

### Parameter Efficiency

The hybrid model achieves superior performance with minimal overhead:

- **Quantum parameters**: 94 (6 conv layers × 15 + 2 pool layers × 2)
- **Classical parameters**: 168 (retained: 20, discarded: 148)
- **Total parameters**: 262

Compared to classical CNNs requiring 10⁴-10⁶ parameters, our approach is 1-2 orders of magnitude more efficient.

## 🔧 Customization

### Changing Target Qubits

Modify qubit selection in `quantum_circuits.py`:

```python
taken_qubits = [0, 4]  # Qubits continuing to final measurement
lost_qubits = [3, 5]   # Qubits recycled after first pooling
```

### Adding New Datasets

Extend `data_processing.py`:

```python
def data_load_and_process(dataset='new_dataset', classes=[0, 1, 2, 3], feature_reduction='autoencoder8'):
    if dataset == 'new_dataset':
        # Load your dataset here
        x_train, y_train = load_new_dataset()
        # ... preprocessing
```

### Custom Feature Reduction

Add new encoding methods in `config.py` and `data_processing.py`:

```python
# In config.py
custom_encoding = ['custom-1', 'custom-2']

# In data_processing.py
elif feature_reduction in custom_encoding:
    # Implement custom feature reduction
```

## 📈 Monitoring Training

The framework provides comprehensive monitoring:

- **Loss curves**: Training and validation loss over iterations
- **Accuracy plots**: Train/test accuracy comparison
- **Confusion matrices**: Per-class performance analysis
- **Metrics**: Precision, recall, F1-score for each class

## 🔍 Key Files Explained

### `quantum_circuits.py`

- Defines quantum circuit architecture
- Implements SU(4) convolutional ansatz
- Handles quantum pooling operations
- **Baseline**: Returns only retained qubit probabilities
- **Hybrid**: Returns both retained and discarded qubit probabilities

### `classical_layers.py`

- Implements fully connected layers with Xavier initialization
- **Baseline**: Single processing path for retained qubits
- **Hybrid**: Dual processing paths with Hadamard fusion

### `training.py`

- End-to-end training loop with parameter-shift rule
- Gradient computation for quantum-classical hybrid
- Performance evaluation and metrics calculation

### `benchmarking.py`

- Comprehensive evaluation framework
- Cross-validation and statistical analysis
- Comparison with baseline methods

## 🎯 Future Directions

Based on the paper's conclusions, potential extensions include:

1. **Quantum-Native Problems**: Extend to quantum phase recognition and error syndrome classification
2. **Deeper Architectures**: Investigate recycling from multiple pooling layers
3. **Hardware Implementation**: Deploy on actual NISQ devices
4. **Scalability**: Test on larger datasets and more complex tasks
5. **Theoretical Analysis**: Formal study of entanglement preservation in recycled qubits

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{anwar2025hybridquantumclassicallearningmulticlass,
  title={Hybrid Quantum-Classical Learning for Multiclass Image Classification},
  author={Shuchismita Anwar and Sowmitra Das and Muhammad Iqbal Hossain and Jishnu Mahmud},
  year={2025},
  eprint={2508.18161},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2508.18161}
}
```

**Preprint Paper**: [arXiv:2508.18161](https://arxiv.org/abs/2508.18161)


