import pennylane as qml
try:
    from .embeddings import data_embedding
except ImportError:
    from embeddings import data_embedding


# Unitary Ansatze for Convolutional Layer
def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


# Pooling Layer ansatzes
def Pooling_ansatz1(params, wires):
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])


def Pooling_ansatz2(wires): #0 params
    qml.CRZ(wires=[wires[0], wires[1]])


def Pooling_ansatz3(*params, wires): #3 params
    qml.CRot(*params, wires=[wires[0], wires[1]])


# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])


def conv_layer2(U, params):
    U(params, wires=[0, 6])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])


def conv_layer3(U, params):
    U(params, wires=[0,4])


# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])


def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])


def pooling_layer3(V, params):
    V(params, wires=[0,4])


def QCNN_structure(U, params, U_params):
    total_conv_params = U_params * 6  # Convolutional layer params
    total_pooling_params = 2 * 2  # Pooling layer params (2 params for each of 2 layers)
    
    # Ensure the params array is large enough
    assert len(params) >= (total_conv_params + total_pooling_params), "Insufficient parameters allocated."

    # Allocate parameters to convolutional layers
    param1 = params[0:U_params]
    param2 = params[U_params:2 * U_params]
    param3 = params[2 * U_params:3 * U_params]
    param4 = params[3 * U_params:4 * U_params]
    param5 = params[4 * U_params:5 * U_params]
    param6 = params[5 * U_params:total_conv_params]

    # Allocate parameters to pooling layers
    param7 = params[total_conv_params:total_conv_params + 2]  # Pooling layer 1
    param8 = params[total_conv_params + 2:total_conv_params + 4]  # Pooling layer 2

    # Apply conv and pooling layers
    conv_layer1(U, param1)
    conv_layer1(U, param2)
    pooling_layer1(Pooling_ansatz1, param7)

    conv_layer2(U, param3)
    conv_layer2(U, param4)
    pooling_layer2(Pooling_ansatz1, param8)

    conv_layer3(U, param5)
    conv_layer3(U, param6)


# Quantum device and circuit
dev = qml.device('default.qubit', wires=8)

@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Amplitude'):
    # Data Embedding
    data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_SU4':
        QCNN_structure(U_SU4, params, U_params)
    else:
        print("Invalid Unitary Ansatze")
        return False

    # CONFIGURATION: Change these qubit indices if you want different qubits for measurement
    # taken_qubits: qubits that continue to the classical processing layer
    # lost_qubits: qubits that are measured and discarded after quantum processing
    taken_qubits = [0, 4]  # qubits taken forward - MODIFY HERE to change taken qubits
    lost_qubits = [3, 5]   # qubits measured and lost - MODIFY HERE to change lost qubits
    probabilities_taken = qml.probs(wires=taken_qubits)
    probabilities_lost = qml.probs(wires=lost_qubits)

    return probabilities_taken, probabilities_lost