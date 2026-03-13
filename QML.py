import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Create results folder
# ---------------------------
os.makedirs("results", exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("6g_dataset.csv")

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert labels to float (-1, 1)
y = np.where(y == 0, -1.0, 1.0)

# ---------------------------
# Normalize features
# ---------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Use first 4 features
n_qubits = 4
X = X[:, :n_qubits]

# ---------------------------
# Train Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Quantum Device
# ---------------------------
dev = qml.device("default.qubit", wires=n_qubits)

# ---------------------------
# Quantum Model
# ---------------------------
n_layers = 3

@qml.qnode(dev)
def quantum_model(x, weights):

    # Feature encoding
    qml.AngleEmbedding(x, wires=range(n_qubits))

    # Variational circuit
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return qml.expval(qml.PauliZ(0))

# ---------------------------
# Loss Function
# ---------------------------
def loss_fn(weights, x, y):

    pred = quantum_model(x, weights)

    return (pred - y) ** 2

# ---------------------------
# Prediction Function
# ---------------------------
def predict(X, weights):

    preds = []

    for x in X:

        pred = quantum_model(x, weights)

        preds.append(np.sign(pred))

    return np.array(preds)

# ---------------------------
# Initialize Weights
# ---------------------------
weights = np.random.randn(n_layers, n_qubits, 3, requires_grad=True)

lr = 0.02
epochs = 50

loss_history = []
acc_history = []

print("\nTraining Started\n")

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(epochs):

    total_loss = 0

    for x, y_true in zip(X_train, y_train):

        grad = qml.grad(loss_fn, argnum=0)(weights, x, y_true)

        weights = weights - lr * grad

        total_loss += loss_fn(weights, x, y_true)

    avg_loss = total_loss / len(X_train)

    preds_train = predict(X_train, weights)

    train_acc = accuracy_score(y_train, preds_train)

    loss_history.append(avg_loss)
    acc_history.append(train_acc)

    print(
        f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}"
    )

# ---------------------------
# Testing
# ---------------------------
preds_test = predict(X_test, weights)

test_accuracy = accuracy_score(y_test, preds_test)

print("\nFinal Test Accuracy:", test_accuracy)

# ---------------------------
# Loss Graph
# ---------------------------
plt.figure()
plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid()
plt.savefig("results/loss_curve.png")
plt.show()

# ---------------------------
# Accuracy Graph
# ---------------------------
plt.figure()
plt.plot(range(1, epochs + 1), acc_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epoch")
plt.grid()
plt.savefig("results/accuracy_curve.png")
plt.show()

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, preds_test)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1], ["-1","+1"])
plt.yticks([0,1], ["-1","+1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/confusion_matrix.png")
plt.show()

# ---------------------------
# Quantum Circuit Diagram
# ---------------------------
drawer = qml.draw_mpl(quantum_model)

fig, ax = drawer(X_train[0], weights)

plt.savefig("results/quantum_circuit.png")
plt.show()