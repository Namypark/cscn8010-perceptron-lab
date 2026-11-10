
# This script trains a single-layer perceptron in PyTorch using manual gradient updates.
# The forward pass computes z = w·x + b, applies a sigmoid, and measures MSE loss.
# PyTorch builds a computation graph during the forward pass so that loss.backward()
# can compute gradients with respect to weights and bias.

# When updating the parameters, we wrap the update step inside `with torch.no_grad():`.
# This tells PyTorch not to record these operations. Without this, PyTorch would treat
# the weight update as part of the computational graph, causing the graph to grow
# every training step and eventually break backpropagation.

# Using no_grad() keeps the computation graph small and fresh each iteration. It prevents
# memory leaks, avoids "backward through the graph a second time" errors, and ensures the
# model trains correctly. After updating, gradients are cleared with .zero_() so they do
# not accumulate between steps.

# TODO: 
# Run this script to see how the weights and bias evolve over epochs as the model learns 
# to fit the OR function.  
# Run the code using the debugger to step through each line and understand how the forward 
# pass, loss computation, backpropagation, and parameter updates work together in training 
# a perceptron.   
# Note: This is a minimal example for educational purposes. In practice, you would typically 
# use a DataLoader and batch training for larger datasets.   

import argparse

import torch

# usage: python scripts/train_or_perceptron.py --lr 0.1 --epochs 200
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

# -------------------------
# 1. Create toy training data
# -------------------------
# Input: 2D points
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

# Labels for OR function
y = torch.tensor([[0.], [1.], [1.], [1.]])

# -------------------------
# 2. Initialize parameters
# -------------------------
torch.manual_seed(0)
weights = torch.randn(2, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

lr = args.lr
epochs = args.epochs

# -------------------------
# 3. Training loop
# -------------------------
for epoch in range(epochs):

    epoch_loss = 0

    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        # Forward pass
        z = torch.dot(weights, x_i) + bias
        y_hat = torch.sigmoid(z)

        # Loss (MSE)
        loss = (y_hat - y_i) ** 2
        epoch_loss += loss.item()

        # Backprop
        loss.backward()

        # Update params without tracking gradients
        with torch.no_grad():
            weights -= lr * weights.grad
            bias -= lr * bias.grad

        # Reset gradients
        weights.grad.zero_()
        bias.grad.zero_()

    print(f"Epoch {epoch+1}, Loss = {epoch_loss:.4f}")

# -------------------------
# 4. Final parameters
# -------------------------
print("\nTrained weights:", weights)
print("Trained bias:", bias)

# check what the trained neuron actually predicts
with torch.no_grad():
    preds = torch.sigmoid(X @ weights + bias)
for x_i, y_i, p in zip(X, y, preds):
    print(f"x = {x_i.int().tolist()}  target = {int(y_i.item())}  sigmoid = {p.item():.3f}  predicted = {int(p.item() >= 0.5)}")