import numpy as np
from autograd import Tensor
from nn import Linear, ReLU, LogSoftmax, NLLLoss, SGD, Module

# 1. Generate XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

X_tensor = Tensor(X)
y_tensor = Tensor(y)

# 2. Define Model
class XORModel(Module):
    def __init__(self):
        self.l1 = Linear(2, 8)
        self.relu = ReLU()
        self.l2 = Linear(8, 2) # Output 2 classes
        self.log_softmax = LogSoftmax(dim=1)

    def __call__(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.log_softmax(x)
        return x
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

model = XORModel()
criterion = NLLLoss()
optimizer = SGD(model.parameters(), lr=0.1)

# 3. Training Loop
epochs = 1000
print("Starting XOR training...")
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

print("Training finished.")
print(f"Final Loss: {loss.data}")

# Test predictions
logits = model(X_tensor)
probs = np.exp(logits.data)
predictions = np.argmax(probs, axis=1)
print(f"Predictions: {predictions}")
print(f"Ground Truth: {y}")
