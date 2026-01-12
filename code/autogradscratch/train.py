import numpy as np
from autograd import Tensor
from nn import Linear, ReLU, MSELoss, SGD, Module

# 1. Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) # 100 samples, 1 feature
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1) # y = 3x + 2 + noise

X_tensor = Tensor(X)
y_tensor = Tensor(y)

# 2. Define Model
class MLP(Module):
    def __init__(self):
        self.l1 = Linear(1, 10)
        self.relu = ReLU()
        self.l2 = Linear(10, 1)

    def __call__(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

model = MLP()
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.05)

# 3. Training Loop
epochs = 500
print("Starting training...")
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

print("Training finished.")
print(f"Final Loss: {loss.data}")

# Test on a new value
test_val = Tensor(np.array([[0.5]]))
pred = model(test_val)
print(f"Prediction for x=0.5: {pred.data[0][0]:.4f} (Expected: {3*0.5 + 2:.4f})")
