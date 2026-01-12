import numpy as np
import urllib.request
from autograd import Tensor
from nn import Linear, ReLU, LogSoftmax, NLLLoss, SGD, Module

# 1. Download and Load Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
print(f"Downloading Iris dataset from {url}...")
with urllib.request.urlopen(url) as f:
    raw_data = f.read().decode('utf-8')

lines = raw_data.strip().split('\n')
data = []
labels = []
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

for line in lines:
    if not line: continue
    parts = line.split(',')
    data.append([float(x) for x in parts[:-1]])
    labels.append(label_map[parts[-1]])

X = np.array(data)
y = np.array(labels)

# 2. Preprocessing
# Shuffle
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Split Train/Test (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_tensor = Tensor(X_train)
y_train_tensor = Tensor(y_train)
X_test_tensor = Tensor(X_test)
# y_test is kept as numpy for evaluation

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Define Model
class IrisModel(Module):
    def __init__(self):
        self.l1 = Linear(4, 16)
        self.relu = ReLU()
        self.l2 = Linear(16, 3) # 3 classes
        self.log_softmax = LogSoftmax(dim=1)

    def __call__(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.log_softmax(x)
        return x
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

model = IrisModel()
criterion = NLLLoss()
optimizer = SGD(model.parameters(), lr=0.1)

# 4. Training Loop
epochs = 500
print("Starting training...")
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

print("Training finished.")
print(f"Final Training Loss: {loss.data}")

# 5. Evaluation
logits = model(X_test_tensor)
probs = np.exp(logits.data)
predictions = np.argmax(probs, axis=1)
accuracy = (predictions == y_test).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
