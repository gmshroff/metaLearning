# autograd_knn_vs_metric.py
# Demonstrates: k-NN fails in raw space but works after contrastive metric learning
# Using custom autograd.py and nn.py instead of PyTorch

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from autograd import Tensor
from nn import Module, Linear, ReLU, AdamW, SGD

def make_noisy_parity(n=2000, d=50, noise_std=5.0):
    """
    Label depends ONLY on x[0] (1D signal)
    Remaining d-1 dimensions are high-variance noise
    """
    X = np.random.randn(n, d) * noise_std
    signal = np.random.randn(n)

    # Inject signal into first dimension
    X[:, 0] = signal

    # Binary label from signal
    y = (signal > 0).astype(int)
    return X, y

def make_pairs(X, y, n_pairs=6000):
    pairs_idx = []
    labels = []

    for _ in range(n_pairs):
        i = np.random.randint(len(X))
        if np.random.rand() < 0.5:
            # same class
            j = np.random.choice(np.where(y == y[i])[0])
            label = 1.0
        else:
            # different class
            j = np.random.choice(np.where(y != y[i])[0])
            label = 0.0

        pairs_idx.append([i, j])
        labels.append(label)

    return pairs_idx, np.array(labels)

# 1. Data Setup
X_raw, y_raw = make_noisy_parity()
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 2. k-NN on raw input space
knn_raw = KNeighborsClassifier(n_neighbors=15)
knn_raw.fit(X_train, y_train)
y_pred_raw = knn_raw.predict(X_test)

print("k-NN accuracy (raw space):", accuracy_score(y_test, y_pred_raw))

# 3. Metric learning components
class EmbeddingNet(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(50, 32)
        self.relu = ReLU()
        self.l2 = Linear(32, 2)

    def __call__(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

class ContrastiveLoss(Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def __call__(self, z1, z2, label):
        # label = 1 for same class, 0 for different
        # Use 1 - cosine similarity as distance
        
        # Calculate Norms
        z1_norm_val = (z1**2).sum(axis=1, keepdims=True)**0.5
        z2_norm_val = (z2**2).sum(axis=1, keepdims=True)**0.5
        
        z1_norm = z1 / (z1_norm_val + 1e-8)
        z2_norm = z2 / (z2_norm_val + 1e-8)
        
        # Cosine similarity
        cosine_sim = (z1_norm * z2_norm).sum(axis=1)
        dist = Tensor(1.0) - cosine_sim
        
        # Loss calculation
        if not isinstance(label, Tensor):
            label = Tensor(label)
            
        loss_same = label * (dist**2)
        # torch.clamp(margin - dist, min=0) is (margin - dist).relu()
        loss_diff = (Tensor(1.0) - label) * (Tensor(self.margin) - dist).relu()**2
        
        total_loss = (loss_same + loss_diff).sum() / Tensor(float(dist.data.shape[0]))
        return total_loss

# 4. Train metric learner
pairs_idx, pair_labels = make_pairs(X_train, y_train)

# Convert whole training set to Tensor once? Or batch?
# For 6000 pairs, it's manageable.
X_train_tensor = Tensor(X_train)
pair_labels_tensor = Tensor(pair_labels)

model = EmbeddingNet()
criterion = ContrastiveLoss(margin=1.0)
optimizer = AdamW(model.parameters(), lr=0.005)
# optimizer = SGD(model.parameters(), lr=0.005)

print("Starting training...")
for epoch in range(501):
    optimizer.zero_grad()
    
    # Extract pairs from X_train_tensor
    # pairs_idx is (6000, 2)
    i_indices = [p[0] for p in pairs_idx]
    j_indices = [p[1] for p in pairs_idx]
    
    z1 = model(X_train_tensor[i_indices])
    z2 = model(X_train_tensor[j_indices])
    
    loss = criterion(z1, z2, pair_labels_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Contrastive loss: {loss.data:.4f}")

# 5. k-NN on learned embedding
# No grad for evaluation
X_train_emb = model(Tensor(X_train)).data
X_test_emb = model(Tensor(X_test)).data

knn_emb = KNeighborsClassifier(n_neighbors=3)
knn_emb.fit(X_train_emb, y_train)
y_pred_emb = knn_emb.predict(X_test_emb)

print("k-NN accuracy (learned metric):", accuracy_score(y_test, y_pred_emb))
