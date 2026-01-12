# knn_vs_metric_learning.py
# Demonstrates: k-NN fails in raw space but works after contrastive metric learning

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

X, y = make_noisy_parity()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. k-NN on raw input space
# -----------------------------
knn_raw = KNeighborsClassifier(n_neighbors=15)
knn_raw.fit(X_train, y_train)
y_pred_raw = knn_raw.predict(X_test)

print("k-NN accuracy (raw space):",
      accuracy_score(y_test, y_pred_raw))

# -----------------------------
# 3. Metric learning components
# -----------------------------
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=.8):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        # label = 1 for same class, 0 for different
        # Use 1 - cosine similarity as distance
        z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (z1_norm * z2_norm).sum(dim=1)
        dist = 1.0 - cosine_sim
        loss_same = label * dist.pow(2)
        loss_diff = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return (loss_same + loss_diff).mean()


def make_pairs(X, y, n_pairs=6000):
    pairs = []
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

        pairs.append([X[i], X[j]])
        labels.append(label)

    return (
        torch.tensor(pairs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)
    )

# -----------------------------
# 4. Train metric learner
# -----------------------------
pairs, pair_labels = make_pairs(X_train, y_train)

model = EmbeddingNet()
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.005)
# optimizer = optim.SGD(model.parameters(),lr=.005)

for epoch in range(500):
    optimizer.zero_grad()

    z1 = model(pairs[:, 0])
    z2 = model(pairs[:, 1])

    loss = criterion(z1, z2, pair_labels)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Contrastive loss: {loss.item():.4f}")

# -----------------------------
# 5. k-NN on learned embedding
# -----------------------------
with torch.no_grad():
    X_train_emb = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
    X_test_emb = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

knn_emb = KNeighborsClassifier(n_neighbors=3)
knn_emb.fit(X_train_emb, y_train)
y_pred_emb = knn_emb.predict(X_test_emb)

print("k-NN accuracy (learned metric):",
      accuracy_score(y_test, y_pred_emb))
