import torch
import torch.nn as nn


# ------------------------------------------------------
# Branch A: Tabular MLP
# ------------------------------------------------------
class TabularBranch(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# Branch B: Sequence LSTM
# ------------------------------------------------------
class SequenceBranch(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x shape: (batch, 5) → need (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        return h[-1]   # final hidden state


# ------------------------------------------------------
# Branch C: Embedding (384 → 64)
# ------------------------------------------------------
class EmbeddingBranch(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------
# Branch D: Graph scalar (1 → 16)
# ------------------------------------------------------
class GraphBranch(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------
# Full Fusion Model
# ------------------------------------------------------
class MultimodalEngagementNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.tab_branch = TabularBranch()
        self.seq_branch = SequenceBranch()
        self.emb_branch = EmbeddingBranch()
        self.graph_branch = GraphBranch()

        # Output dimensions:
        # Tabular = 32
        # Sequence = 32
        # Embedding = 64
        # Graph = 16
        fusion_dim = 32 + 32 + 64 + 16

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        tab = self.tab_branch(batch["tabular"])
        seq = self.seq_branch(batch["sequence"])
        emb = self.emb_branch(batch["embedding"])
        graph = self.graph_branch(batch["graph"])

        fused = torch.cat([tab, seq, emb, graph], dim=1)

        out = self.fusion(fused)
        return out
