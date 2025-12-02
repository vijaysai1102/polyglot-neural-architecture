import torch
from torch.utils.data import Dataset
import random
from src import db_client


class MultiModalDataset(Dataset):
    def __init__(self, users, movies, length=2000):
        """
        users: list of user_ids
        movies: list of movie_ids
        length: number of samples 
        """
        self.users = users
        self.movies = movies
        self.length = length

        # randomly generate (user, movie) pairs
        self.samples = [
            (random.choice(users), random.choice(movies))
            for _ in range(length)
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        user_id, movie_id = self.samples[idx]

        # -------------------------
        # SQL → tabular features
        # -------------------------
        tab = db_client.get_user_tabular(user_id)
        tab_tensor = torch.tensor(
            [tab["age"], tab["city"], tab["sub_price"]],
            dtype=torch.float32
        )

        # -------------------------
        # Mongo → sequence features
        # -------------------------
        sessions = db_client.get_last_sessions(user_id, n=5)
        seq_tensor = torch.tensor(sessions, dtype=torch.float32)

        # -------------------------
        # Vector → movie embedding
        # -------------------------
        emb = db_client.get_movie_embedding(movie_id)
        emb_tensor = torch.tensor(emb, dtype=torch.float32)

        # -------------------------
        # Graph → scalar feature
        # -------------------------
        graph_val = db_client.get_graph_feature(user_id, movie_id)
        graph_tensor = torch.tensor([graph_val], dtype=torch.float32)

        # -------------------------
        # Target label (synthetic)
        # User engagement = random 0-1 float
        # -------------------------
        target = torch.tensor([random.random()], dtype=torch.float32)

        return {
            "tabular": tab_tensor,
            "sequence": seq_tensor,
            "embedding": emb_tensor,
            "graph": graph_tensor,
            "target": target
        }
