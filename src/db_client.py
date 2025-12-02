import json
import numpy as np
import pandas as pd

# ----------------------------
# Load SQL CSVs
# ----------------------------
users_df = pd.read_csv("data/sql/users.csv")
subs_df = pd.read_csv("data/sql/subscriptions.csv")
payments_df = pd.read_csv("data/sql/payments.csv")

# ----------------------------
# Load Mongo JSONs
# ----------------------------
with open("data/mongo/playback_logs.json", "r") as f:
    playback_logs = json.load(f)

with open("data/mongo/reviews.json", "r") as f:
    reviews = json.load(f)

# ----------------------------
# Load Graph CSVs
# ----------------------------
edges_df = pd.read_csv("data/graph/edges.csv")

# ----------------------------
# Load vectors
# ----------------------------
movie_embeddings = np.load("data/vectors/movie_embeddings.npy")
with open("data/vectors/movie_ids.json", "r") as f:
    movie_ids = json.load(f)

movie_id_to_index = {m: i for i, m in enumerate(movie_ids)}


# ---------------------------------------------
# SQL ACCESSORS
# ---------------------------------------------
def get_user_tabular(user_id):
    """Return user age, city encoded, subscription price."""
    row = users_df[users_df["user_id"] == user_id].iloc[0]
    
    # Simple encoding for city
    city_map = {"New York": 0, "Dallas": 1, "San Jose": 2, "Chicago": 3}
    city_encoded = city_map.get(row["city"], 0)
    
    # Subscription price
    sub = subs_df[subs_df["subscription_id"] == row["subscription_id"]].iloc[0]
    price = sub["price"]

    return {
        "age": float(row["age"]),
        "city": float(city_encoded),
        "sub_price": float(price)
    }


# ---------------------------------------------
# MONGO ACCESSORS
# ---------------------------------------------
def get_last_sessions(user_id, n=5):
    """Return last n durations for this user (sequence data)."""
    doc = next(x for x in playback_logs if x["user_id"] == user_id)
    sessions = sorted(doc["sessions"], key=lambda x: x["timestamp"], reverse=True)
    
    durations = [s["duration_seconds"] for s in sessions[:n]]
    
    # pad if fewer than n
    while len(durations) < n:
        durations.append(0.0)

    return durations


# ---------------------------------------------
# VECTOR ACCESSOR
# ---------------------------------------------
def get_movie_embedding(movie_id):
    idx = movie_id_to_index[movie_id]
    return movie_embeddings[idx]


# ---------------------------------------------
# GRAPH ACCESSOR
# ---------------------------------------------
def get_graph_feature(user_id, movie_id):
    """
    Example feature: How many actors/directors are connected to this movie?
    (A very simple 'graph centrality' score.)
    """
    mask = edges_df["end"] == movie_id
    connected = edges_df[mask]
    return float(len(connected))
