import os
import json
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    paths = [
        "data/sql", "data/mongo", "data/graph", "data/vectors"
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)

def random_date():
    return datetime.now() - timedelta(days=random.randint(0, 365))


# -----------------------------
# 1. SQL DATA
# -----------------------------
def generate_sql_data(num_users=500):
    # Subscriptions
    subscriptions = [
        {"subscription_id": 1, "tier_name": "Basic", "price": 7.99},
        {"subscription_id": 2, "tier_name": "Standard", "price": 12.99},
        {"subscription_id": 3, "tier_name": "Premium", "price": 17.99}
    ]
    pd.DataFrame(subscriptions).to_csv("data/sql/subscriptions.csv", index=False)

    # Users
    users = []
    for _ in range(num_users):
        uid = str(uuid.uuid4())
        users.append({
            "user_id": uid,
            "full_name": f"User {_}",
            "age": random.randint(18, 70),
            "city": random.choice(["New York", "Dallas", "San Jose", "Chicago"]),
            "country": "USA",
            "subscription_id": random.randint(1, 3)
        })
    pd.DataFrame(users).to_csv("data/sql/users.csv", index=False)

    # Payments
    payments = []
    for u in users:
        for _ in range(random.randint(1, 5)):
            payments.append({
                "payment_id": str(uuid.uuid4()),
                "user_id": u["user_id"],
                "amount": random.choice([7.99, 12.99, 17.99]),
                "payment_date": random_date().isoformat(),
                "method": random.choice(["Card", "PayPal", "UPI"])
            })
    pd.DataFrame(payments).to_csv("data/sql/payments.csv", index=False)

    return users


# -----------------------------
# 2. MONGO DATA
# -----------------------------
def generate_mongo_data(users):
    playback_docs = []
    review_docs = []

    for u in users:
        sessions = []
        for _ in range(random.randint(5, 20)):
            sessions.append({
                "session_id": str(uuid.uuid4()),
                "timestamp": random_date().isoformat(),
                "duration_seconds": random.randint(100, 4000),
                "device": random.choice(["Android", "iOS", "SmartTV", "Laptop"]),
                "buffer_rate": round(random.uniform(0.01, 0.3), 3),
                "errors": []
            })
        playback_docs.append({
            "user_id": u["user_id"],
            "sessions": sessions
        })

        # reviews
        for _ in range(random.randint(0, 3)):
            review_docs.append({
                "review_id": str(uuid.uuid4()),
                "user_id": u["user_id"],
                "movie_id": f"m{random.randint(1,300)}",
                "rating": random.uniform(1, 5),
                "text": "Sample review text...",
                "timestamp": random_date().isoformat()
            })

    with open("data/mongo/playback_logs.json", "w") as f:
        json.dump(playback_docs, f, indent=2)

    with open("data/mongo/reviews.json", "w") as f:
        json.dump(review_docs, f, indent=2)


# -----------------------------
# 3. GRAPH DATA (Neo4j style)
# -----------------------------
def generate_graph_data(num_movies=300):
    # Movie nodes
    movies = [{"movie_id": f"m{i}", "title": f"Movie {i}"} for i in range(num_movies)]
    pd.DataFrame(movies).to_csv("data/graph/movies.csv", index=False)

    # Actor, Director, Genre lists
    actors = [{"actor_id": f"a{i}", "name": f"Actor {i}"} for i in range(150)]
    directors = [{"director_id": f"d{i}", "name": f"Director {i}"} for i in range(40)]
    genres = [{"genre_id": f"g{i}", "name": f"Genre {i}"} for i in range(12)]

    pd.DataFrame(actors).to_csv("data/graph/actors.csv", index=False)
    pd.DataFrame(directors).to_csv("data/graph/directors.csv", index=False)
    pd.DataFrame(genres).to_csv("data/graph/genres.csv", index=False)

    # Edges CSV
    edges = []
    for m in movies:
        # One director
        edges.append({
            "start": random.choice(directors)["director_id"],
            "type": "DIRECTED",
            "end": m["movie_id"]
        })
        # Few actors
        for _ in range(random.randint(2, 5)):
            edges.append({
                "start": random.choice(actors)["actor_id"],
                "type": "ACTED_IN",
                "end": m["movie_id"]
            })
        # Genres
        edges.append({
            "start": m["movie_id"],
            "type": "HAS_GENRE",
            "end": random.choice(genres)["genre_id"]
        })

    pd.DataFrame(edges).to_csv("data/graph/edges.csv", index=False)


# -----------------------------
# 4. VECTOR DATA (Embeddings)
# -----------------------------
def generate_vector_data(num_movies=300):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    movie_ids = [f"m{i}" for i in range(num_movies)]
    descriptions = [f"This is the plot summary for movie {i}. The movie is about adventure and drama."
                    for i in range(num_movies)]

    embeddings = model.encode(descriptions, convert_to_numpy=True)

    np.save("data/vectors/movie_embeddings.npy", embeddings)

    with open("data/vectors/movie_ids.json", "w") as f:
        json.dump(movie_ids, f, indent=2)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ensure_dirs()

    users = generate_sql_data()
    generate_mongo_data(users)
    generate_graph_data()
    generate_vector_data()

    print("Mock data generation completed!")
