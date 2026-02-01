# Polyglot Project: Multimodal Engagement Network

A sophisticated machine learning project that combines multiple data sources (SQL, MongoDB, Graph Databases, and Vector Embeddings) to predict user engagement with movies using a multimodal deep learning architecture.

## Project Overview

This project demonstrates a **polyglot data architecture** where different types of data are stored and accessed from heterogeneous sources:

- **SQL Database**: User demographics and subscription data
- **MongoDB**: Playback logs and user reviews
- **Graph Database**: Movie relationships with actors, directors, and genres
- **Vector Database**: Movie embeddings for semantic similarity

All data sources are integrated into a **MultimodalEngagementNet** - a PyTorch neural network that fuses diverse data modalities to predict user engagement scores.

## Architecture

### Data Flow

```
User Data (SQL)           → Tabular Branch (MLP)
↓
Playback Sessions (MongoDB) → Sequence Branch (LSTM)
↓
Movie Embeddings (Vectors)  → Embedding Branch (Dense)
↓
Graph Features (Neo4j)      → Graph Branch (Dense)
↓
Fusion Layer → Engagement Score Prediction
```

### Model Architecture

The **MultimodalEngagementNet** consists of four specialized branches:

1. **Tabular Branch** (SQL Data)
   - Input: User age, city (encoded), subscription price
   - Architecture: 3 → 32 → 32 dimensions
   - Activation: ReLU

2. **Sequence Branch** (MongoDB Data)
   - Input: Last 5 playback session durations
   - Architecture: LSTM with hidden dimension 32
   - Captures temporal patterns in user behavior

3. **Embedding Branch** (Vector Data)
   - Input: 384-dim movie embeddings (all-MiniLM-L6-v2)
   - Architecture: 384 → 64 dimensions
   - Activation: ReLU

4. **Graph Branch** (Graph Data)
   - Input: Graph centrality score (number of connected entities)
   - Architecture: 1 → 16 dimensions
   - Activation: ReLU

5. **Fusion Layer**
   - Concatenates all branch outputs (32+32+64+16 = 144 dims)
   - Architecture: 144 → 64 → 1
   - Activation: Sigmoid (outputs engagement probability)

## Project Structure

```
polyglot-project/
├── README.md                      # This file
├── model_checkpoint.pth           # Trained model weights
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── model.py                   # MultimodalEngagementNet definition
│   ├── db_client.py               # Data loading from all sources
│   ├── dataset.py                 # PyTorch Dataset implementation
│   ├── train.py                   # Training script
│   └── data_generation.py         # Synthetic data generator
│
├── data/                          # Data directory
│   ├── sql/                       # Relational data
│   │   ├── users.csv              # User demographics
│   │   ├── subscriptions.csv      # Subscription tiers
│   │   └── payments.csv           # Payment records
│   │
│   ├── mongo/                     # Document data
│   │   ├── playback_logs.json     # User viewing sessions
│   │   └── reviews.json           # User reviews
│   │
│   ├── graph/                     # Graph relationships
│   │   ├── movies.csv
│   │   ├── actors.csv
│   │   ├── directors.csv
│   │   ├── genres.csv
│   │   └── edges.csv              # Graph connections
│   │
│   └── vectors/                   # Embedding data
│       ├── movie_embeddings.npy   # 300×384 numpy array
│       └── movie_ids.json         # Movie ID mapping
│
├── docs/                          # Documentation
│   ├── sql_schema.sql             # SQL table definitions
│   ├── graph_schema.txt           # Graph node/relationship types
│   ├── vector_strategy.txt        # Vector preprocessing & indexing
│   ├── mongo_playback_example.json
│   └── mongo_reviews.json
│
└── notebooks/                     # Jupyter notebooks
    └── demo.ipynb                 # Interactive demo (template)
```

## Data Schema

### SQL Database

**Users Table**
- `user_id` (UUID): Unique user identifier
- `full_name` (VARCHAR): User's name
- `age` (INT): User's age
- `city` (VARCHAR): User's city
- `country` (VARCHAR): User's country
- `subscription_id` (INT FK): Reference to subscription tier

**Subscriptions Table**
- `subscription_id` (INT PK): Tier identifier
- `tier_name` (VARCHAR): "Basic", "Standard", "Premium"
- `price` (DECIMAL): Monthly subscription cost

**Payments Table**
- `payment_id` (UUID): Payment transaction ID
- `user_id` (UUID FK): Reference to user
- `amount` (DECIMAL): Payment amount
- `payment_date` (TIMESTAMP): When payment was made
- `method` (VARCHAR): Payment method (Card, PayPal, UPI)

### MongoDB Collections

**playback_logs** (Document per user)
```json
{
  "user_id": "uuid",
  "sessions": [
    {
      "session_id": "uuid",
      "timestamp": "ISO-8601",
      "duration_seconds": 2400,
      "device": "iOS|Android|SmartTV|Laptop",
      "buffer_rate": 0.15,
      "errors": []
    }
  ]
}
```

**reviews** (Individual review documents)
```json
{
  "review_id": "uuid",
  "user_id": "uuid",
  "movie_id": "m42",
  "rating": 4.5,
  "text": "Sample review text",
  "timestamp": "ISO-8601"
}
```

### Graph Database (Neo4j-style)

**Node Types**
- `:User` - User entities
- `:Movie` - Movie entities
- `:Actor` - Actor entities
- `:Director` - Director entities
- `:Genre` - Genre categories

**Relationships**
- `(:User)-[:FOLLOWS]->(:User)` - User following
- `(:User)-[:WATCHED]->(:Movie)` - User watched movie
- `(:User)-[:RATED {score}]->(:Movie)` - User rating with score
- `(:Actor)-[:ACTED_IN]->(:Movie)` - Actor in movie
- `(:Director)-[:DIRECTED]->(:Movie)` - Director of movie
- `(:Movie)-[:HAS_GENRE]->(:Genre)` - Movie genre

### Vector Data

**Movie Embeddings**
- Format: NumPy array (300 movies × 384 dimensions)
- Model: `all-MiniLM-L6-v2` (Sentence Transformers)
- Storage: `movie_embeddings.npy`
- Preprocessing: Lowercase, punctuation removal, 256-token truncation

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support optional)
- pandas, numpy
- sentence-transformers

### Installation

1. **Clone and setup virtual environment:**
   ```bash
   cd polyglot-project
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install torch pandas numpy sentence-transformers
   ```

3. **Generate synthetic data:**
   ```bash
   python src/data_generation.py
   ```
   This creates all required CSV and JSON files in the `data/` directory.

### Training the Model

```bash
python src/train.py
```

**What happens:**
- Loads data from all sources (SQL, MongoDB, Graph, Vectors)
- Creates a MultimodalDataset with 2000 random (user, movie) pairs
- Initializes the MultimodalEngagementNet
- Trains for 1 epoch with batch size 32
- Uses Adam optimizer (lr=0.001) and MSELoss
- Saves trained weights to `model_checkpoint.pth`

**Expected output:**
```
Using device: cuda  (or cpu)
Batch 0, Loss: 0.2834
Batch 50, Loss: 0.2345
...
Epoch Completed — Avg Loss: 0.2412
Model saved as model_checkpoint.pth
```

## Key Components

### [model.py](src/model.py) - Neural Network Architecture

Defines the complete multimodal architecture with 4 specialized branches and a fusion layer.

**Key Classes:**
- `TabularBranch`: MLP for structured data
- `SequenceBranch`: LSTM for temporal sequences
- `EmbeddingBranch`: Dense network for high-dim vectors
- `GraphBranch`: Dense network for graph features
- `MultimodalEngagementNet`: Fusion model combining all branches

### [db_client.py](src/db_client.py) - Data Integration Layer

Provides unified interface to access data from all sources:
- `get_user_tabular()`: Fetch demographics from SQL
- `get_last_sessions()`: Fetch viewing history from MongoDB
- `get_movie_embedding()`: Fetch semantic embedding from vectors
- `get_graph_feature()`: Compute graph-based features

### [dataset.py](src/dataset.py) - PyTorch Dataset

`MultiModalDataset` class that:
- Creates random (user, movie) pairs
- Fetches features from all data sources
- Formats data as PyTorch tensors
- Generates synthetic engagement targets

### [train.py](src/train.py) - Training Loop

Complete training pipeline:
- Device detection (GPU/CPU)
- DataLoader creation with batch processing
- Model initialization and optimizer setup
- Single epoch training with loss tracking
- Model checkpoint saving

### [data_generation.py](src/data_generation.py) - Synthetic Data Generator

Generates realistic mock data across all modalities:
- **SQL**: 500 users with demographics and subscription info
- **MongoDB**: Playback sessions and reviews
- **Graph**: 300 movies with actors, directors, genres
- **Vectors**: 384-dim embeddings for all movies

## Use Cases

This architecture is ideal for:

1. **Streaming Platform Engagement Prediction** - Predict which movies users will watch
2. **Recommendation Systems** - Score user-movie compatibility
3. **Churn Prediction** - Identify at-risk subscribers
4. **Content Ranking** - Personalize content feed
5. **A/B Testing** - Measure feature impact on engagement

## Extending the Project

### Adding New Data Sources

1. Create a new branch class in `model.py`
2. Add data loading function in `db_client.py`
3. Update `MultiModalDataset.__getitem__()` to include new data
4. Modify `MultimodalEngagementNet.forward()` to fuse new branch

### Hyperparameter Tuning

Edit `train.py`:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate
loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Batch size
```

Edit `dataset.py`:
```python
dataset = MultiModalDataset(users, movies, length=2000)  # Dataset size
```

### Model Architecture Changes

Edit branch dimensions in `model.py`:
```python
class TabularBranch(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):  # Adjust hidden_dim
```

## Performance Metrics

- **Model Size**: ~50K parameters
- **Batch Processing**: ~2000 samples per epoch
- **Training Time**: ~1 min per epoch (CPU), <10s (GPU)
- **Memory Footprint**: ~100MB (with data)

## Documentation Files

- [sql_schema.sql](docs/sql_schema.sql) - SQL table definitions
- [graph_schema.txt](docs/graph_schema.txt) - Graph database schema
- [vector_strategy.txt](docs/vector_strategy.txt) - Embedding preprocessing
- [mongo_playback_example.json](docs/mongo_playback_example.json) - Sample MongoDB document
- [mongo_reviews.json](docs/mongo_reviews.json) - Sample review data
- [graph_queries.cypher](docs/graph_queries.cypher) - Neo4j query examples

## Key Features

**Polyglot Data Architecture** - Seamlessly integrates SQL, NoSQL, Graph, and Vector databases

**Multimodal Learning** - Combines diverse data types through dedicated neural branches

**Synthetic Data Generation** - Realistic mock data for development and testing

**Modular Design** - Easy to extend with new data sources and model components

**PyTorch-based** - Industry-standard framework for deep learning

**GPU-ready** - Automatic CUDA detection and support

## 🔍 Project Highlights

- **4 Data Modalities**: Demonstrates real-world polyglot data architecture
- **Fusion Architecture**: Shows effective way to combine heterogeneous data
- **Production-Ready Code**: Clean, well-structured, documented code
- **Scalable Design**: Easy to add more data sources and model branches

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Feel free to extend this project by:
- Adding new data sources
- Experimenting with different architectures
- Improving data generation realism
- Creating additional analysis notebooks

---

**Last Updated**: January 2026

**Project Status**: Complete - Ready for training and experimentation
