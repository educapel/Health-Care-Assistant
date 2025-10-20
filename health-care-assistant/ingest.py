import os
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

DATA_PATH = os.getenv("DATA_PATH", "../Data_csvs/data_v1.csv")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "med-rag"
EMBEDDING_DIMENSIONALITY = 512


def load_index(data_path=DATA_PATH, force_reindex=False):
    """
    Load Qdrant client and embedding model.
    Only re-index if collection doesn't exist or force_reindex=True
    """
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL)

    # Initialize embedding model
    model = TextEmbedding("jinaai/jina-embeddings-v2-small-en")

    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
    except Exception as e:
        print(f"Error checking collections: {e}")
        collection_exists = False

    # If collection exists and not forcing reindex, just return
    if collection_exists and not force_reindex:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists with data")
        print(f"   Skipping re-indexing (use force_reindex=True to recreate)")
        return client, model

    # Only index if needed
    print(f"📥 {'Re-indexing' if collection_exists else 'Creating'} collection '{COLLECTION_NAME}'...")

    # Load data
    df = pd.read_csv(data_path)
    documents = df.to_dict(orient="records")

    # Delete existing collection if it exists
    if collection_exists:
        print(f"🗑️  Deleting existing collection...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    # Create collection
    print(f"📦 Creating collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY,
            distance=models.Distance.COSINE
        )
    )

    # Create and upload points
    print(f"⏳ Indexing {len(documents)} documents...")
    points = []
    for idx, doc in enumerate(documents):
        # Generate embedding from Answer field
        embedding = list(model.embed([doc['Answer']]))[0]

        point = models.PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "doc_id": doc['id'],
                "topic": doc['topic'],
                "Question": doc['Question'],
                "Answer": doc['Answer']
            }
        )
        points.append(point)

    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

    print(f"✅ Successfully indexed {len(points)} documents in Qdrant")

    return client, model