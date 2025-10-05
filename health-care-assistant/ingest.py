import os
import pandas as pd
import minsearch

import os
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

DATA_PATH = os.getenv("DATA_PATH", "../Data_csvs/data_v1.csv")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "med-rag"
EMBEDDING_DIMENSIONALITY = 512


def load_index(data_path=DATA_PATH):
    # Load data
    df = pd.read_csv(data_path)

    documents = df.to_dict(orient="records")

    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL)

    # Initialize embedding model
    model = TextEmbedding("jinaai/jina-embeddings-v2-small-en")

    # Recreate collection
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY,
            distance=models.Distance.COSINE
        )
    )

    # Create and upload points
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

    print(f"Indexed {len(points)} documents in Qdrant")

    return client, model