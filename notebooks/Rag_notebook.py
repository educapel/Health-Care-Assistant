#%%
import pandas as pd
import minsearch
import os
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch
import json
from tqdm.auto import tqdm
import time
from fastembed import TextEmbedding
from huggingface_hub import login
from qdrant_client import QdrantClient, models


#%%
path = '../Data_csvs/data.csv'
df = pd.read_csv(path)
df.columns
#%%
## transforming document into dicc format
documents = df.to_dict(orient='records')
documents[5:10]

#%%
len(documents)
#%% md
# # Ingestion- Retrieval
#%% md
# ### Minsearch
#%%
# Create index
index = minsearch.Index(
    text_fields=['chunk_text', 'question', 'topic'],
    keyword_fields=['doc_id']  # Filterable fields 'chunk_index'
)

index.fit(documents)

# Search examples
results = index.search('childhood cancer treatments')
#%%
results
#%%
question1 = 'What are these three main ways that skin cancer can spread from the original location to other parts of my body?'
question2 = 'What exactly are the high-risk features that determine how nonmelanoma skin cancer is staged?'
question3 = 'how can i diagnose Lung cancer?'

question4 = 'What are the latest therapies for Pancreatic Cancer?'
question5 = 'How can Colon Cancer be prevented?'
question6 = 'How to reduce the risk of Prostate Cancer?'

#%%
start_time = time.time()
results = index.search(question1, num_results=5)
end_time = time.time()
elapsed = end_time - start_time

print(f"Query took {elapsed:.4f} seconds")
print(results)
#%%
## boosting
boost = {'question': 3.0, 'topic': 0.5}

index.search(
    query= question1,
    boost_dict = boost,
    num_results=5
)

#%% md
# ## Elastic search
# 
#%%
#generating the json
json_list = df.to_dict(orient='records')
with open('../Data_csvs/documents_ids.json', 'w', encoding='utf-8') as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)
#%%
##using it
with open('../Data_csvs/documents_ids.json', 'rt') as f_in:
    documents = json.load(f_in)
documents
#%%
import elasticsearch
print(elasticsearch.__version__)
#%%
es = Elasticsearch(
    "http://localhost:9200",
)
es. info()

#%%
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "medical topic": {"type": "text"},
            "question": {"type": "text"},
            "id": {"type": "keyword"},
        }
    }
}

index_name = "medial-questions-answers"

es.indices.delete(index=index_name, ignore_unavailable=True)
es.indices.create(index=index_name, body=index_settings)
#%%
for doc in tqdm(documents):
    es.index(index=index_name, document=doc)
#%%
def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "chunk_text", "topic"],
                        "type": "best_fields"
                    }
                },

            }
        }
    }

    response = es.search(index=index_name, body=search_query)

    result_docs = []

    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
#%%
start_time = time.time()
elastic_search(
    query=question3,
)
end_time = time.time()

elapsed = end_time - start_time
print(f"Query took {elapsed:.4f} seconds")
#%% md
# ## Semantic search with Elastic search
#%% md
# to create embeddings for our dataset we are going to choose to do embeddings in the chunk size field since medical topic doesn't offer a lot of information
# 
#%% md
# #### Choosing embedding model (
#%%
TextEmbedding.list_supported_models()

#%%
EMBEDDING_DIMENSIONALITY = 512

for model in TextEmbedding.list_supported_models():
    if model["dim"] == EMBEDDING_DIMENSIONALITY:
        print(json.dumps(model, indent=2))

model_handle = "jinaai/jina-embeddings-v2-small-en"

#%%
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#%%
load_dotenv()
login(token=token)
#%%
model_handle = TextEmbedding("jinaai/jina-embeddings-v2-small-en")
model_handle
#%% md
# #### Creating embedding with pretrained model
#%%
embedding = []
model_handle = TextEmbedding("jinaai/jina-embeddings-v2-small-en")

for doc in tqdm(documents):
    embeddings = list(model_handle.embed([doc['chunk_text']]))
    doc['chunk_text_embedded'] = embeddings[0]
    embedding.append(doc)

print(embedding)
#%%
embedding
#%% md
# ### Creating mapping and index
#%%
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "topic": {"type": "text"}, # data types
            "question": {"type": "text"},
            "original_answers": {"type": "keyword"},
            "chunk_text": {"type": "keyword"},
            "chunk_text_embedded": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}
#%%
index_name= 'embedded_text'
es.indices.delete(index=index_name, ignore_unavailable=True)
es.indices.create(index=index_name, body=index_settings) #when you update your data you need to delete the index with the index name selected
#%% md
# ### Add documents to into the index
#%%
for doc in embedding:
    try:
        es.index(index=index_name, document=doc)
    except Exception as e:
        print (e)
#%% md
# ### New search query
#%%
search_term = 'How can Stomach Cancer be prevented?'
vector_search_term = list(model_handle.embed(search_term))[0]
print(vector_search_term)
#%%
query = {
    'field': 'chunk_text_embedded',
    'query_vector' : vector_search_term,
    'k':5,
    'num_candidates': 10000
}
#%%
res = es.search(index = index_name, knn = query, source=['topic', 'question', 'original_answers'])
res['hits']['hits']
#%% md
# ## Vector search: Quadrant
#%%
client = QdrantClient("http://localhost:6333")
#%% md
# ### Creating collection
#%%
collection_name = "med-rag"

# Create the collection with specified vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSIONALITY,
        distance=models.Distance.COSINE  # Distance metric for similarity search
    )
)
#%% md
# ### Creating indexes
#%%
model_handle = "jinaai/jina-embeddings-v2-small-en"

#%%
points = []
id = 0
for doc in documents:

    point = models.PointStruct(
        id=id,
        vector=models.Document(text=doc['chunk_text'], model=model_handle),
        payload={
            "doc_id": doc['doc_id'],
            "chunk_index": doc['chunk_index'],
            "topic": doc['topic'],
            "question": doc['question'],
            "chunk_text": doc['chunk_text']
        }
    )
    points.append(point)
    id += 1
#%% md
# ### Creating dataset
# 
#%%
client.upsert(
    collection_name=collection_name,
    points=points
)
#%%
###
def search(query, limit=1):

    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )

    return results
#%%
import random

medical_answ = random.choice(documents)
course_piece = random.choice(medical_answ['documents'])
print(json.dumps(course_piece, indent=2))
#%%
result = search(course_piece['question'])
print(f"Question:\n{course_piece['question']}\n")
print("Top Retrieved Answer:\n{}\n".format(result.points[0].payload['text']))
print("Original Answer:\n{}".format(course_piece['text']))
#%% md
# # LMM with no Augmented Retrieval
#%%
load_dotenv()

# Get the API key
api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key found: {api_key is not None}")
#%%
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

response = client.chat.completions.create(
  extra_body={},
  model="deepseek/deepseek-chat-v3.1:free",
  messages=[
    {
      "role": "user",
      "content": question3
    }
  ]
)
print(response.choices[0].message.content)
#%%
def search(query):
    boost = {}
    results = index.search(
        query = question6,
        filter_dict={},
        boost_dict=boost,
        num_results=5
    )
    return results

#%%
prompt_template = """
You are a medical expert. Answer the QUESTION using only the information provided in the CONTEXT from our medical expertise database.
Use only the facts from the CONTEXT when answering the QUESTION.


QUESTION: {question}

CONTEXT:
{context}
""".strip()
entry_template = """
medical_topic: {topic}
medical expertise database ; {chunk_text}

""".strip()

def build_prompt(query, search_results):

    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
#%%
search_results = search(question3)
search_results
#%%
prompt = build_prompt(question3, search_results)
print(prompt)
#%%
def llm(prompt):
    response = client.chat.completions.create(
      extra_body={},
      model="deepseek/deepseek-chat-v3.1:free",
      messages=[
        {
          "role": "user",
          "content": prompt
        }
      ]
    )
    return response.choices[0].message.content

#%%
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

answer= rag(question3)
print(answer)