import json
from time import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import ingest
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

load_dotenv()

# OpenAI/OpenRouter client
api_key = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Load Qdrant index and embedding model
qdrant_client, embedding_model = ingest.load_index()

# Qdrant configuration
collection_name = "med-rag"


def search(query, k=5):
    """Search using Qdrant vector database"""
    query_embedding = list(embedding_model.embed([query]))[0]

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )

    results = []
    for result in search_results:
        results.append(result.payload)

    return results


prompt_template = """
You are a medical expert. Answer the QUESTION using only the information provided in the CONTEXT from our medical expertise database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()



def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + prompt_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model="openai/gpt-oss-120b"):  # Your model as default
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, token_stats


evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model="openai/gpt-oss-120b")  # Changed to your model

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == "openai/gpt-oss-120b":
        # $0.05 input, $0.27 output per 1M tokens
        openai_cost = (
                tokens["prompt_tokens"] * 0.00000005 +
                tokens["completion_tokens"] * 0.00000027
        )
    else:
        print(f"Model {model} not recognized. OpenAI cost calculation failed.")

    return openai_cost


def rag(query, model="openai/gpt-oss-120b"):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    openai_cost_rag = calculate_openai_cost(model, token_stats)
    openai_cost_eval = calculate_openai_cost(model, rel_token_stats)  # Same model now

    openai_cost = openai_cost_rag + openai_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "openai_cost": openai_cost,
    }

    return answer_data