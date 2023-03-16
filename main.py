import openai
import os
from typing import List
import numpy as np

openai.organization = "org-qu4lCRmyoAcUXUOigj76QGnv"
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_embedding(
  text:str, 
  model:str = "text-embedding-ada-002"
) -> List[float]:
  embedding =  openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]
  return np.array(embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def vector_search(
  query:str, 
  embeddings:List[List[float]], 
  k:int=3,
  distance_metric: str = "cosine"
) -> List[int]:
  query_embedding = get_embedding(query)
  
  distances = []
  for i, item_embedding in enumerate(embeddings):
    if distance_metric == "cosine":
      cosine_distance = cosine_similarity(query_embedding, item_embedding)
      distances.append((i, cosine_distance))
  
  distances = sorted(distances, key=lambda x: x[1], reverse=True)[:k]
  top_k = [k[0] for k in distances]
  
  return top_k 

  
