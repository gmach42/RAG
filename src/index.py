import bm25s
import json
from pathlib import Path
from tqdm import tqdm
import time

K = 2000

# corpus = bm25s.load("vllm-0.10.1/")

# Create corpus
# corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a fish is a creature that lives in water and swims",
# ]

def chunk_file(path: Path, root: Path, size: int = K, overlap: int = 200) -> list[dict]:



# Tokenize the corpus
corpus_tokens = bm25s.tokenize(corpus)

# Index the corpus
retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

# Query the corpus
query = "What is an llm?"
query_tokens = bm25s.tokenize([query])

# Get top-k results as a tuple of (doc ids, scores)
results, scores = retriever.retrieve(query_tokens, k=4)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# You can save the arrays to a directory...
retriever.save("vllm_index_bm25")

# You can save the corpus along with the model
retriever.save("vllm_index_bm25", corpus=corpus)

# ...and load them when you need them
reloaded_retriever = bm25s.BM25.load("vllm_index_bm25", load_corpus=True)
# set load_corpus=False if you don't need the corpus
