import bm25s
import json
from pathlib import Path
from tqdm import tqdm
import time
from pydantic import BaseModel

K = 2000

# Create corpus
# corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a fish is a creature that lives in water and swims",
# ]

# # Tokenize the corpus
# corpus_tokens = bm25s.tokenize(corpus)

# # Index the corpus
# retriever = bm25s.BM25(corpus=corpus)
# retriever.index(corpus_tokens)

# # Query the corpus
# query = "What is an llm?"
# query_tokens = bm25s.tokenize([query])

# # Get top-k results as a tuple of (doc ids, scores)
# results, scores = retriever.retrieve(query_tokens, k=4)

# for i in range(results.shape[1]):
#     doc, score = results[0, i], scores[0, i]
#     print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# # You can save the arrays to a directory...
# retriever.save("vllm_index_bm25")

# # You can save the corpus along with the model
# retriever.save("vllm_index_bm25", corpus=corpus)

# # ...and load them when you need them
# reloaded_retriever = bm25s.BM25.load("vllm_index_bm25", load_corpus=True)
# # set load_corpus=False if you don't need the corpus


class Chunk(BaseModel):
    """A Basic Chunk pydantic model"""
    file_path: Path
    first_character_index: int
    last_character_index: int
    text: str


# === Read corpus ===

def list_corpus_files():
    pass


def read_file(path: Path) -> str | None:
    pass


# === Chunking ===

def chunk_python(text: str, file_path: str, max_chunk_size: int) -> list[Chunk]:
    """Chunk by k_max = 2000 to accomodate moulinette"""
    pass


def chunk_markdown(text: str, file_path: str, max_chunk_size: int) -> list[Chunk]:
    """Chunk a .md/.txt file according to titles (# ##) and paragraph"""
    pass


# === Tokenization ===

def tokenize(texts: list[str]) -> ...:
    """Transform texts into tokens for BM25"""
    pass


# === Construct ===

def build_index():
    """files -> chunks (tqdm) -> tokens -> BM25 -> save"""
    pass


def save_index():
    """Save BM25 index and chunks metadatas (json) in data/processed/"""
    pass


def load_index():
    """Reload index + chunks. Used by retrieval.py"""
    pass
