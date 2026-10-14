"""Once the source (vllm) is indexed (verified it is the case), research
the chunks (minimalSource) which respond the best to the query (prompt)
"""

from .index import load_index
from .data_models import MinimalSource


def is_indexed(output_dir: str) -> bool:
    """Check if the index exists in the given output directory"""
    try:
        load_index(output_dir)
        return True
    except (FileNotFoundError, ValueError):
        return False


def retrieve_chunks(output_dir: str, query: str) -> list[MinimalSource]:
    """Retrieve the chunks that best respond to the query using the BM25 index"""
    retriever, chunks = load_index(output_dir)
    tokenized_query = retriever.tokenize([query])[0]
    scores = retriever.score(tokenized_query)
    ranked_indices = sorted(range(len(scores)),
                            key=lambda i: scores[i],
                            reverse=True)
    return [chunks[i] for i in ranked_indices]


def retrieve_top_chunk(output_dir: str, query: str) -> MinimalSource | None:
    """Retrieve the top chunk that best responds to the query using the BM25 index"""
    ranked_chunks = retrieve_chunks(output_dir, query)
    return ranked_chunks[0] if ranked_chunks else None


def retrieve_top_chunks(output_dir: str, query: str,
                        top_k: int) -> list[MinimalSource]:
    """Retrieve the top_k chunks that best respond to the query using the BM25 index"""
    ranked_chunks = retrieve_chunks(output_dir, query)
    return ranked_chunks[:top_k] if ranked_chunks else []
