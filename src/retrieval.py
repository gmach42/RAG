"""Once the source (vllm) is indexed (verified it is the case), research
the chunks (minimalSource) which respond the best to the query (prompt)
"""

import json
import os
from pathlib import Path
from .index import load_index, tokenize
from pydantic import ValidationError
from .data_models import (MinimalSource, UnansweredQuestion, RagDataset,
                          MinimalSearchResults, StudentSearchResults)


class Retriever:

    def __init__(self, index_dir: str) -> None:
        """Load the BM25 index (index + chunks)"""
        if not Path(index_dir).is_dir():
            raise FileNotFoundError(
                f"No index in {index_dir!r}, run the index command first.")
        self.bm25, self.chunks = load_index(index_dir)

    def search(self, query: str, top_k: int) -> list[MinimalSource]:
        k = min(top_k, len(self.chunks))
        if k <= 0 or not query.strip():
            return []
        tokens = tokenize([query])
        # Get the ids of the best chunks and their respective score
        chunk_ids, scores = self.bm25.retrieve(tokens,
                                               k=k,
                                               show_progress=False)
        results = []
        # Associate ids and score with zip then return the best chunks in order
        for i, score in zip(chunk_ids[0], scores[0]):
            if score > 0:
                results.append(self.chunk[i])
        return results


def load_questions(dataset_path: str) -> list[UnansweredQuestion]:
    """Load the questions of a RagDataset JSON file"""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_path!r} does not exist.")
    try:
        dataset = RagDataset.model_validate_json(
            path.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise ValueError(f"Malformed dataset {dataset_path!r}: {e}") from e
    return list(dataset.rag_questions)


def search_questions(retriever: Retriever, questions: list[UnansweredQuestion],
                     top_k: int) -> StudentSearchResults:
    """Retrieve the top_k chunks for each question"""
    results = [
        MinimalSearchResults(question_id=q.question_id,
                             question=q.question,
                             retrieved_sources=retriever.search(
                                 q.question, top_k))
        for q in tqdm(questions, desc="Searching")
    ]
    return StudentSearchResults(search_results=results, k=top_k)


def save_search_results(search_results: StudentSearchResults,
                        dataset_path: str, save_directory: str) -> Path:
    """Save the results under save_directory, named after the dataset"""
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name
    out_path.write_text(search_results.model_dump_json(indent=4),
                        encoding="utf-8")
    return out_path


def retrieve_and_save_results(source_dir: str, questions_file: str, top_k: int,
                              output_dir: str) -> StudentSearchResults:
    check_error(source_dir, questions_file, output_dir, top_k)
    questions = load_questions(questions_file)
    search_results = get_search_results(source_dir, questions, top_k)
    save_search_results(search_results, output_dir)
    return search_results
