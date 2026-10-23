"""Search the BM25 index for the chunks that best answer a query.

The index must have been built first (checked by load_index).
"""

from tqdm import tqdm
from pathlib import Path
from .index import load_index, tokenize
from pydantic import ValidationError
from .data_models import (MinimalSource, UnansweredQuestion, RagDataset,
                          MinimalSearchResults, StudentSearchResults)


class Retriever:
    """Load the index ONCE and then search it."""

    def __init__(self, index_dir: str) -> None:
        """Load the BM25 index (index + chunks)."""
        self.bm25, self.chunks = load_index(index_dir)

    def search(self, query: str, top_k: int) -> list[MinimalSource]:
        """Return the top_k chunks for the query, best BM25 score first.

        Chunks with a null score (no word in common) are left out.
        """
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
                results.append(self.chunks[i])
        return results


def load_questions(dataset_path: str) -> list[UnansweredQuestion]:
    """Load the questions of a RagDataset JSON file."""
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
    """Retrieve the top_k chunks for each question of the dataset."""
    results = [
        MinimalSearchResults(question_id=q.question_id,
                             question=q.question,
                             retrieved_sources=retriever.search(
                                 q.question, top_k))
        for q in tqdm(questions, desc="Searching", colour="yellow")
    ]
    return StudentSearchResults(search_results=results, k=top_k)


def save_search_results(search_results: StudentSearchResults,
                        dataset_path: str, save_directory: str) -> Path:
    """Save the results under save_directory, named after the dataset."""
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / Path(dataset_path).name
    output_path.write_text(search_results.model_dump_json(indent=4),
                           encoding="utf-8")
    return output_path


def search_dataset(index_dir: str, dataset_path: str, top_k: int,
                   save_directory: str) -> Path:
    """Search every question of a dataset and save the results.

    Returns the path of the saved StudentSearchResults file.
    """
    if top_k <= 0:
        raise ValueError(f"k must be a positive integer, got {top_k}.")
    input_path = Path(dataset_path)
    if (Path(save_directory) / input_path.name).resolve() == \
            input_path.resolve():
        raise ValueError(
            f"save_directory would overwrite the dataset {dataset_path!r}.")
    questions = load_questions(dataset_path)
    retriever = Retriever(index_dir)
    results = search_questions(retriever, questions, top_k)
    return save_search_results(results, dataset_path, save_directory)
