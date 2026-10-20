"""Compute our own recall@k against a ground-truth dataset."""

from .data_models import StudentSearchResults, AnsweredQuestion, MinimalSource
from .answer_gen import load_search_results
from .retrieval import load_questions
from statistics import mean

K_VALUES: list[int] = [1, 3, 5, 10]


def load_answered(dataset_path: str) -> list[AnsweredQuestion]:
    """Load the questions of a ground-truth dataset, with their sources."""
    questions = load_questions(dataset_path)
    answered = [q for q in questions if isinstance(q, AnsweredQuestion)]
    if not answered:
        raise ValueError(f"{dataset_path!r} has no answered questions.")
    return answered


def source_found(expected: MinimalSource, retrieved: MinimalSource) -> bool:
    """Return True if retrieved overlaps expected in the same file."""
    return (expected.file_path == retrieved.file_path
            and expected.first_character_index < retrieved.last_character_index
            and expected.last_character_index
            > retrieved.first_character_index)


def recall_at_k(expected: list[MinimalSource], retrieved: list[MinimalSource],
                k: int) -> float:
    """Return the fraction of expected sources found in the top-k."""
    # check div 0
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    found = 0
    for exp in expected:
        for ret in top_k:
            if source_found(exp, ret):
                found += 1
                break
    return found / len(expected)


def evaluate(dataset_path: str,
             student_search_results_path: str) -> tuple[dict[int, float], int]:
    """Return the mean recall@k for each k, and the number of questions."""
    # For each question, compute the proportion of expected sources found
    # in the top-k results, then average over all questions for each k.
    results: StudentSearchResults = load_search_results(
        student_search_results_path)
    answered: list[AnsweredQuestion] = load_answered(dataset_path)
    # Index the sources found for each question by its id
    results_by_id = {
        r.question_id: r.retrieved_sources
        for r in results.search_results
    }
    nb_entries = len(answered)
    recalls: dict[int, float] = {}
    for k in K_VALUES:
        recalls[k] = mean(
            recall_at_k(q.sources, results_by_id.get(q.question_id, []), k)
            for q in answered)
    return recalls, nb_entries
