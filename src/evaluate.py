# student_search_results_path <path> -dataset_path <path>

from .data_models import RagDataset, StudentSearchResults, MinimalSearchResults, AnsweredQuestion, MinimalSource
from .answer_gen import load_search_results
from .retrieval import load_questions
from statistics import mean

K_VALUES: list[int] = [1, 3, 5, 10]


def load_answered(dataset_path: str) -> list[AnsweredQuestion]:
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
             student_search_results_path: str) -> dict[int, float]:
    # Load rag_dataset and student search results
    # Associate questions by question_id
    # For each question calculate the proportion of sources expected found in the k top result
    # Calculate an average for all questions for k = 1,3,5 and 10, display it same format as moulinette
    # r = results.search_results: list[MinimalSearchResults], r.retrieved_sources: list[MinimalSource]
    results: StudentSearchResults = load_search_results(
        student_search_results_path)
    answered: list[AnsweredQuestion] = load_answered(dataset_path)
    results_by_id = {
        r.question_id: r.retrieved_sources
        for r in results.search_results
    }
    # recalls = {}
    # for k in K_VALUES:
    #     recalls = {
    #         k:
    #         mean(
    #             recall_at_k(q.sources, results_by_id.get(q.question_id, []), k)
    #             for q in answered)
    #     }
    recalls = {
        k:
        mean(
            recall_at_k(q.sources, results_by_id.get(q.question_id, []), k)
            for q in answered)
        for k in K_VALUES
    }
    return recalls
