"""Once the source (vllm) is indexed (verified it is the case), research
the chunks (minimalSource) which respond the best to the query (prompt)
"""

import json
import os
from .index import load_index, tokenize
from .data_models import MinimalSource, UnansweredQuestion, MinimalSearchResults, StudentSearchResults


def is_indexed(source_dir: str) -> bool:
    """Check if the index exists in the given output directory"""
    try:
        load_index(source_dir)
        return True
    except (FileNotFoundError, ValueError):
        return False


def get_chunks(source_dir: str, query: str, top_k: int) -> list[MinimalSource]:
    """Retrieve the chunks that best respond to the query using the BM25 index"""
    retriever, chunks = load_index(source_dir)
    tokenized_query = tokenize([query])[0]
    scores = retriever.retrieve(tokenized_query, k=top_k)
    ranked_indices = sorted(range(len(scores)),
                            key=lambda i: scores[i],
                            reverse=True)
    return [chunks[i] for i in ranked_indices]


def get_best_chunk(source_dir: str, query: str,
                   top_k: int) -> MinimalSource | None:
    """Retrieve the top chunk that best responds to the query using the BM25 index"""
    ranked_chunks = get_chunks(source_dir, query, top_k)
    return ranked_chunks[0] if ranked_chunks else None


def get_top_chunks(source_dir: str, query: str,
                   top_k: int) -> list[MinimalSource]:
    """Retrieve the top_k chunks that best respond to the query using the BM25 index"""
    ranked_chunks = get_chunks(source_dir, query, top_k)
    return ranked_chunks[:top_k] if ranked_chunks else []


def load_questions(file_path: str) -> list[UnansweredQuestion]:
    """Load unanswered questions from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        questions.append(
            UnansweredQuestion(question_id=item["question_id"],
                               question=item["question"]))
    return questions


def get_minimal_search_results(source_dir: str,
                               questions: list[UnansweredQuestion],
                               top_k: int) -> list[MinimalSearchResults]:
    """Retrieve the top_k chunks for each question and return the minimal search results"""
    minimal_search_results = []
    for query in questions:
        query.retrieved_chunks = get_top_chunks(source_dir,
                                                query.question,
                                                top_k=top_k)
        minimal_search_results.append(
            MinimalSearchResults(question_id=query.question_id,
                                 question=query.question,
                                 retrieved_sources=query.retrieved_chunks))
    return minimal_search_results


def get_search_results(source_dir: str, questions: list[UnansweredQuestion],
                       top_k: int) -> StudentSearchResults:
    return StudentSearchResults(
        student_search_results=get_minimal_search_results(
            source_dir, questions, top_k),
        k=top_k)


def save_search_results(search_results: StudentSearchResults,
                        output_dir: str) -> None:
    """Save the student search results to a JSON file in the specified output directory"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "student_search_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(search_results.model_dump_json(indent=4))


def check_error(source_dir: str, questions_file: str, output_dir: str,
                top_k: int) -> None:
    if not os.path.exists(questions_file):
        raise ValueError(f"Questions file '{questions_file}' does not exist.")
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not is_indexed(source_dir):
        raise ValueError(f"Source directory '{source_dir}' is not indexed.")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")


def retrieve_and_save_results(source_dir: str, questions_file: str, top_k: int,
                              output_dir: str) -> StudentSearchResults:
    check_error(source_dir, questions_file, output_dir, top_k)
    questions = load_questions(questions_file)
    search_results = get_search_results(source_dir, questions, top_k)
    save_search_results(search_results, output_dir)
    return search_results
