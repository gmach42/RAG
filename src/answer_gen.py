"""Module for generating answers based on retrieved chunks.
"""

from transformers import pipeline
from .data_models import RagDataset, MinimalAnswer, MinimalSource, MinimalSearchResults, UnansweredQuestion, StudentSearchResults, StudentSearchResultsAndAnswer
from pathlib import Path
import bm25s
from pydantic import ValidationError
from tqdm import tqdm


class Generator:

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the LLM pipeline one"""
        self.pipe = pipeline(task="text-generation",
                             model=model_name,
                             device_map="auto")

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Build the prompt from the sources and generate the answer"""
        prompt: str = ""
        if not question:
            raise ValueError("no quesiton asked")
        prompt += f"User request: {question}\n"
        if not sources:
            raise ValueError(
                f"no sources has been found for this question {question}")
        prompt += f"Here's is documentation used to respond to it: {sources!r}"

        answer = pipeline(prompt, max_new_tokens=512)

        # TODO to fix / comprehend what qwen hope for + what it gives back

        return answer


def load_search_results(search_dir: str) -> StudentSearchResults:
    """Load the results (questions + chunks from src corresponding)"""
    path = Path(search_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"StudentSearchResults {search_dir!r} does not exist.")
    try:
        results = StudentSearchResults.model_validate_json(
            path.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise ValueError(
            f"Malformed StudentSearchResults {search_dir!r}: {e}") from e
    return results


def answer_questions(
        generator: Generator,
        results: StudentSearchResults) -> StudentSearchResultsAndAnswer:
    answers: list[MinimalAnswer] = []
    for result in tqdm(results.search_results, desc="Answering"):
        generated: str = generator.answer(result.question,
                                          result.retrieved_sources)
        answer = MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=generated,
        )
        answers.append(answer)
    return StudentSearchResultsAndAnswer(search_results=answers, k=results.k)


def save_answers(answer_results: StudentSearchResultsAndAnswer,
                 dataset_path: str, save_directory: str) -> Path:
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name
    out_path.write_text(answer_results.model_dump_json(indent=4),
                        encoding="utf-8")
    return out_path


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> Path:
    """Main function, same shape as search_dataset"""
    # TODO
    pass


# messages = [
#     {
#         "role": "system",
#         "content": "Answer only from the context. If it's not there, say so."
#     },
#     {
#         "role": "user",
#         "content": f"Context:\n{context}\n\nQuestion: {question} /no_think"
#     },
# ]
# # do_sample=False ensures deterministic output (greedy)
# out = pipe(messages, max_new_tokens=256, do_sample=False)
# answer = out[0]["generated_text"][-1]["content"]
