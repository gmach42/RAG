"""Module for generating answers based on retrieved chunks.

Using libraries such as Olama to generate answers from the retrieved chunks.
Randomness is deactivated to ensure consistent answer generation.
Hallucinations are minimized but may still occur.
"""

from transformers import pipeline
from .data_models import RagDataset, MinimalAnswer, MinimalSource, MinimalSearchResults, UnansweredQuestion, StudentSearchResults, StudentSearchResultsAndAnswer
from pathlib import Path


class Generator:

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the LLM pipeline one"""
        self.pipe = pipeline(task="text-generation",
                             model=model_name,
                             device_map="auto")

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Build the prompt from the sources and generate the answer"""
        pass


def load_search_results(search_dir: str) -> StudentSearchResults:
    """Load the results (questions + chunks from src corresponding)"""
    # TODO
    pass


def answer_questions(
        generator: Generator,
        results: StudentSearchResults) -> StudentSearchResultsAndAnswer:
    # TODO
    pass


def save_answers(answer_results: StudentSearchResultsAndAnswer,
                 dataset_path: str, save_directory: str) -> Path:
    # TODO
    pass


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
