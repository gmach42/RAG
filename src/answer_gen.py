"""Module for generating answers based on retrieved chunks.

Using libraries such as Olama to generate answers from the retrieved chunks.
Randomness is deactivated to ensure consistent answer generation.
Hallucinations are minimized but may still occur.
"""

from transformers import pipeline
from .data_models import RagDataset, MinimalAnswer, MinimalSource, MinimalSearchResults, UnansweredQuestion

# device_map allow to make use of the GPU if there's one available (else CPU)
pipe = pipeline(task="text-generation",
                model="Qwen/Qwen3-0.6B",
                device_map="auto")


def load_unsanwered_question() -> list[UnansweredQuestion]:
    pass  # TODO


def init_data_set() -> RagDataset:
    pass  # TODO


def get_minimal_search_results() -> MinimalSearchResults:
    """"""
    pass  # TODO


def get_minimal_answer(search_results: MinimalSearchResults,
                       query: str) -> MinimalAnswer:
    """Give the minimal answer from the model to the query constrained by the chunks selected"""
    pass  # TODO


messages = [
    {
        "role": "system",
        "content": "Answer only from the context. If it's not there, say so."
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question} /no_think"
    },
]
# do_sample=False ensures deterministic output (greedy)
out = pipe(messages, max_new_tokens=256, do_sample=False)
answer = out[0]["generated_text"][-1]["content"]
