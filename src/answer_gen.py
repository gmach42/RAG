"""Generate answers with Qwen from the retrieved chunks."""

from transformers import pipeline
from .data_models import (MinimalAnswer, MinimalSource, StudentSearchResults,
                          StudentSearchResultsAndAnswer)
from pathlib import Path
from pydantic import ValidationError
from tqdm import tqdm

REFUSAL: str = "I don't know based on the vLLM sources."
MAX_ANSWER_SOURCES: int = 30


class Generator:
    """Qwen text-generation pipeline, loaded once and reused."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the LLM pipeline once."""
        self.pipe = pipeline(task="text-generation",
                             model=model_name,
                             device_map="auto")
        config = self.pipe.model.generation_config
        config.update(max_new_tokens=256,
                      do_sample=False,
                      temperature=None,
                      top_p=None,
                      top_k=None)
        self.pipe.generation_config = config

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Build the prompt from the sources and generate the answer."""
        context: str = ""
        for i, src in enumerate(sources[:MAX_ANSWER_SOURCES], start=1):
            entry = f"[{i}] {src.file_path}\n{src.text}\n\n"
            context += entry
        chat = [{
            "role":
            "system",
            "content":
            ("You are a documentation assistant for the vLLM library. "
             "You only answer questions about vLLM, "
             "using only the sources given by the user.")
        }, {
            "role":
            "user",
            "content":
            (f"Sources:\n{context}\n"
             f"Question: {question}\n\n"
             "Rules:\n"
             "- If the question is not about vLLM, or the sources do not "
             f"contain the answer, reply exactly: \"{REFUSAL}\"\n"
             "- Otherwise answer in a few sentences, using only the "
             "sources, and cite them like [1].\n"
             "/no_think")
        }]
        out = self.pipe(chat, clean_up_tokenization_spaces=False)
        # See https://huggingface.co/docs/transformers/conversations
        # for more explanation on text and clean_up_tokenization_spaces
        text: str = out[0]["generated_text"][-1]["content"]
        text = text.split("</think>")[-1]
        if REFUSAL in text:
            return REFUSAL
        return text.strip()


def load_search_results(
        student_search_results_path: str) -> StudentSearchResults:
    """Load a StudentSearchResults file (questions + retrieved chunks)."""
    path = Path(student_search_results_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"StudentSearchResults {student_search_results_path!r} "
            "does not exist.")
    try:
        results = StudentSearchResults.model_validate_json(
            path.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise ValueError(f"Malformed StudentSearchResults "
                         f"{student_search_results_path!r}: {e}") from e
    return results


def answer_questions(
        generator: Generator,
        results: StudentSearchResults) -> StudentSearchResultsAndAnswer:
    """Generate an answer for each question of the search results."""
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
                 student_search_results_path: str,
                 save_directory: str) -> Path:
    """Save the answers under save_directory, named after the input file."""
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / Path(student_search_results_path).name
    output_path.write_text(answer_results.model_dump_json(indent=4),
                           encoding="utf-8")
    return output_path


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> Path:
    """Answer every question of a search results file and save them."""
    results = load_search_results(student_search_results_path)
    generator = Generator()
    answers = answer_questions(generator, results)
    return save_answers(answers, student_search_results_path, save_directory)
