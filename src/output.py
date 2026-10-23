"""Command line interface of the RAG, exposed with Python Fire."""

from . import retrieval, answer_gen, evaluate
from .retrieval import Retriever
from .answer_gen import Generator
from .index import build_index
from pathlib import Path
import shutil

DEFAULT_RAW_DIR: str = "data/raw/"
DEFAULT_INDEX_DIR: str = "data/processed/"
DEFAULT_SEARCH_SAVE_DIRECTORY: str = "data/output/search_results/"
DEFAULT_ANSWER_SAVE_DIRECTORY: str = "data/output/search_results_and_answer/"
DEFAULT_MAX_CHUNK_SIZE: int = 2000
MIN_CHUNK_SIZE: int = 100
DEFAULT_K: int = 10

# Colors / Font in terminal
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


class CLI:
    """RAG command line interface."""

    @staticmethod
    def _check_k(k: int) -> int:
        """Check that k is a positive integer."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        return k

    @staticmethod
    def _check_max_chunk_size(value: object,
                              minimum: int = MIN_CHUNK_SIZE,
                              maximum: int = 2000) -> int:
        """Check that max_chunk_size is an integer in [minimum, maximum]."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"max_chunk_size must be a positive integer, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"max_chunk_size must be between {minimum} "
                             f"and {maximum}, got {value}")
        return value

    @staticmethod
    def _check_path(value: object, name: str) -> str:
        """Check that a path argument is a non-empty string."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path, got {value}")
        return value

    @staticmethod
    def _scoped_dir(base_dir: str, input_path: str) -> str:
        """Return base_dir/<scope>, scope = parent dir name of input_path."""
        return str(Path(base_dir) / Path(input_path).parent.name)

    @staticmethod
    def _check_query(query: str) -> str:
        """Join a query split by Fire and check that it is not empty."""
        if isinstance(query, (list, tuple)):
            query = " ".join(str(part) for part in query)
        res = str(query).strip()
        if not res:
            raise ValueError("Query must not be empty")
        return res

    @staticmethod
    def _print_centered(text: str, color: str = "") -> None:
        """Print text centered in terminal, optionally colored."""
        width = shutil.get_terminal_size().columns
        print()
        for line in text.splitlines():
            print(f"{color}{line.center(width)}{RESET}")

    def index(self,
              max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
              raw_dir: str = DEFAULT_RAW_DIR,
              index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Chunk raw_dir and save the BM25 index under index_dir."""
        max_chunk_size = self._check_max_chunk_size(max_chunk_size)
        raw_dir = self._check_path(raw_dir, "raw_dir")
        index_dir = self._check_path(index_dir, "index_dir")
        self._print_centered("=== INDEXATION ===\n\n", RED + BOLD)
        nb_chunks = build_index(raw_dir, index_dir, max_chunk_size)
        self._print_centered(
            f"Indexation complete, indexed {nb_chunks} chunks "
            f"under {index_dir}", RED)

    def search(self,
               query: str,
               k: int = DEFAULT_K,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the top_k sources for a single query."""
        q = self._check_query(query)
        top_k = self._check_k(k)
        index_dir = self._check_path(index_dir, "index_dir")
        self._print_centered("=== SEARCH ===\n\n", YELLOW + BOLD)
        retriever = Retriever(index_dir)
        sources = retriever.search(q, top_k)
        if not sources:
            raise ValueError("No source found for this query")
        for source in sources:
            src = source.model_dump(mode="json", exclude={"text"})
            print(
                f"{YELLOW}{src['file_path']} [{src['first_character_index']}:"
                f"{src['last_character_index']}]{RESET}")
        print()

    def search_dataset(self,
                       dataset_path: str,
                       k: int = DEFAULT_K,
                       save_directory: str | None = None,
                       index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Search every question of a dataset and save the results."""
        top_k = self._check_k(k)
        dataset_path = self._check_path(dataset_path, "dataset_path")
        if save_directory is None:
            save_directory = self._scoped_dir(DEFAULT_SEARCH_SAVE_DIRECTORY,
                                              dataset_path)
        save_directory = self._check_path(save_directory, "save_directory")
        index_dir = self._check_path(index_dir, "index_dir")
        self._print_centered("=== SEARCH_DATASET ===\n\n", YELLOW + BOLD)
        path = retrieval.search_dataset(index_dir, dataset_path, top_k,
                                        save_directory)
        self._print_centered(f"Saved student_search_results to {path}\n",
                             YELLOW)

    def answer(self,
               query: str,
               k: int = DEFAULT_K,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the answer for a single query."""
        q = self._check_query(query)
        top_k = self._check_k(k)
        index_dir = self._check_path(index_dir, "index_dir")
        self._print_centered("=== ANSWER ===\n\n", GREEN + BOLD)
        retriever = Retriever(index_dir)
        srcs = retriever.search(q, top_k)
        if not srcs:
            raise ValueError("No source found for this query")
        generator = Generator()
        print(f"{GREEN}\nThe answer for the query '{q}' is: "
              f"\n{generator.answer(q, srcs)}\n{RESET}")

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str | None = None) -> None:
        """Answer every question of a search results file and save them."""
        student_search_results_path = self._check_path(
            student_search_results_path, "student_search_results_path")
        if save_directory is None:
            save_directory = self._scoped_dir(DEFAULT_ANSWER_SAVE_DIRECTORY,
                                              student_search_results_path)
        save_directory = self._check_path(save_directory, "save_directory")
        self._print_centered("=== ANSWER_DATASET ===\n\n", GREEN + BOLD)
        path = answer_gen.answer_dataset(student_search_results_path,
                                         save_directory)
        self._print_centered(
            f"Saved student_search_results_and_answer to {path}\n", GREEN)

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print our recall@k against a ground-truth dataset."""
        student_search_results_path = self._check_path(
            student_search_results_path, "student_search_results_path")
        dataset_path = self._check_path(dataset_path, "dataset_path")
        self._print_centered("=== EVALUATE ===", BLUE + BOLD)
        recalls, q = evaluate.evaluate(dataset_path,
                                       student_search_results_path)
        results = "\n".join(f"Recall@{k}: {r:.2f} ({r:.1%})"
                            for k, r in recalls.items())
        text = ("Evaluation Results:\n===================\n"
                f"Questions evaluated: {q}\n" + results)
        self._print_centered(text, BLUE)
