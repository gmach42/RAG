"""Command line interface of the RAG, exposed with Python Fire."""

from . import retrieval, answer_gen, evaluate
from .retrieval import Retriever
from .answer_gen import Generator
from .index import build_index
import shutil

# Naming: *_path = a file, *_dir = a directory,
# except save_directory which is the flag name imposed by the subject.
DEFAULT_RAW_DIR: str = "data/raw/"
DEFAULT_INDEX_DIR: str = "data/processed/"
DEFAULT_SEARCH_SAVE_DIRECTORY: str = "data/output/search_results/"
DEFAULT_ANSWER_SAVE_DIRECTORY: str = "data/output/search_results_and_answer/"
DEFAULT_MAX_CHUNK_SIZE: int = 2000

RED = "\x1b[31m"
GREEN = "\x1b[32m"
BLUE = "\x1b[34m"
YELLOW = "\x1b[33m"
WHITE = "\x1b[0m"
BOLD = "\033[1m"
RESET = "\033[22m"


class CLI:
    """RAG command line interface."""

    @staticmethod
    def _check_k(k: int) -> int:
        """Check that k is a positive integer."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        return k

    @staticmethod
    def _check_max_chunk_size(value: object, maximum: int = 2000) -> int:
        """Check that max_chunk_size is a positive integer <= maximum."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"max_chunk_size must be a positive integer, got {value!r}")
        if value > maximum:
            raise ValueError(
                f"max_chunk_size must be <= {maximum}, got {value}")
        return value

    @staticmethod
    def _check_path(value: object, name: str) -> str:
        """Check that a path argument is a non-empty string."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path, got {value}")
        return value

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
    def _print_centered(text: str, color: str = "", bold: str = "") -> None:
        """Print text centered in terminal, optionally colored."""
        width = shutil.get_terminal_size().columns
        print(f"{bold}{color}{text.center(width)}{RESET}{WHITE}")

    def index(self,
              max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
              raw_dir: str = DEFAULT_RAW_DIR,
              index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Chunk raw_dir and save the BM25 index under index_dir."""
        self._print_centered("=== INDEXATION ===", RED, BOLD)
        max_chunk_size = self._check_max_chunk_size(max_chunk_size)
        raw_dir = self._check_path(raw_dir, "raw_dir")
        index_dir = self._check_path(index_dir, "index_dir")
        build_index(raw_dir, index_dir, max_chunk_size)
        self._print_centered(
            f"Indexation complete, indices saved under {index_dir}\n", RED)

    def search(self,
               query: str,
               k: int = 10,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the top_k sources for a single query."""
        self._print_centered("=== SEARCH ===", YELLOW, BOLD)
        q = self._check_query(query)
        top_k = self._check_k(k)
        retriever = Retriever(index_dir)
        sources = retriever.search(q, top_k)
        if not sources:
            print("No source found for this query.")
        for source in sources:
            print(source.model_dump_json(indent=2))

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 10,
                       save_directory: str = DEFAULT_SEARCH_SAVE_DIRECTORY,
                       index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Search every question of a dataset and save the results."""
        self._print_centered("=== SEARCH_DATASET ===", YELLOW, BOLD)
        top_k = self._check_k(k)
        dataset_path = self._check_path(dataset_path, "dataset_path")
        save_directory = self._check_path(save_directory, "save_directory")
        index_dir = self._check_path(index_dir, "index_dir")
        path = retrieval.search_dataset(index_dir, dataset_path, top_k,
                                        save_directory)
        self._print_centered(f"Saved student_search_results to {path}\n",
                             YELLOW)

    def answer(self,
               query: str,
               k: int = 10,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the answer for a single query."""
        self._print_centered("=== ANSWER ===", GREEN, BOLD)
        q = self._check_query(query)
        top_k = self._check_k(k)
        retriever = Retriever(index_dir)
        srcs = retriever.search(q, top_k)
        generator = Generator()
        self._print_centered(
            f"The answer for the query '{q}' is: "
            f"\n{generator.answer(q, srcs)}\n", GREEN)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = DEFAULT_ANSWER_SAVE_DIRECTORY) -> None:
        """Answer every question of a search results file and save them."""
        self._print_centered("=== ANSWER_DATASET ===", GREEN, BOLD)
        student_search_results_path = self._check_path(
            student_search_results_path, "student_search_results_path")
        save_directory = self._check_path(save_directory, "save_directory")
        path = answer_gen.answer_dataset(student_search_results_path,
                                         save_directory)
        self._print_centered(
            f"Saved student_search_results_and_answer to {path}\n", GREEN)

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print our recall@k against a ground-truth dataset."""
        self._print_centered("=== EVALUATE ===", BLUE, BOLD)
        student_search_results_path = self._check_path(
            student_search_results_path, "student_search_results_path")
        dataset_path = self._check_path(dataset_path, "dataset_path")
        recalls, q = evaluate.evaluate(dataset_path,
                                       student_search_results_path)
        print(BLUE, "\nEvaluation Results:")
        print("===================")
        print(f"Questions evaluated: {q}")
        print(
            "\n".join(f"Recall@{k}: {r:.2f} ({r:.1%})"
                      for k, r in recalls.items()), "\n", WHITE)
