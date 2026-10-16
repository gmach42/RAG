"""Manage CLI config / output"""

from . import retrieval, answer_gen
from .retrieval import Retriever
from .answer_gen import Generator
from .index import build_index

# Naming: *_path = a file, *_dir = a directory,
# except save_directory which is the flag name imposed by the subject.
DEFAULT_RAW_DIR: str = "data/raw/"
DEFAULT_INDEX_DIR: str = "data/processed/"
DEFAULT_SEARCH_SAVE_DIRECTORY: str = "data/output/search_results/"
DEFAULT_ANSWER_SAVE_DIRECTORY: str = "data/output/search_results_and_answer/"
DEFAULT_MAX_CHUNK_SIZE: int = 2000


class CLI:
    """RAG command line interface."""

    @staticmethod
    def _check_k(k: int) -> int:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, gor {k!r}")
        return k

    @staticmethod
    def _check_max_chunk_size(value: object, maximum: int = 2000) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"max_chunk_size must be a positive integer, gor {value!r}")
        if value > maximum:
            raise ValueError(
                f"max_chunk_size must be <= {maximum}, got {value}")
        return value

    @staticmethod
    def _check_path(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path, got {value}")
        return value

    @staticmethod
    def _check_query(query: str) -> str:
        if isinstance(query, (list, tuple)):
            query = " ".join(str(part) for part in query)
        res = str(query).strip()
        if not res:
            raise ValueError("Query must not be empty")
        return res

    def index(self,
              max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
              raw_dir: str = DEFAULT_RAW_DIR,
              index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Index the src used by the llm to resolve the query and save it under the designed directory"""
        max_chunk_size = self._check_max_chunk_size(max_chunk_size)
        raw_dir = self._check_path(raw_dir, "raw_dir")
        index_dir = self._check_path(index_dir, "index_dir")
        build_index(raw_dir, index_dir, max_chunk_size)
        print(f"Indexation complete, indices saved under {index_dir}")

    def search(self,
               query: str,
               k: int = 10,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the top_k sources for a single query."""
        q = self._check_query(query)
        top_k = self._check_k(k)
        retriever = Retriever(index_dir)
        for source in retriever.search(q, top_k):
            print(source.model_dump_json(indent=2))

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 10,
                       save_directory: str = DEFAULT_SEARCH_SAVE_DIRECTORY,
                       index_dir: str = DEFAULT_INDEX_DIR) -> None:
        top_k = self._check_k(k)
        dataset_path = self._check_path(dataset_path, "dataset_path")
        save_directory = self._check_path(save_directory, "save_directory")
        index_dir = self._check_path(index_dir, "index_dir")
        path = retrieval.search_dataset(index_dir, dataset_path, top_k,
                                        save_directory)
        print(f"Saved student_seach_results to {path!r}")

    def answer(self,
               query: str,
               k: int = 10,
               index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """Print the answer for a single query"""
        q = self._check_query(query)
        top_k = self._check_k(k)
        retriever = Retriever(index_dir)
        srcs = retriever.search(q, top_k)
        generator = Generator()
        print(
            f"The answer for the query {q} is: \n{generator.answer(q, srcs)}")

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = DEFAULT_ANSWER_SAVE_DIRECTORY) -> None:
        student_search_results_path = self._check_path(
            student_search_results_path, "student_search_results_path")
        save_directory = self._check_path(save_directory, "save_directory")
        # TODO print(f"Loaded 100 questions ... Processed 100 of 100 questions")
        path = answer_gen.answer_dataset(student_search_results_path,
                                         save_directory)
        print(f"Saved student_search_results_path_and_answer to {path}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        student_search_results_path = self._check_path(
            student_search_results_path, "student_search_results_path")
        dataset_path = self._check_path(dataset_path, "dataset_path")
