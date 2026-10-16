"""Manage CLI config / output"""
# TODO
"""
Commands to do:

- index -max_chunk_size <int>
    Ingest data/raw/ and build the index under data/processed/.
- search <query> -k <int>
    Return the top-k sources for a single query
- search_dataset -dataset_path <path> -k <int> -save_directory <dir>
    Run search over a whole dataset and write a StudentSearchResults JSON file.
- answer <query> -k <int>
    Answer a single query using the retrieved context
- answer_dataset -student_search_results_path <path> -save_directory <dir>
    Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file.
- evaluate -student_search_results_path <path> -dataset_path <path>
    Report your own recall@k against a ground-truth dataset, for your own testing.

All input and output paths must be configurable CLI arguments and never hard-coded.
Include progress bars (tqdm) for long-running operations, and handle degenerate inputs
(empty query, nonsensical query, k=0, missing files, malformed JSON) gracefully. The
CLI is tested with such edge cases and must never crash with an unhandled traceback.
"""

"""
uv run python -m src index --max_chunk)size 2000
uv run python -m src search_dataset
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
    --k 10
    --save_directory data/output/search_results/UnansweredQuestions
/moulinette evaluate_student_search_results
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json
    data/datasets/AnsweredQuestions/dataset_docs_public.json
    --k 10 --max_context_length 2000
uv run python -m src answer_dataset
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
"""

import fire


DEFAULT_RAW_PATH = "data/raw/"
DEFAULT_INDEX_PATH = "data/processed/"
DEFAULT_OUTPUT_PATH = "data/output/"

class CLI:
    def __init__(self) -> None:

