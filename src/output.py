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
