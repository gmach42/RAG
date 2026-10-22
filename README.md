*This project has been created as part of the 42 curriculum by gmach.*

# Description

**RAG against the machine** is a Retrieval-Augmented Generation system that
answers questions about the [vLLM](https://github.com/vllm-project/vllm)
codebase (version 0.10.1, in `data/raw/`).

It works in four stages, each one feeding the next:

1. **Index**: split every `.py`, `.md` and `.txt` file into chunks of at most
   2000 characters and build a BM25 index under `data/processed/`.
2. **Search**: for a question, return the top-k chunks as source locations
   (`file_path` + character range).
3. **Answer**: give these chunks to **Qwen3-0.6B**, which answers using only
   them, or refuses when they do not contain the answer.
4. **Evaluate**: measure the retrieval quality with recall@k against
   reference datasets.

Everything is driven by a CLI built with Python Fire.

# Instructions

**Requirements:** Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).
A GPU is used when available, otherwise everything runs on CPU.

```bash
uv sync        # install every dependency (torch, transformers, bm25s, ...)
```

Every command is run as `uv run python -m src <command> [options]`:

| Command | What it does |
|---|---|
| `index --max_chunk_size 2000` | Chunk `data/raw/` and save the index in `data/processed/` |
| `search "<query>" --k 10` | Print the top-k sources for one query |
| `search_dataset --dataset_path <file> --k 10 --save_directory <dir>` | Search a whole dataset, write a `StudentSearchResults` JSON |
| `answer "<query>" --k 10` | Answer one query from the retrieved sources |
| `answer_dataset --student_search_results_path <file> --save_directory <dir>` | Answer a whole search results file, write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path <file> --dataset_path <file>` | Print our own recall@1/3/5/10 against an `AnsweredQuestions` dataset |

The **Makefile** wraps them: `make index`, `make search QUERY="..."`,
`make search-dataset`, `make evaluate`, `make answer-dataset`, `make run`
(the whole pipeline) and `make lint` (flake8 + mypy). Paths and `K` can be
overridden, e.g. `make search-dataset K=5 DATASET=<file>`.

Wrong inputs (empty query, `k=0`, missing or malformed file, missing index…)
print a one-line `Error: ...` and exit with code 1, never a traceback.

# System architecture

![RAG schema](RAG_schema.png)

| Module | Role |
|---|---|
| `data_models.py` | Pydantic models exchanged between the stages |
| `index.py` | Read the corpus, chunk it, build / save / load the BM25 index |
| `retrieval.py` | `Retriever` (index loaded once) and `search_dataset` |
| `answer_gen.py` | `Generator` (Qwen loaded once) and `answer_dataset` |
| `evaluate.py` | recall@k against a ground-truth dataset |
| `output.py` | CLI: argument checks, calls, printing |
| `__main__.py` | Fire entry point, turns every error into a clean message |

# Chunking strategy

Every chunk is a `MinimalSource` whose text is exactly
`file[first_character_index:last_character_index]` (checked by a pydantic
validator), so a result always points to a real place in the corpus.

**Markdown / text (`.md`, `.txt`): by sections.**
1. Cut before each title line (`#`, `##`, ...), ignoring the `#` inside
   ```` ``` ```` code blocks.
2. Merge consecutive sections as long as the result stays ≤ `max_chunk_size`,
   so that short sections do not become tiny chunks.
3. Split a section still longer than `max_chunk_size` into fixed-size pieces.

In `.txt` files (`CMakeLists.txt`, `requirements*.txt`), `#` starts a
comment: chunks then begin right before a comment, which usually introduces
the block that follows.

**Python (`.py`): fixed-size windows of `max_chunk_size` characters.**
This looks naive, but it was measured as the best choice: the expected
sources of the code dataset are arbitrary windows of ~900 characters that
often start in the middle of a function. Cutting along `def` / `class`
lines was tested and gave a *lower* recall@5 (55.6 % vs 59.6 %).

No overlap between chunks: it doubled the number of chunks, filled the top-k
with near-duplicates and did not improve recall.

# Retrieval method

**BM25** with the [`bm25s`](https://github.com/xhluca/bm25s) library and
English stopwords.

- **File path words are indexed with each chunk.** Before indexing, the words
  of the path are added in front of the chunk text:
  `fused_moe/fused_batched_moe.py` → `fused moe fused batched moe py`.
  A question about "the fused batched MoE layer" then matches the file name,
  even when these words are not in the chunk itself. Only the indexed text
  changes: the stored chunks and the context given to Qwen stay the same.
- Chunks with a null score (no word in common with the query) are dropped,
  so a nonsensical query returns no source instead of random ones.
- The index is saved once (`bm25s` files + `chunks.json`, where the i-th
  chunk is the i-th BM25 document) and reloaded by every search:

![Saving/Loading index](save_load_index.png)

**Answer generation.** The prompt lists the numbered sources, then the
question, then the rules: answer only from the sources and cite them, or
reply exactly `I don't know based on the vLLM sources.` when the question is
not about vLLM or the sources do not answer it. If the answer contains the refusal
sentence, only the refusal is kept: Qwen sometimes refuses after a made-up
start. Decoding is greedy (same answer on every run, less invented content).
At most 30 sources are given to Qwen: the biggest chunk takes about 1300
tokens, so they always fit in its 40960-token window (`k=10` takes about
5000).

# Performance analysis

Results of the moulinette on the public datasets (`k=10`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required (recall@5) |
|---|---|---|---|---|---|
| docs (100 questions) | 61.0 % | 82.0 % | **87.0 %** | 90.0 % | 80 % ✅ |
| code (99 questions) | 45.5 % | 64.6 % | **69.7 %** | 75.8 % | 50 % ✅ |

What each idea brought (recall@5, same corpus and datasets):

| Variant | docs | code |
|---|---|---|
| Sections for `.md`, fixed windows for `.py` | 85.0 % | 59.6 % |
| **+ file path words (kept)** | **87.0 %** | **69.7 %** |
| `.py` cut along `def` / `class` instead | 85.0 % | 55.6 % |
| First line + English stemmer | +1 pt | +0 pt |
| + splitting `snake_case` identifiers into words | 81.0 % | 80.8 % |

Splitting identifiers is the best idea for code questions, but it brings the
docs recall down to 81 %, too close to the threshold, with a single index for
both datasets: it was not kept.

**Speed:** indexing the whole corpus (11,157 chunks) takes about 3 s, a
search about 0.1 s, and an answer about 2 s on GPU (about 25 s on
CPU, mostly spent reading the 5000-token context).

Our `evaluate` command counts a source as found as soon as a chunk overlaps
it, while the moulinette is a bit stricter (it seems to require an overlap of
about 5 % of the union of both ranges), so `evaluate` reports slightly higher
numbers (89 % vs 87 % recall@5 on docs).

# Design decisions

- **Lexical search, file names included.** A file often deals with a topic
  without repeating its name in every chunk, while the file name is usually
  explicit: indexing the path words was the most effective change of the
  project (+10 points on code).
- **Load once, use many times.** `Retriever` keeps the index and `Generator`
  keeps Qwen in memory, so `search_dataset` and `answer_dataset` load them a
  single time for 100 questions.
- **`transformers` for generation.** The reviewer only runs `uv sync`, so the
  model has to come from a pip-installable library (no external server such
  as Ollama).
- **`MinimalSource` carries the chunk text.** The subject allows extra
  fields; keeping the text avoids re-reading the files at search and answer
  time. It is optional, since the reference datasets have no text.
- **One error handler.** Functions raise exceptions with a clear message and
  `__main__.py` prints them, so no traceback ever reaches the user.
- **Naming:** `*_path` is a file, `*_dir` a directory (except
  `save_directory`, imposed by the subject).

# Challenges faced

- **Understanding the data models.** The subject only says that
  `AnsweredQuestion` "represents an answered question". Its real role is to
  be the ground truth: its `sources` are what the retriever should find, and
  `evaluate` compares our `MinimalSearchResults` against it. Drawing the
  pipeline (schema above) made the role of each model clear.
- **Nested models.** Results contain questions, which contain lists of
  sources: list comprehensions were the natural tool, but also the source of
  a few headaches (e.g. building a list of lists instead of a list).
- **transformers warnings.** The text-generation pipeline copies a legacy
  `max_length=20` that conflicts with `max_new_tokens`, and merges its config
  with Qwen's own sampling parameters. The fix was to start from Qwen's
  config, set it once, and give that single config to the pipeline.
- **Keeping Qwen inside the sources.** The model happily gave a chocolate cake
  recipe, because BM25 always returns *some* chunks. Placing the rules right
  after the sources and asking for an exact refusal sentence fixed it.
- **Errors that hide the real cause.** An empty `data/processed/` folder made
  `bm25s` fail with a raw `[Errno 2]`; `load_index` now checks the index
  files themselves and says to run `index` first.

# Example usage

Titles and results are colored and centered in the terminal; they are shown
here without colors and left-aligned.

## With make

```text
$ make run
Parameters
  MAX_CHUNK_SIZE       2000
  K                    10
  SCOPE                UnansweredQuestions
  DATASET_FILE         dataset_docs_public.json
  DATASET              data/datasets/UnansweredQuestions/dataset_docs_public.json
  ANSWERED_QUESTIONS   data/datasets/AnsweredQuestions/dataset_docs_public.json
  SEARCH_SAVE_DIR      data/output/search_results/UnansweredQuestions
  SEARCH_RESULTS       data/output/search_results/UnansweredQuestions/dataset_docs_public.json
  ANSWER_SAVE_DIR      data/output/search_results_and_answer/UnansweredQuestions

=== INDEXATION ===
BM25S Compute Scores: 100%|██████████| 11157/11157 [...]
Indexation complete, indexed 11157 chunks under data/processed/

=== SEARCH_DATASET ===
Searching: 100%|██████████| 100/100 [...]
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

=== EVALUATE ===
Evaluation Results:
===================
Questions evaluated: 100
Recall@1: 0.61 (61.0%)
Recall@3: 0.83 (83.0%)
Recall@5: 0.89 (89.0%)
Recall@10: 0.92 (92.0%)

=== ANSWER_DATASET ===
Answering: 100%|██████████| 100/100 [03:02<00:00, 1.83s/it]
Saved student_search_results_and_answer to data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json
```

## With the CLI

```text
$ uv run python -m src index --max_chunk_size 2000
=== INDEXATION ===
Indexation complete, indexed 11157 chunks under data/processed/

$ uv run python -m src search "How do I enable LoRA adapters?" --k 3
=== SEARCH ===
data/raw/vllm-0.10.1/docs/features/lora.md [14472:15148]
data/raw/vllm-0.10.1/docs/features/lora.md [4075:4695]
data/raw/vllm-0.10.1/docs/features/lora.md [0:2000]

$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
=== SEARCH_DATASET ===
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
=== EVALUATE ===
Evaluation Results:
===================
Questions evaluated: 100
Recall@1: 0.61 (61.0%)
...

$ uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter?"
=== ANSWER ===
The answer for the query 'What HTTP endpoint is used to dynamically load a LoRA adapter?' is:
The HTTP endpoint used to dynamically load a LoRA adapter is `/v1/load_lora_adapter`. [...] [1]

$ uv run python -m src answer "What is the recipe of the chocolate cake?"
=== ANSWER ===
The answer for the query 'What is the recipe of the chocolate cake?' is:
I don't know based on the vLLM sources.

$ uv run python -m src search ""
Error: Query must not be empty
```

# Resources

**Libraries**

| Library | Used for | Key functions |
|---|---|---|
| `bm25s` | Tokenization and BM25 index | `bm25s.tokenize`, `BM25().index`, `.retrieve`, `.save`, `BM25.load` |
| `transformers` | Running Qwen3-0.6B | `pipeline("text-generation")`, `GenerationConfig.update` |
| `pydantic` | Data models and JSON validation | `model_validate_json`, `model_dump_json`, `model_validator` |
| `fire` | CLI from a class | `fire.Fire(CLI)` |
| `tqdm` | Progress bars | `tqdm(iterable, desc=...)` |
| `pathlib` | Files and folders | `rglob`, `read_text`, `write_text`, `mkdir(parents=True)` |

**Documentation**

- RAG chunking strategies: https://blog.stephane-robert.info/docs/developper/programmation/python/rag-chunking/
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- transformers pipelines: https://huggingface.co/docs/transformers/main_classes/pipelines
- Chatting with a model: https://huggingface.co/docs/transformers/conversations
- Chat templates: https://huggingface.co/docs/transformers/chat_templating
- LLM generation tutorial: https://huggingface.co/docs/transformers/llm_tutorial
- Nested list comprehensions: https://www.geeksforgeeks.org/python/nested-list-comprehensions-in-python/
- printf command in Linux / bash for Makefile formatting: https://linuxcapable.com/printf-command-in-linux/
- Makefile MAKECMDGOALS: https://www.gnu.org/software/make/manual/html_node/Goals.html
- Makefile error message: https://www.gnu.org/software/make/manual/html_node/Make-Control-Functions.html

## AI usage

Claude (Anthropic) was used throughout the project as an assistant:

- explaining the documentation of `transformers` and `bm25s`, and the tools
  of the project;
- debugging `transformers` warnings and errors that did not come from the
  project itself (the `max_length` / `max_new_tokens` conflict, sampling
  parameters, tokenizer clean-up);
- final review: subject compliance, edge cases, flake8 / mypy, docstrings,
  and the writing of this README from my notes.
