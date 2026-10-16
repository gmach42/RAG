4 stages:

Indexing: Organise the data so it can be searched.
Retrieving: match a question against the index and pull the most relevant snippets.
Augmenting: filter those snippets and place them in the model's context window.
Generating: read that context and produce the answer.

each stage feeds the next.

Data models must use pydantic (structures exchanged between stages such as those in the Data Models section)
Service or orchestration classes (indexer, retriever, pipeline) do not have to be pydantic.

Your system must expose a Command-Line Interface (CLI) built with python fire and progress bars (tqdm) for long-running operations.

model: Qwen/Qwn3-0.6B

Obj:
	Build a Retrieval-Augmented Generation system that answers questions about a codebase.
	Ingest the provided vLLM repo into a searchable index, retrieve the most relevan snippets for a question, generate an answer from them with Qwen and measure retrieval quality with recall@k?
	The system is judge on whether it retrieves the right source locations and produces answers grounded in those sources.

Indexing:
	Read the files judged useful from the vLLM repo, split each one into chunks and persist an index that retrieval can query in ms.
	Index the whole corpus must take **at most 5 minutes**.
	2 distinct chunking strategies:
		1. Python code chunking,
		2. Markdown / text chunking
	For retrieval, implemet at least one of the two classic lexical methods:
		1. TF-IDF,
		2. BM25

Chunk size is configurable through a CLI arg (--max_chunk_size). Default is 2000 characters, do not go above it.

Retrieval:
	Index built -> you can search it. Given a question, your system returns the top-k most relevant snippets.
	Each result is a source location: a file_path and the character range (first_character_index, last_character_index) it covers, at most 2000 characters wide.
	CAUTION: file_path must match the corpus path exactly (e.g.data/raw/vllm-0.10.1/docs/features/lora.md)
	Retrieval must work for a single query and in batch over a whole dataset of questions read from JSON.
	On the reference datasets, you must reach at least 80% reacall@5 on docs questions and 50% on code questions.

Answer Generation:
	With the right snippets retreved, the system generates a natural-language answer to the question using Qwen.
	Pass the retreved context to the model within its token budget and produce JSON.
	A satisfactory answer is:
		Coherent and understandable,
		Grounded in the retrieved sources, with no major hallucinations.
		On point: it answer the question

Data Models:
	Implement the following pydantic models:
		MinimalSource Model: represents a single source of information.
		The UnansweredQuestion and Answered Question models: represent an unanswered and answered question, respectively.
		RagDataset Model: represents a dataset of RAG questions.
		MinimalSearchResult and MinimalAnswer models: represent a search result and an answer.
		StudentSearchResults and StudentSearchResultsAndAnswer

Output:
	Each command writes a JSON file that conforms to the provided pydantic models:
	• For search operations: Use StudentSearchResults model with:
		◦ search_results: List of MinimalSearchResults containing question_id, ques-
		tion and retrieved_sources
		◦ k: Number of results requested
	• For answer generation: Use StudentSearchResultsAndAnswer model with:
		◦ search_results: List of MinimalAnswer containing question_id, question,
		retrieved_sources, and answer
		◦ k: Number of results requested
	• Source information: Each MinimalSource contains:
		◦ file_path: path to the source file, relative to your project root and writ-
		ten exactly as in the ingested corpus (e.g. data/raw/vllm-0.10.1/...); it is
		compared verbatim to the reference
		◦ first_character_index: Starting character position
		◦ last_character_index: Ending character position

Command-Line Interface:
	CLI built with Python Fire. Every command is invoked as uv run python -m src <command> [options].
	Required commands:
		- index -max_chunk_size <int> # Ingest data/raw/ and build the index under data/processed/
		- search <query> -k <int> # Return the top-k sources for a single query
		- search_dataset -dataset_path <path> -k <int> -save_directory <dir> # Run search over a whole dataset and write a StudentSearchResults JSON file.
		- answer <query> -k <int> # Answer a single query using the retrieved context
		- answer_dataset -student_search_results_path <path> -save_directory <dir> # Generate answers for a dataset, producing StudentSearchResultsAndAnswer JSON file.
		- evaluate -student_search_results_path <path> - dataset_path <path> # Report your own recall@k against a ground-truth dataset for your own testing.

	Input and output paths must be configurable CLI args.
	Include progress bars (tqdm) for long-running operations and handle degenerate inputs (empty query, nonsensical query, k=0, missing files, malformed JSON) gracefully.
	Your solution must never import or call the moulinette.

Expected project layout for evaluation:
	During the defense, the retrieval pipeline is run end-to-end as a single automated flow:
	index -> search_dataset -> moulinette evaluate_student_search_results
	your repo must repect the following layout:
		- src/
		- pyproject.toml
		- README.md
		- data/raw/
		- data/processed/
		- data/datasets/UnansweredQuestions/ and data/datasets/AnsweredQuestions/
		- data/output/search_results/<DatasetScope>/
		- data/output/search_results_and_answer/<DatasetScope>/

	All input and output paths (--dataset_path, --save_directory, --student_search_results_path)
	must be configurable CLI arguments and must never be hard-coded.

Running the full pipeline:
	1. Index the corpus once:
		```
		uv run python -m src index --max_chunk_size 2000
		Ingestion complete! Indices saved under data/processed/
		```
	2. Search the dataset. Always scope --save_directory by dataset (UnansweredQuestions
	or AnsweredQuestions): the public datasets share file names, so writing every run into
	the same folder would overwrite previous results.
		```
		uv run python -m src search_dataset
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
		--k 10
		--save_directory data/output/search_results/UnansweredQuestions
		Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
		```
	3. Score with the moulinette (rename moulinette-ubuntu/-fedora to moulinette
	first). The student results come first, the ground-truth AnsweredQuestions dataset
	second:
		```
		./moulinette evaluate_student_search_results
		data/output/search_results/UnansweredQuestions/dataset_docs_public.json
		data/datasets/AnsweredQuestions/dataset_docs_public.json
		--k 10 --max_context_length 2000
		Student data is valid: True
		Evaluation Results
		========================================
		Recall@1: 0.450 Recall@3: 0.590 Recall@5: 0.650 Recall@10: 0.720
		```
	4. Generate answers from the search results:
		```
		uv run python -m src answer_dataset
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
		--save_directory data/output/search_results_and_answer/UnansweredQuestions
		Loaded 100 questions ... Processed 100 of 100 questions
		Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json
		```


Bonus:
	1. Semantic embeddings: use a vector index built with a lightweight CPU model model (eg all-MiniLM-L6-v2)
	2. Hybrid retrieval: combine lexical and semantic into a single result list
	3. Incremental indexing: when a file changes re-index only that file instead of rebuilding the whole index.
	4. Caching: cache the index and query results to speed up cold start and repeated queries.
	5. Local HTTP API: expose querying the index and answering questions over a small local http api, so the system can be driven by something other than the CLI.


TLDR:

Indexation:
	- indexer les fichiers présents dans la vllm (avec une limite max de 2000 charactères) à l'aide de BN25 ou TF-IDF.
	- l'indexation s'effectue par chunk et une indexation "bête" est préférée dû à comment est codé la moulinette.
	- majoritairement 2 types de fichiers a parse -> indexer les fichiers md et py (mais txt sympa aussi)

Retrieval:
	- Une fois la source indexée (il faut vérifier que la source est bien indexée avant de retrieve, le cas échéant re-indexer), on peut effectuer la recherche sur l'index.
	- On recherche les chunks qui répondent le mieux à la query -> Basiquement c'est un CTRL-F sur l'index.

Answering:
	- Il s'agit ici de recréer ce que l'on a fait durant Call_me_maybe mais avec l'aide d'une librairie comme Olama, torch, trandformers... Afin de faciliter la tache
	- On utilise les chunks récupérés lors de la phase de retrieval pour générer les réponses aux questions posées.
	- Il est conseillé de désactiver l'aléatoire de Qwen
	- Faire en sorte que Qwen n'hallucine pas en se basant uniquement sur les chunks récupérés.


============================== OUTPUT PREP ==============================

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

• data/output/search_results/<DatasetScope>/: the output of search_dataset,
scoped by dataset.
• data/output/search_results_and_answer/<DatasetScope>/: the output of answer_dataset,
scoped by dataset.

All input and output paths (--dataset_path, --save_directory, --student_search_results_path)
must be configurable CLI arguments and must never be hard-coded: the evaluator points
them at the datasets and at dedicated output directories. If the structure or the com-
mand interface is not respected, the reference scripts cannot run and the corresponding
checks are considered failed by design

1. index the corpus
2. search a whole dataset
3. score the results with the moulinette
4. generate answers

// for the search and answer query commands, it's the same behavior but on one question at a time

Example:
```python
1. Index the corpus once:

uv run python -m src index --max_chunk)size 2000
Index done, indeces saved under data/processed/

2. Search a dataset:

uv run python -m src search_dataset
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
    --k 10
    --save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

3. Score with the moulinette

/moulinette evaluate_student_search_results
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json
    data/datasets/AnsweredQuestions/dataset_docs_public.json
    --k 10 --max_context_length 2000

Student data is valid: True
Evaluation Results
========================================
Recall@1: 0.450 Recall@3: 0.590 Recall@5: 0.650 Recall@10: 0.720

4. Generate answers

uv run python -m src answer_dataset
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
--save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions ... Processed 100 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json
