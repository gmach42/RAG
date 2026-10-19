LINT_PATHS := src

# Overridable:
MAX_CHUNK_SIZE ?= 2000
K ?= 10
DATASET ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
ANSWERED_QUESTIONS ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
SEARCH_DIR ?= data/output/search_results/UnansweredQuestions
ANSWER_DIR ?= data/output/search_results_and_answer/UnansweredQuestions
SEARCH_RESULTS ?= $(SEARCH_DIR)/dataset_docs_public.json
QUERY ?= What's a vllm?

install:
	uv sync

index:
	uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python -m src search "$(QUERY)" --k $(K)

answer:
	uv run python -m src answer "$(QUERY)" --k (K)


search-dataset:
	uv run python -m src search_dataset \
		--dataset_path $(DATASET) \
		--k $(K) \
		--save_directory $(SEARCH_DIR)

evaluate:
	uv run python -m src evaluate \
        --student_search_results_path $(SEARCH_RESULTS) \
        --dataset_path $(ANSWERED_QUESTIONS)

answer-dataset:
	uv run python -m src answer_dataset \
		--student_search_results_path $(SEARCH_RESULTS) \
		--save_directory $(ANSWER_DIR)

run: index search-dataset evaluate answer-dataset

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} + 2>/dev/null || true

lint:
	uv run flake8 $(LINT_PATHS)
	uv run mypy $(LINT_PATHS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(LINT_PATHS)
	uv run mypy $(LINT_PATHS) --strict

.PHONY: install run debug clean lint lint-strict index search search-dataset answer answer-dataset evaluate
