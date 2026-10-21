LINT_PATHS := src

# Overridable parameters:
MAX_CHUNK_SIZE ?= 2000
K ?= 10
SCOPE ?= UnansweredQuestions
DATASET_FILE ?= dataset_docs_public.json
DATASET ?= data/datasets/$(SCOPE)/$(DATASET_FILE)
ANSWERED_QUESTIONS ?= data/datasets/AnsweredQuestions/$(DATASET_FILE)
SEARCH_SAVE_DIR ?= data/output/search_results/$(SCOPE)
ANSWER_SAVE_DIR ?= data/output/search_results_and_answer/$(SCOPE)
SEARCH_RESULTS ?= $(SEARCH_SAVE_DIR)/$(DATASET_FILE)
QUERY ?= How to configure the OpenAI server?

# Colors:
RED = \033[31m
GREEN = \033[32m
BLUE = \033[34m
YELLOW = \033[33m
BOLD = \033[1m
RESET = \033[0m

# Echo prefix: empty by default, '@' under 'make run' (inherited by its prerequisites)
Q :=

install:
	uv sync

index:
	$(Q)uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python -m src search "$(QUERY)" --k $(K)

answer:
	uv run python -m src answer "$(QUERY)" --k $(K)


search-dataset:
	$(Q)uv run python -m src search_dataset \
		--dataset_path $(DATASET) \
		--k $(K) \
		--save_directory $(SEARCH_SAVE_DIR)

evaluate:
	$(Q)uv run python -m src evaluate \
        --student_search_results_path $(SEARCH_RESULTS) \
        --dataset_path $(ANSWERED_QUESTIONS)

answer-dataset:
	$(Q)uv run python -m src answer_dataset \
		--student_search_results_path $(SEARCH_RESULTS) \
		--save_directory $(ANSWER_SAVE_DIR)

run: Q := @
run: params index search-dataset evaluate answer-dataset

params:
	@printf "$(BOLD)Parameters$(RESET)\n"
	@printf "  %-20s %s\n" "MAX_CHUNK_SIZE" "$(MAX_CHUNK_SIZE)"
	@printf "  %-20s %s\n" "K" "$(K)"
	@printf "  %-20s %s\n" "SCOPE" "$(SCOPE)"
	@printf "  %-20s %s\n" "DATASET_FILE" "$(DATASET_FILE)"
	@printf "  %-20s %s\n" "DATASET" "$(DATASET)"
	@printf "  %-20s %s\n" "ANSWERED_QUESTIONS" "$(ANSWERED_QUESTIONS)"
	@printf "  %-20s %s\n" "SEARCH_SAVE_DIR" "$(SEARCH_SAVE_DIR)"
	@printf "  %-20s %s\n" "SEARCH_RESULTS" "$(SEARCH_RESULTS)"
	@printf "  %-20s %s\n\n" "ANSWER_SAVE_DIR" "$(ANSWER_SAVE_DIR)"

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

.PHONY: install run params debug clean lint lint-strict index search search-dataset answer answer-dataset evaluate
