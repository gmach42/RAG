TARGETS := install run params debug clean lint lint-strict \
           index search search-dataset answer answer-dataset evaluate

UNKNOWN_GOALS := $(filter-out $(TARGETS),$(MAKECMDGOALS))
ifneq ($(UNKNOWN_GOALS),)
$(error Unknown target(s): $(UNKNOWN_GOALS) -- valid: $(TARGETS))
endif

LINT_PATHS := src

# == Overridable parameters: ==
# index
MAX_CHUNK_SIZE ?= 2000
# search / answer
K ?= 10
QUERY ?= How to configure the OpenAI server?
# search_dataset
UNANSWERED_DATASET ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
SEARCH_SAVE_DIR ?= data/output/search_results/UnansweredQuestions
# answer_dataset
SEARCH_RESULTS_PATH ?= data/output/search_results/UnansweredQuestions/dataset_docs_public.json
ANSWER_SAVE_DIR ?= data/output/search_results_and_answer/UnansweredQuestions
# evaluate
EVALUATE_DATASET ?= data/datasets/AnsweredQuestions/dataset_docs_public.json

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
		--dataset_path $(UNANSWERED_DATASET) \
		--k $(K) \
		--save_directory $(SEARCH_SAVE_DIR)

evaluate:
	$(Q)uv run python -m src evaluate \
        --student_search_results_path $(SEARCH_RESULTS_PATH) \
        --dataset_path $(EVALUATE_DATASET)

answer-dataset:
	$(Q)uv run python -m src answer_dataset \
		--student_search_results_path $(SEARCH_RESULTS_PATH) \
		--save_directory $(ANSWER_SAVE_DIR)

run: Q := @
run: params index search-dataset evaluate answer-dataset

params:
	@printf "$(BOLD)Parameters$(RESET)\n"
	@printf "  %-20s %s\n" "MAX_CHUNK_SIZE" "$(MAX_CHUNK_SIZE)"
	@printf "  %-20s %s\n" "K" "$(K)"
	@printf "  %-20s %s\n" "DATASET_SCOPE" "$(DATASET_SCOPE)"
	@printf "  %-20s %s\n" "DATASET_FILE" "$(DATASET_FILE)"
	@printf "  %-20s %s\n" "UNANSWERED_DATASET" "$(UNANSWERED_DATASET)"
	@printf "  %-20s %s\n" "EVALUATE_DATASET" "$(EVALUATE_DATASET)"
	@printf "  %-20s %s\n" "SEARCH_SAVE_DIR" "$(SEARCH_SAVE_DIR)"
	@printf "  %-20s %s\n" "SEARCH_RESULTS_PATH" "$(SEARCH_RESULTS_PATH)"
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

.PHONY: $(TARGETS)
