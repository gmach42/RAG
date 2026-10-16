PYTHON := uv run python
FLAKE8 := uv run flake8
MYPY := uv run mypy
LINT_PATHS := src

install:
	uv sync

run:
	$(PYTHON) -m src index --max_chunk_size 2000
	$(PYTHON) -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
	$(PYTHON) -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
	$(PYTHON) -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} + 2>/dev/null || true

lint:
	$(FLAKE8) $(LINT_PATHS)
	$(MYPY) $(LINT_PATHS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(FLAKE8) $(LINT_PATHS)
	$(MYPY) $(LINT_PATHS) --strict

.PHONY: install run debug clean lint lint-strict
