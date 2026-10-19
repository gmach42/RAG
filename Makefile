PYTHON := uv run python
FLAKE8 := uv run flake8
MYPY := uv run mypy
LINT_PATHS := src

install:
	uv sync

run:
	@echo "Please enter the CLI commands in order, here's a default example:"
	@echo "INDEX:"
	@echo "  uv run python -m src index --max_chunk_size 2000"
	@echo "RETRIEVE:"
	@echo "  uv run python -m src search_dataset"
	@echo "		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json"
	@echo "		--k 10"
	@echo "		--save_directory data/output/search_results/UnansweredQuestions"
	@echo "EVALUATE:"
	@echo "  /moulinette evaluate_student_search_results"
	@echo "  	data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
	@echo "  	data/datasets/AnsweredQuestions/dataset_docs_public.json"
	@echo "  	--k 10 --max_context_length 2000""
	@echo "ANSWER:"
	@echo "  uv run python -m src answer_dataset"
	@echo "		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
	@echo "		--save_directory data/output/search_results_and_answer/UnansweredQuestions"


commands:
	@echo "Please enter the CLI commands in order, here's a default example:"
	@echo "INDEX:"
	@echo "  uv run python -m src index --max_chunk_size <value>"
	@echo "RETRIEVE:"
	@echo "  uv run python -m src search_dataset"
	@echo "		--dataset_path <file_path>"
	@echo "		--k <k>"
	@echo "		--save_directory <dir_path>"
	@echo "EVALUATE:"
	@echo "  /moulinette evaluate_student_search_results"
	@echo "  	<unanswered_questions_file_path>"
	@echo "  	<answered_questions_file_path>"
	@echo "  	--k <k> --max_context_length <value>"
	@echo "ANSWER:"
	@echo "  uv run python -m src answer_dataset"
	@echo "		--student_search_results_path <file_path>"
	@echo "		--save_directory <dir_path>"

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

.PHONY: install run debug clean lint lint-strict setup
