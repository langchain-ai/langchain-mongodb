.PHONY: all lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

install:
	poetry install --with dev

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

lint:
	poetry run pre-commit run ruff --files .

typing:
	poetry run mypy --files .

codespell:
	pre-commit run --hook-stage manual codespell --files .


######################
# HELP
######################

help:
	@echo '----'
	@echo 'install                      - install the package in dev mode'
	@echo 'lint                         - run linter and formatter'
	@echo 'typing                       - run type check'
	@echo 'test                         - run unit tests'
	@echo 'integration_test             - run integration tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
