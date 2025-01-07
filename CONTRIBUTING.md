# Contributing Guide

We welcome contributions to this project! Please follow the following guidance to setup the project for development and start contributing.

### Fork and clone the repository

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) workflow. Please do not try to push directly to this repo unless you are a maintainer.


### Dependency Management: Poetry and other env/dependency managers

This project utilizes [Poetry](https://python-poetry.org/) v1.7.1+ as a dependency manager.

Install Poetry: **[documentation on how to install it](https://python-poetry.org/docs/#installation)**.

### Local Development Dependencies

The project configuration and the makefile for running dev commands are located under the `libs/langchain-mongodb` or `libs/langgraph-checkpoint-mongodb` directories.

```bash
cd libs/langchain-mongodb
```

Install langchain-mongodb development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
make install
```

Then verify the installation.

```bash
make test
```

### Testing

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.

To run unit tests:

```bash
make test
```

Integration tests cover the end-to-end service calls as much as possible.
However, in certain cases this might not be practical, so you can mock the
service response for these tests. There are examples of this in the repo,
that can help you write your own tests. If you have suggestions to improve
this, please raise an issue.

To run the integration tests:

```bash
make integration_test
```

### Formatting and Linting

Formatting ensures that the code in this repo has consistent style so that the
code looks more presentable and readable. It corrects these errors when you run
the formatting command. Linting finds and highlights the code errors and helps
avoid coding practicies that can lead to errors.

Run both of these locally before submitting a PR. The CI scripts will run these
when you submit a PR, and you won't be able to merge changes without fixing
issues identified by the CI.

We use pre-commit for [pre-commit](https://pypi.org/project/pre-commit/) for
automatic formatting of the codebase.

To set up `pre-commit` locally, run:

```bash
brew install pre-commit
pre-commit install
```

#### Manual Linting and Formatting

Linting and formatting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run lint:

```bash
make lint
```

To run the type checker:

```bash
make typing
```

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

#### Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so it could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:

```bash
make codespell
```

If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `.pre-commit-config.yaml` file.
