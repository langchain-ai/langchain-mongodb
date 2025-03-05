set shell := ["bash", "-c"]

# Default target executed when no arguments are given.
[private]
default:
  @just --list

docs:
	uv run sphinx-build -T -W -b html docs docs/_build/html
