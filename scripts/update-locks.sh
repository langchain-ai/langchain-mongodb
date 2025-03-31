#!/bin/bash
set -eu

python -m uv lock

pushd libs/langgraph-checkpoint-mongodb
python -m uv lock
popd

pushd libs/langchain-mongodb
python -m uv lock
popd
