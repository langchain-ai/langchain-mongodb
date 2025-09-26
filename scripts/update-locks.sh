#!/bin/bash
set -eu

find . -type f -name "uv.lock" | while read -r file; do
  dir=$(dirname "$file")
  (
    cd "$dir" && uv lock
  )
done
