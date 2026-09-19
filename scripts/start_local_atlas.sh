#!/bin/bash
set -eu

if [[ ${REQUIRE_AUTO_EMBEDDING:-false} == true && -z ${VOYAGE_API_KEY:-} ]]; then
  echo "Auto-embedding tests require VOYAGE_API_KEY; configure it for Actions and Dependabot runs." >&2
  exit 1
fi

echo "Starting the container"

IMAGE=mongodb/mongodb-atlas-local:preview@sha256:37e6794e5ff9926328b8e437dfc0c444b9f49014fa52876edc80ac1f9844cc02
DOCKER=$(command -v docker || command -v podman)

"$DOCKER" pull "$IMAGE"

"$DOCKER" kill mongodb_atlas_local || true

EMBEDDING_ENV=()
if [[ -n ${VOYAGE_API_KEY:-} ]]; then
  EMBEDDING_ENV+=(--env VOYAGE_API_KEY)
fi
if [[ -n ${EMBEDDING_PROVIDER_ENDPOINT:-} ]]; then
  EMBEDDING_ENV+=(--env EMBEDDING_PROVIDER_ENDPOINT)
fi
CONTAINER_ID=$("$DOCKER" run --rm -d --name mongodb_atlas_local -P "${EMBEDDING_ENV[@]}" "$IMAGE")

function wait_until_healthy() {
  for ((attempt = 0; attempt < 60; attempt++)); do
    status=$("$DOCKER" inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$CONTAINER_ID")
    case "$status" in
      healthy) return 0 ;;
      unhealthy|exited|dead) break ;;
    esac
    sleep 2
  done
  echo "Local Atlas did not become healthy within 120 seconds; inspect the container locally." >&2
  return 1
}

wait_until_healthy

EXPOSED_PORT=$("$DOCKER" inspect --format='{{ (index (index .NetworkSettings.Ports "27017/tcp") 0).HostPort }}' "$CONTAINER_ID")
export MONGODB_URI="mongodb://127.0.0.1:$EXPOSED_PORT/?directConnection=true"
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
echo "MONGODB_URI=$MONGODB_URI" > "$ROOT_DIR/.local_atlas_uri"
