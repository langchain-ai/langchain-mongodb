#!/bin/bash
set -eu

echo "Starting the container"

IMAGE=mongodb-atlas-local:latest
podman pull $IMAGE

CONTAINER_ID=$(podman run --rm -d -e DO_NOT_TRACK=1 -P --health-cmd "/usr/local/bin/runner healthcheck" $IMAGE)

function wait() {
  CONTAINER_ID=$1
  echo "waiting for container to become healthy..."
  podman healthcheck run "$CONTAINER_ID"
  for _ in $(seq 600); do
      STATE=$(podman inspect -f '{{ .State.Health.Status }}' "$CONTAINER_ID")

      case $STATE in
          healthy)
          echo "container is healthy"
          return 0
          ;;
          unhealthy)
          echo "container is unhealthy"
          podman logs "$CONTAINER_ID"
          stop
          exit 1
          ;;
          *)
          echo "Unrecognized state $STATE"
          sleep 1
      esac
  done

  echo "container did not get healthy within 120 seconds, quitting"
  podman logs mongodb_atlas_local
  stop
  exit 2
}

wait "$CONTAINER_ID"

EXPOSED_PORT=$(podman inspect --format='{{ (index (index .NetworkSettings.Ports "27017/tcp") 0).HostPort }}' "$CONTAINER_ID")
export CONN_STRING="mongodb://127.0.0.1:$EXPOSED_PORT/?directConnection=true"
SCRIPT_DIR=$(realpath "$(dirname ${BASH_SOURCE[0]})")
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo "MONGODB_URI=mongodb://127.0.0.1:$EXPOSED_PORT/?directConnection=true" > $ROOT_DIR/.local_atlas_uri
