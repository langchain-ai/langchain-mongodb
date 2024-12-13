#!/bin/bash
set -eu

echo "Starting the container"

IMAGE=mongodb/mongodb-atlas-local:latest
podman pull $IMAGE

CONTAINER_ID=$(podman run --rm -d -e DO_NOT_TRACK=1 --name mongodb_atlas_local -P $IMAGE)

function wait() {
  CONTAINER_ID=$1
  echo "waiting for container to become healthy..."
  podman logs mongodb_atlas_local
}

wait "$CONTAINER_ID"

EXPOSED_PORT=$(podman inspect --format='{{ (index (index .NetworkSettings.Ports "27017/tcp") 0).HostPort }}' "$CONTAINER_ID")
export CONN_STRING="mongodb://127.0.0.1:$EXPOSED_PORT/?directConnection=true"
SCRIPT_DIR=$(realpath "$(dirname ${BASH_SOURCE[0]})")
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo "MONGODB_URI=mongodb://127.0.0.1:$EXPOSED_PORT/?directConnection=true" > $ROOT_DIR/.local_atlas_uri

# Sleep for a bit to let all services start.
sleep 5
