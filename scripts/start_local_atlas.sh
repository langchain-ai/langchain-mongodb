#!/bin/bash
set -eu

echo "Starting the container"

IMAGE=mongodb/mongodb-atlas-local:latest
DOCKER=$(which docker || which podman)

$DOCKER pull $IMAGE

$DOCKER kill mongodb_atlas_local || true

CONTAINER_ID=$($DOCKER run --rm -d --name mongodb_atlas_local -P $IMAGE)

function wait() {
  CONTAINER_ID=$1
  echo "waiting for container to become healthy..."

  # Get the exposed port first so we can connect
  EXPOSED_PORT=$($DOCKER inspect --format='{{ (index (index .NetworkSettings.Ports "27017/tcp") 0).HostPort }}' "$CONTAINER_ID")

  # Wait for MongoDB to be ready (up to 120 seconds)
  echo "Waiting for MongoDB to accept connections on port $EXPOSED_PORT..."
  timeout 120 bash -c "until $DOCKER exec mongodb_atlas_local mongosh --quiet --eval 'db.adminCommand({ping: 1})' &>/dev/null; do sleep 2; done" || {
    echo "ERROR: MongoDB failed to start within 120 seconds"
    $DOCKER logs mongodb_atlas_local
    exit 1
  }

  echo "MongoDB is ready. Waiting for Atlas Search services to initialize..."
  # Atlas Search and auto-embedding services need additional time to start
  # This is especially important for auto-embedding functionality
  sleep 30

  echo "Container is ready!"
  $DOCKER logs mongodb_atlas_local
}

wait "$CONTAINER_ID"

EXPOSED_PORT=$($DOCKER inspect --format='{{ (index (index .NetworkSettings.Ports "27017/tcp") 0).HostPort }}' "$CONTAINER_ID")
export MONGODB_URI="mongodb://127.0.0.1:$EXPOSED_PORT/?directConnection=true"
SCRIPT_DIR=$(realpath "$(dirname ${BASH_SOURCE[0]})")
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo "MONGODB_URI=$MONGODB_URI" > $ROOT_DIR/.local_atlas_uri
