#!/bin/bash

# Make script strict. Will not fail silently. 
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-mlops-project-two}"
IMAGE="${IMAGE:-mlops-dev-project-two}"

echo "Stopping/removing old container (if any): $CONTAINER_NAME"
docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Building image: $IMAGE"
# If your Dockerfile supports build args (ENV_NAME, etc.), pass them here:
# docker build --build-arg ENV_NAME=mlops --build-arg ENV_FILE=/app/environment.yml -t "$IMAGE" .
# docker build -t "$IMAGE" .


NO_CACHE=${NO_CACHE:-0}
if [[ "$NO_CACHE" == "1" ]]; then 
    NC_FLAG="--no-cache" 
else 
    NC_FLAG=""
fi 

docker build $NC_FLAG --pull -t "$IMAGE" .

echo "Rebuilt image: $IMAGE"

echo "Starting container: $CONTAINER_NAME"

# Warn (but don't fail) if WANDB_API_KEY isn't set in your WSL env
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "⚠️  WANDB_API_KEY is not set. Set it in ~/.bashrc or export it before running."
fi

docker run -d --name "$CONTAINER_NAME" \
  -v "$(pwd)":/app \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e PYTHONPATH="/app:${PYTHONPATH:-}" \
  -w /app \
  "$IMAGE" \
  bash -lc "sleep infinity"

echo "Container is up. Exec with: docker exec -it $CONTAINER_NAME bash"


# To run no cache rebuild, use command below.
# NO_CACHE=1 ./rebuild.sh

# Original Version

#!/bin/bash
#docker stop mlops-container && docker rm mlops-container
#docker build -t mlops-dev .
