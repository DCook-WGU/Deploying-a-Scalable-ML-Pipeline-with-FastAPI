#!/bin/bash

# Make script strict. Will not fail silently. 
set -euo pipefail

# Defaults you can override per folder or at runtime:
#   CONTAINER_NAME=mlops-project ./start.sh
CONTAINER_NAME="${CONTAINER_NAME:-mlops-project-two}"
IMAGE="${IMAGE:-mlops-dev-project-two}"

# Warn (but don't fail) if WANDB_API_KEY isn't set in your WSL env
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "⚠️  WANDB_API_KEY is not set. Set it in ~/.bashrc or export it before running."
fi

echo "Starting container: $CONTAINER_NAME (image: $IMAGE)"
docker run -it \
  --name "$CONTAINER_NAME" \
  --rm \
  -v "$(pwd)":/app \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "$IMAGE"


# Original build
#!/bin/bash

# Optional: Export your API key once per shell session
#export WANDB_API_KEY=your-api-key-here

#docker run -it \
#  --name mlops-container \
#  -v $(pwd):/app \
#  -e WANDB_API_KEY=$WANDB_API_KEY \
#  mlops-dev






