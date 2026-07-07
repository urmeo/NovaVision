FROM python:3.12-slim

WORKDIR /app

# CPU wheels by default: this container serves CPU inference, and torch's Linux
# metadata otherwise pulls the multi-GB CUDA stack. For a GPU image, override:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" ".[app,ml,research]"

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 nova && chown -R nova:nova /app
USER nova

EXPOSE 7860

# Let orchestrators see whether the app is actually serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '7860') + '/')" || exit 1

# One worker: the rate limiter, concurrency cap, and model live per process.
# gunicorn binds all interfaces (a container is reached via mapped ports);
# protect the generate route with NOVA_API_TOKEN when exposing it.
CMD python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind "0.0.0.0:${PORT:-7860}" server:app
