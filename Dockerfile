FROM python:3.12-slim

WORKDIR /app

# Layer-cache the pinned dependency set: source edits below never re-resolve deps,
# and the shipped environment matches the paper lockfile instead of floating.
# CPU wheels by default: this container serves CPU inference, and torch's Linux
# metadata otherwise pulls the multi-GB CUDA stack. For a GPU image, override:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.lock ./
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" -r requirements.lock

COPY . .
# The lock pins the ml/research core; this adds the app extras (gradio, gunicorn)
# and installs novavision itself. The lock doubles as a constraints file so the
# extras' resolution can never silently move a locked pin.
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" -c requirements.lock ".[app]"

# A container is an isolated sandbox reached via mapped ports, so it opts into a
# public bind here; protect the generate route with NOVA_API_TOKEN when exposing it.
ENV NOVA_PUBLIC=1

# uid >= 10000 cannot collide with a host system account.
RUN useradd --create-home --uid 10001 nova && chown -R nova:nova /app
USER nova

EXPOSE 7860

CMD ["python", "app.py"]
