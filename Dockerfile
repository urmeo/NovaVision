FROM python:3.12-slim

WORKDIR /app

# Layer-cache the pinned dependency set: source edits below never re-resolve deps,
# and the shipped environment matches the paper lockfile instead of floating.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
# The lock pins the ml/research core; this adds the app extras (gradio, gunicorn)
# and installs novavision itself. The lock doubles as a constraints file so the
# extras' resolution can never silently move a locked pin. Note: torch's Linux
# metadata pulls the CUDA stack (multi-GB); add the pytorch cpu wheel index for
# a slim CPU-only image.
RUN pip install --no-cache-dir -c requirements.lock ".[app]"

# A container is an isolated sandbox reached via mapped ports, so it opts into a
# public bind here; protect the generate route with NOVA_API_TOKEN when exposing it.
ENV NOVA_PUBLIC=1

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 nova && chown -R nova:nova /app
USER nova

EXPOSE 7860

CMD ["python", "app.py"]
