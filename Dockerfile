FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[app,ml]"

# A container is an isolated sandbox reached via mapped ports, so it opts into a
# public bind here; protect the generate route with NOVA_API_TOKEN when exposing it.
ENV NOVA_PUBLIC=1

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 nova && chown -R nova:nova /app
USER nova

EXPOSE 7860

CMD ["python", "app.py"]
