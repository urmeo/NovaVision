FROM python:3.12-slim

WORKDIR /app

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" ".[app,ml,research]"

RUN useradd --create-home --uid 10001 nova && chown -R nova:nova /app
USER nova

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '7860') + '/')" || exit 1

CMD python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind "0.0.0.0:${PORT:-7860}" server:app
