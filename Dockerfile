FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG USE_FLASH_ATTENTION=true

ENV PYTHON_VERSION=3.10
ENV POETRY_VENV=/app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends python3.10-venv ffmpeg && rm -rf /var/lib/apt/lists/*

RUN python -m venv $POETRY_VENV
RUN $POETRY_VENV/bin/pip install -U pip setuptools
RUN $POETRY_VENV/bin/pip install poetry==1.7.1

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

COPY . .

RUN poetry install
RUN $POETRY_VENV/bin/pip install -U wheel
RUN $POETRY_VENV/bin/pip install ninja packaging

RUN if [ "$USE_FLASH_ATTENTION" = "true" ]; then \
      $POETRY_VENV/bin/pip install flash-attn --no-build-isolation; \
    else \
      echo "Skipping flash-attn installation"; \
    fi

EXPOSE 8887

ENV USE_FLASH_ATTENTION=$USE_FLASH_ATTENTION
ENV WHISPER_DEVICE=cuda:0

# Set memory fraction to limit GPU usage (60% of available memory)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CMD ["gunicorn", "--bind", "0.0.0.0:8887", "--workers", "1", "--timeout", "0", "app.app:app", "-k", "uvicorn.workers.UvicornWorker"]
