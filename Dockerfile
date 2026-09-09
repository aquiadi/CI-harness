# Multi-stage: the build stage carries uv and the toolchain, the runtime stage
# carries the virtualenv and the application and nothing else.

FROM python:3.12-slim-bookworm AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

WORKDIR /build

# Dependencies first, from the lockfile alone: this layer is cached until the
# lockfile changes, so an application edit does not reinstall the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

# A non-root user with no login shell and no home directory to write to.
RUN groupadd --system --gid 10001 evalgate \
    && useradd --system --uid 10001 --gid evalgate --no-create-home --shell /usr/sbin/nologin evalgate

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EVALGATE_ROOT=/app \
    EVALGATE_OVERRIDES="+experiment=baseline"

WORKDIR /app

COPY --from=build /build/.venv /app/.venv
COPY src /app/src
# The config tree, the prompts and the corpus are runtime inputs, not build
# artifacts: the image serves what these directories contain.
COPY configs /app/configs
COPY prompts /app/prompts
COPY data/corpus/synthetic /app/data/corpus/synthetic
COPY baseline.json /app/baseline.json

# The index is content-addressed and built at startup; it is the only thing the
# process writes, so it is the only writable directory.
RUN mkdir -p /app/data/index && chown -R evalgate:evalgate /app/data/index

USER evalgate
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "evalgate.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
