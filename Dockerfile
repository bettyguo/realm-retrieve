# =============================================================================
# ReaLM-Retrieve — single-stage CPU image (good for demos / smoke tests).
# For GPU training, see Dockerfile.gpu or the README.
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: compiler chain for some PyPI wheels, git for editable installs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy metadata first to maximise Docker layer cache.
COPY pyproject.toml requirements.txt requirements-dev.txt README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 \
 && pip install -e . \
 && pip install pytest

# Now bring in the rest (examples, tests, configs).
COPY . .

# Drop privileges.
RUN useradd --create-home --uid 1000 realm \
 && chown -R realm:realm /app
USER realm

# Quick smoke test that the image is healthy. Override at run time:
#   docker run --rm <image> realm-quickstart
CMD ["realm-quickstart"]
