# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12

FROM public.ecr.aws/docker/library/python:${PYTHON_VERSION}
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=1001
RUN groupadd -g ${UID} app \
    && useradd -l -r -u ${UID} -g ${UID} -d /app app && \
    mkdir -p /app && chown app:app /app

WORKDIR /app

# Control whether to use the cache or not, useful for reducing image
# size when building with Kaniko. Kaniho doesn't support cache mounts.
ARG UV_NO_CACHE=false
ENV UV_NO_CACHE=${UV_NO_CACHE}

# Install dependencies.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --frozen --no-install-project --no-dev


# Copy the rest of the app's code.
COPY --chown=app:app . .

USER app

# Expose the port that the application listens on.
EXPOSE 8001

# Run the application.
ENTRYPOINT ["./run.sh"]
