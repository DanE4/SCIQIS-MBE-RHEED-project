# Fallback for a machine where the local setup will not cooperate: everything the notebook
# needs, pinned by uv.lock, on any OS that runs Docker. See the README's Docker section.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Every batch workflow stamps its artifacts with the commit and whether the tree was dirty,
# by shelling out to git, so git has to be here and .git has to be copied in.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project
# Dependencies first, so editing a notebook or a script does not re-resolve the environment.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project
COPY . .
RUN uv sync --locked

EXPOSE 2718
# --host 0.0.0.0 is what makes the port reachable from outside the container; --no-token
# matches `make notebook`, and is only safe because the port is published to localhost.
CMD ["uv", "run", "marimo", "edit", "notebooks/mbe_rheed.py", \
     "--host", "0.0.0.0", "--port", "2718", "--no-token"]
