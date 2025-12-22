# Get base image and install uv
#TODO: pin version
FROM python:3.11.7-slim-bookworm 
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

# Set the environment variable to ignore warnings
ENV PYTHONWARNINGS="ignore"

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Set up working directory
WORKDIR /amrgnn
COPY . /amrgnn
RUN mkdir /amrgnn/data
RUN mkdir /amrgnn/experiments

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Installing separately from its dependencies allows optimal layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Entry point
ENTRYPOINT [ "uv", "run"]

# CMD
CMD ["src/train.py"]



# #TODO
# # Create mounting point for data
# RUN mkdir /preprocessed_MALDI && mkdir /checkpoints
# RUN mkdir /data && touch /data/test_ids.csv

# TODO
# Set the entry point
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "maldist", \ 
#             "python", "/maldist/src/predict_multiclass.py", \
#             "dataset.input_dir=/preprocessed_MALDI", \
#             "dataset.predict_set_fp=/data/test_ids.csv", \
#             "model.name=MALDICNN", \
#             "trainer.accelerator=auto", \
#             "trainer.devices=auto", \]