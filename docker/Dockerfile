ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/collage-cache \
    JUPYTER_CONFIG_DIR=/tmp/jupyter-config \
    JUPYTER_DATA_DIR=/tmp/jupyter-data \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime \
    IPYTHONDIR=/tmp/ipython

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY module /src/module
RUN python -m pip install --no-cache-dir '/src/module[cli]' \
    && useradd --create-home --uid 10001 collage \
    && mkdir /work && chown collage:collage /work
WORKDIR /work
USER collage
CMD ["python"]

FROM runtime AS notebook
USER root
RUN python -m pip install --no-cache-dir '/src/module[examples]' \
    && rm -rf /src
COPY jupyter/examples /opt/collage/examples
COPY sample_data /opt/collage/sample_data
USER collage
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--ServerApp.root_dir=/work"]

FROM runtime AS cli
USER root
RUN rm -rf /src
USER collage
ENTRYPOINT ["collageradiomics"]
CMD ["--help"]
