FROM nathanhillyer/ml-base
ENV LANG C.UTF-8
RUN PIP_INSTALL="python -m pip install --upgrade --no-cache-dir --retries 10 --timeout 60" && \
    $PIP_INSTALL \
        collageradiomics==0.2.1 \
        Pillow \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
CMD JUPYTER_RUNTIME_DIR=/tmp jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/root'
EXPOSE 8888
