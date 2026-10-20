Containers
==========

Build from the repository root. These targets install the checkout, including
the CLI and SimpleITK; they do not install a stale package from PyPI::

   docker build -f docker/Dockerfile --target cli -t collageradiomics-cli .
   docker run --rm collageradiomics-cli --help

Mount a data directory at /work and use the host user ID for output permissions::

   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" collageradiomics-cli \
     -i image.nii.gz -m mask.nii.gz -o features.csv

For JupyterLab::

   docker build -f docker/Dockerfile --target notebook -t collageradiomics-notebook .
   docker run --rm --user "$(id -u):$(id -g)" -p 127.0.0.1:8888:8888 \
     -v "$PWD:/work" collageradiomics-notebook

Use the token-bearing URL printed by Jupyter. The image preserves default
authentication. A copy of the runnable notebook is at /opt/collage/examples.
Both images use a non-root default user and temporary runtime/cache directories.

The former dev_environment image hierarchy is retired. It depended on personal
base images and pinned collageradiomics 0.2.6. The single Dockerfile now supplies
both supported targets. The Containers workflow builds and smoke-tests both on
pull requests and relevant pushes. A manual run with publish=true publishes
ghcr.io/radxtools/collageradiomics-cli and collageradiomics-notebook with commit
and latest tags after the build checks pass.
