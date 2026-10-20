#!/bin/sh
set -eu
cd "$(dirname "$0")"
python -m build "$@"
