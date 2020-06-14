DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
docker run -it -p 8888:8888 -v $DIR:/root nathanhillyer/ml-base