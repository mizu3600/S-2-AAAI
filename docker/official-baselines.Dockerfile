FROM python:3.11-slim AS common

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "numpy<3" \
    aiohttp \
    graspologic \
    nano-vectordb \
    networkx \
    openai \
    pydantic \
    tenacity \
    tiktoken \
    xxhash

WORKDIR /project

FROM common AS path-light

RUN pip install --no-cache-dir "lightrag-hku==1.5.5rc1"

FROM common AS graphrag

RUN pip install --no-cache-dir "graphrag==3.1.1"
