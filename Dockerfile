# This Dockerfile is ONLY for deploying the ChromaDB server to Railway --
# it has nothing to do with the Python client code in imgint/, which runs
# locally (needs webcam access and holds the local encryption keys, so it
# must never be deployed anywhere). Railway auto-detects this file as the
# build target when the repo is connected.
FROM chromadb/chroma:latest
