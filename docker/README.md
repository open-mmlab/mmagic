# Docker Image

We provide a [Dockerfile](Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmagic docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmagic/data mmagic
```

**Note**:
Versions defined in this [Dockerfile](Dockerfile) is not up-to-date.
If you use this Dockerfile in your project, you probably want to make some updates.
Feel free to submit an issue or PR for the update.
