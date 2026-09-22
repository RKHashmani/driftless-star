# Container Image Use Overview

## Docker

### Building Docker container images

The Dockerfile lives in `stages/`, which is also the build context. Run the Docker build commands from the repository root.

```
docker build --file stages/Dockerfile --build-arg <build-args> --tag <tag> stages/
```

Example:

```
docker build \
    --file stages/Dockerfile \
    --build-arg ENVIRONMENT="stage-1-vmex" \
    --tag ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu \
    stages/
```

### Pulling Docker container images from a registry

```
docker pull <registry>/<repository>/<image>:<tag>
```

Example:

```
docker pull ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu
```

### Running Docker container images

```
docker run --rm -ti [--volume <local mount path>:<container-side mount path>] [--gpus <gpu>] <container>
```

Examples:

* Run a `stage-1-vmex-cpu` container in an interactive shell

```console
$ docker run --rm -ti ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu bash
root@060500d71aaf:/app# command -v python
/app/.pixi/envs/stage-1-vmex/bin/python
```

* Run a `stage-1-vmex-cpu` container in an interactive shell with the local working directory mounted

```console
$ docker run --rm -ti -v $PWD:/work -w /work ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu bash
root@1d603f18cd72:/work# pwd
/work
root@1d603f18cd72:/work# command -v python
/app/.pixi/envs/stage-1-vmex/bin/python
```

* Execute a command in a `stage-1-vmex-cpu` container

```console
$ docker run --rm -ti ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu python -c 'import vmex; print(vmex.__name__)'
vmex
```

* Run a `stage-1-vmex-gpu` container in an interactive shell with [NVIDIA driver support](https://github.com/NVIDIA/nvidia-container-toolkit)

```console
$ docker run --rm -ti --gpus all ghcr.io/driftless-star/driftless-star:stage-1-vmex-gpu bash
root@24982ece960b:/app# nvidia-smi --version
NVIDIA-SMI version  : 590.48.01
NVML version        : 590.48
DRIVER version      : 590.48.01
CUDA Version        : 13.1
```

* Run the same container pinned to a single GPU, the form the pipeline's slot allocator emits for each job

```console
$ docker run --rm -ti --gpus device=4 ghcr.io/driftless-star/driftless-star:stage-1-vmex-gpu bash
```

## Apptainer

On Linux machines Apptainer can be installed from conda-forge with

```
pixi global install apptainer
```

### Building Apptainer container images

Apptainer has no concept of "context" and so requires you to operate from the directory the Apptainer definition file expects to be executed from. For driftless-star, that directory is `stages/`.

```
cd stages
apptainer build [local options...] <IMAGE PATH> <BUILD SPEC>
```

Example:

```
cd stages
apptainer build \
    --build-arg ENVIRONMENT="stage-1-vmex" \
    stage-1-vmex.sif \
    apptainer.def
```

### Pulling Apptainer container images from a registry

```
apptainer pull [pull options...] [output file] <URI>
```

where for OCI container registries, the `<URI>` follows

```
oras://<registry>/<repository>/<image>:<tag>
```

Example:

```
apptainer pull stage-1-vmex-cpu.sif oras://ghcr.io/driftless-star/driftless-star:apptainer-stage-1-vmex-cpu
```

### Running Apptainer container images

```
apptainer run [run options...] <container> [args...]
```

Examples:

* Run a `stage-1-vmex-cpu` container in an interactive shell

```console
$ apptainer run --containall --writable-tmpfs ./stage-1-vmex-cpu.sif
(driftless-star-stages:stage-1-vmex)
```

* Run a `stage-1-vmex-cpu` container in an interactive shell with the local working directory mounted

```console
$ apptainer run --containall --writable-tmpfs --bind "$PWD":/work --pwd /work ./stage-1-vmex-cpu.sif
(driftless-star-stages:stage-1-vmex) pwd
/work
(driftless-star-stages:stage-1-vmex)
```

* Execute a command in a `stage-1-vmex-cpu` container

```console
$ apptainer run --containall --writable-tmpfs ./stage-1-vmex-cpu.sif python -c 'import vmex; print(vmex.__name__)'
vmex
```

* Run a `stage-1-vmex-gpu` container in an interactive shell with [NVIDIA driver support](https://apptainer.org/docs/user/latest/gpu.html)

```console
$ apptainer run --containall --writable-tmpfs --nv ./stage-1-vmex-gpu.sif
(driftless-star-stages:stage-1-vmex-gpu) nvidia-smi --version
NVIDIA-SMI version  : 590.48.01
NVML version        : 590.48
DRIVER version      : 590.48.01
CUDA Version        : 13.1
```
