# Building from source
To build and install FlexFlow, follow the instructions below.

## 1. Download the source code
Clone the FlexFlow source code, and the third-party dependencies from GitHub.
```
git clone --recursive https://github.com/flexflow/FlexFlow.git
```

## 2. Install system dependencies
FlexFlow has system dependencies on cuda and/or rocm depending on which gpu backend you target. The gpu backend is configured by the cmake variable `FF_GPU_BACKEND`. By default, FlexFlow targets CUDA. `docker/flexflow-environment/Dockerfile` installs system dependencies in a standard ubuntu system.

### Targeting CUDA
If you are targeting CUDA, FlexFlow requires CUDA and CUDNN to be installed. You can follow the standard nvidia installation instructions [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CUDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

Disclaimer: CUDA architectures < 60 (Maxwell and older) are no longer supported.

### Targeting ROCM
If you are targeting ROCM, FlexFlow requires a ROCM and HIP installation with a few additional packages. Note that this can be done on a system with or without an AMD GPU. You can follow the standard installation instructions [ROCM](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.3/page/Introduction_to_ROCm_Installation_Guide_for_Linux.html) and [HIP](https://docs.amd.com/bundle/HIP-Installation-Guide-v5.3/page/Introduction_to_HIP_Installation_Guide.html). When running `amdgpu-install`, install the use cases hip and rocm. You can avoid installing the kernel drivers (not necessary on systems without an AMD graphics card) with `--no-dkms` I.e. `amdgpu-install --usecase=hip,rocm --no-dkms`. Additionally, install the packages `hip-dev`, `hipblas`, `miopen-hip`, and `rocm-hip-sdk`.

See `./docker/flexflow-environment/Dockerfile` for an example ROCM install.

## 3. Install the Python dependencies
You can install the python dependencies by running:
```
pip install -r requirements.txt
```

Alternatively, if using conda, you can create new environment and automatically install all dependencies with:
```
conda env create -f conda/flexflow.yml
conda activate flexflow
```

## 4. Configure and Build FlexFlow
After having installed the dependencies, you can configure and build FlexFlow with the commands below. If you do not want to use the default build options, you will need to set the relevant environment variables before running `config/config.linux`. We recommend that you spend some time familiarizing with the available options by scanning the `config/config.linux` file.

<details>
<summary>Expand here to see the configuration options</summary>
<br>

The main parameters are:

1. `FF_GPU_BACKEND`: specifies the target hardware (`cuda` or `hip_rocm`) 
2. `CUDA_DIR` is used to specify the directory of CUDA. It is only required when CMake can not automatically detect the installation directory of CUDA.
3. `CUDNN_DIR` is used to specify the directory of CUDNN. It is only required when CUDNN is not installed in the CUDA directory.
4. `FF_CUDA_ARCH` is used to set the architecture of targeted GPUs, for example, the value can be 60 if the GPU architecture is Pascal. To build for more than one architecture, pass a list of comma separated values (e.g. `FF_CUDA_ARCH=70,75`). To compile FlexFlow for all GPU architectures that are detected on the machine, pass `FF_CUDA_ARCH=autodetect` (this is the default value, so you can also leave `FF_CUDA_ARCH` unset. If you want to build for all GPU architectures compatible with FlexFlow, pass `FF_CUDA_ARCH=all`. **If your machine does not have any GPU, you have to set FF_CUDA_ARCH to at least one valid architecture code (or `all`)**, since the compiler won't be able to detect the architecture(s) automatically.
5. `FF_USE_NCCL` controls whether to build FlexFlow with NCCL support. By default, it is set to ON.
6. `FF_LEGION_NETWORKS` is used to enable distributed run of FlexFlow. If you want to run FlexFlow on multiple nodes, follow instructions in the [Multinode tutorial](https://flexflow.readthedocs.io/en/latest/multinode.html) and set the corresponding parameters as follows:
* To build FlexFlow with GASNet, set `FF_LEGION_NETWORKS=gasnet` and `FF_GASNET_CONDUIT` as a specific conduit (e.g. `ibv`, `mpi`, `udp`, `ucx`) in `config/config.linux` when configuring the FlexFlow build. Set `FF_UCX_URL` when you want to customize the URL to download UCX.
* To build FlexFlow with native UCX, set `FF_LEGION_NETWORKS=ucx` in `config/config.linux` when configuring the FlexFlow build. Set `FF_UCX_URL` when you want to customize the URL to download UCX.
7. `FF_MAX_DIM` is used to set the maximum dimension of tensors, by default it is set to 4. 

More options are available in cmake, please run `ccmake` and search for options starting with FF. 

</details>

To configure and build run:

```
mkdir build
cd build
../config/config.linux
make -j N
```
where N is the desired number of threads to use for the build.

## 5. Install FlexFlow

If you wish to install FlexFlow, you can do so by running:
```
make install
```

This will allow you to run all the code from any folder, and without having to set any additional environment variable (e.g. `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`).

