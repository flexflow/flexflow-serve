ARG FF_GPU_BACKEND=cuda
ARG gpu_backend_version=12.1
FROM flexflow-environment-$FF_GPU_BACKEND$gpu_backend_version:latest

LABEL org.opencontainers.image.source=https://github.com/flexflow/flexflow-serve
LABEL org.opencontainers.image.description="flexflow-serve container"

# Copy flexflow-serve repository
RUN mkdir flexflow-serve
ENV FF_HOME=/usr/flexflow-serve
WORKDIR ${FF_HOME}
COPY . .

# Args to build flexflow-serve
ARG BUILD_CONFIGS
ARG N_BUILD_CORES

# Create install directory if needed
RUN for pair in $BUILD_CONFIGS; do \
        key=${pair%%=*}; \
        value=${pair#*=}; \
        if [ "$key" = "INSTALL_DIR" ] && [ -n "$value" ]; then \
            mkdir -p "$value"; \
        fi; \
    done

# Build and install C++ and Python versions of flexflow-serve
RUN mkdir -p build && cd build && \
    eval "$BUILD_CONFIGS" ../config/config.linux && \
    make -j $N_BUILD_CORES install && \
    ldconfig

ENTRYPOINT ["/bin/bash"]
