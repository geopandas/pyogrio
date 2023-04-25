FROM quay.io/pypa/manylinux_2_28_aarch64:latest

# building openssl needs IPC-Cmd (https://github.com/microsoft/vcpkg/issues/24988)
RUN dnf -y install curl zip unzip tar ninja-build perl-IPC-Cmd elfutils-devel

# require python >= 3.7 (python 3.6 is default on base image) for meson 
RUN ln -s /opt/python/cp38-cp38/bin/python3 /usr/bin/python3

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

# Must be set when building on arm
ENV VCPKG_FORCE_SYSTEM_BINARIES=1

# mkdir & touch -> workaround for https://github.com/microsoft/vcpkg/issues/27786
RUN bootstrap-vcpkg.sh && \
    mkdir -p /root/.vcpkg/ $HOME/.vcpkg && \
    touch /root/.vcpkg/vcpkg.path.txt $HOME/.vcpkg/vcpkg.path.txt && \
    vcpkg integrate install && \
    vcpkg integrate bash

COPY ci/custom-triplets/arm64-linux-dynamic-release.cmake opt/vcpkg/custom-triplets/arm64-linux-dynamic-release.cmake
COPY ci/custom-triplets/arm64-linux.cmake opt/vcpkg/custom-triplets/arm64-linux.cmake
COPY ci/vcpkg-custom-ports/ opt/vcpkg/custom-ports/
COPY ci/vcpkg.json opt/vcpkg/

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/vcpkg/installed/arm64-linux-dynamic-release/lib"
RUN vcpkg install --overlay-triplets=opt/vcpkg/custom-triplets \
    --triplet=arm64-linux-dynamic-release \
    --overlay-ports=opt/vcpkg/custom-ports \
    --feature-flags="versions,manifests" \
    --x-manifest-root=opt/vcpkg \
    --x-install-root=opt/vcpkg/installed && \
    vcpkg list

# setting git safe directory is required for properly building wheels when
# git >= 2.35.3
RUN git config --global --add safe.directory "*"
