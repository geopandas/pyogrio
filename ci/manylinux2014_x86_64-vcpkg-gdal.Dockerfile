FROM quay.io/pypa/manylinux2014_x86_64:2025.09.19-1

# building openssl needs IPC-Cmd (https://github.com/microsoft/vcpkg/issues/24988)
RUN yum install -y curl unzip zip tar perl-IPC-Cmd

# require python >= 3.7 (python 3.6 is default on base image) for meson
RUN ln -s /opt/python/cp38-cp38/bin/python3 /usr/bin/python3

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg && \
    git -C /opt/vcpkg checkout da096fdc67db437bee863ae73c4c12e289f82789

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

ENV VCPKG_DEFAULT_TRIPLET="x64-linux-dynamic-release"

# mkdir & touch -> workaround for https://github.com/microsoft/vcpkg/issues/27786
RUN bootstrap-vcpkg.sh && \
    mkdir -p /root/.vcpkg/ $HOME/.vcpkg && \
    touch /root/.vcpkg/vcpkg.path.txt $HOME/.vcpkg/vcpkg.path.txt && \
    vcpkg integrate install && \
    vcpkg integrate bash

COPY ci/custom-triplets/x64-linux-dynamic-release.cmake opt/vcpkg/custom-triplets/x64-linux-dynamic-release.cmake
COPY ci/vcpkg-custom-ports/ opt/vcpkg/custom-ports/
COPY ci/vcpkg-manylinux2014.json opt/vcpkg/vcpkg.json

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/vcpkg/installed/x64-linux-dynamic-release/lib"
RUN vcpkg install --overlay-triplets=opt/vcpkg/custom-triplets \
    --overlay-ports=opt/vcpkg/custom-ports \
    --feature-flags="versions,manifests" \
    --x-manifest-root=opt/vcpkg \
    --x-install-root=opt/vcpkg/installed && \
    vcpkg list

# setting git safe directory is required for properly building wheels when
# git >= 2.35.3
RUN git config --global --add safe.directory "*"
