FROM quay.io/pypa/manylinux2014_x86_64:2022-04-03-da6ecb3

RUN yum install -y curl unzip zip tar python3

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg

ENV VCPKG_INSTALLATION_ROOT="/opt/vcpkg"
ENV PATH="${PATH}:/opt/vcpkg"

RUN bootstrap-vcpkg.sh && \
    vcpkg integrate install && \
    vcpkg integrate bash

COPY ci/custom-triplets/x64-linux-dynamic.cmake opt/vcpkg/custom-triplets/x64-linux-dynamic.cmake
COPY ci/vcpkg-custom-ports/ opt/vcpkg/custom-ports/
COPY ci/vcpkg.json opt/vcpkg/

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/vcpkg/installed/x64-linux-dynamic/lib"
RUN vcpkg install --overlay-triplets=opt/vcpkg/custom-triplets \
    --triplet=x64-linux-dynamic \
    --overlay-ports=opt/vcpkg/custom-ports \
    --feature-flags="versions,manifests" \
    --x-manifest-root=opt/vcpkg \
    --x-install-root=opt/vcpkg/installed && \
    vcpkg list
