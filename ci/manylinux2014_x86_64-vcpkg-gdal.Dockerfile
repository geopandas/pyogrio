FROM quay.io/pypa/manylinux2014_x86_64:2026.06.03-1

# Additional system dependencies:
# - vcpkg needs: curl zip unzip tar ninja
# - openssl needs IPC-Cmd and perl-core (https://github.com/microsoft/vcpkg/issues/24988, https://github.com/openssl/openssl/issues/28579)
# - libspatialite needs full autotools suite to build (autoconf autoconf-archive automake libtool)
RUN yum install -y curl unzip zip tar perl-core perl-IPC-Cmd autoconf autoconf-archive automake libtool

# require python >= 3.7 (python 3.6 is default on base image) for meson
RUN ln -s /opt/python/cp312-cp312/bin/python3 /usr/bin/python3

# vcpkg currently requires ninja >= 1.13.2; build it from source so the binary
# is compatible with the older manylinux2014 runtime libraries.
# (vcpkg otherwise downloads a pre-built binary of ninja, which fails to run due to missing symbols in the older runtime)
RUN curl -L -o /tmp/ninja-1.13.2.tar.gz https://github.com/ninja-build/ninja/archive/refs/tags/v1.13.2.tar.gz && \
    tar -xzf /tmp/ninja-1.13.2.tar.gz -C /tmp && \
    cd /tmp/ninja-1.13.2 && \
    python3 configure.py --bootstrap && \
    install -m 0755 ninja /usr/local/bin/ninja && \
    /usr/local/bin/ninja --version && \
    rm -rf /tmp/ninja-1.13.2 /tmp/ninja-1.13.2.tar.gz

ARG VCPKG_GDAL_COMMIT
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg && \
    git -C /opt/vcpkg checkout ${VCPKG_GDAL_COMMIT}

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
