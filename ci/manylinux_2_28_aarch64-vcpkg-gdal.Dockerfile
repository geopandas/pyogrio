FROM quay.io/pypa/manylinux_2_28_aarch64:2025.09.19-1

# building openssl needs IPC-Cmd (https://github.com/microsoft/vcpkg/issues/24988)
RUN dnf -y install curl zip unzip tar ninja-build perl-IPC-Cmd

RUN git clone https://github.com/Microsoft/vcpkg.git /usr/local/share/vcpkg && \
    git -C /usr/local/share/vcpkg checkout da096fdc67db437bee863ae73c4c12e289f82789

ENV VCPKG_INSTALLATION_ROOT="/usr/local/share/vcpkg"
ENV PATH="${PATH}:/usr/local/share/vcpkg"

ENV VCPKG_DEFAULT_TRIPLET="arm64-linux-dynamic-release"
# pkgconf fails to build with default debug mode of arm64-linux host
ENV VCPKG_DEFAULT_HOST_TRIPLET="arm64-linux-release"

# Must be set when building on arm
ENV VCPKG_FORCE_SYSTEM_BINARIES=1

# mkdir & touch -> workaround for https://github.com/microsoft/vcpkg/issues/27786
RUN bootstrap-vcpkg.sh && \
    mkdir -p /root/.vcpkg/ $HOME/.vcpkg && \
    touch /root/.vcpkg/vcpkg.path.txt $HOME/.vcpkg/vcpkg.path.txt && \
    vcpkg integrate install && \
    vcpkg integrate bash

COPY ci/custom-triplets/arm64-linux-dynamic-release.cmake /usr/local/share/vcpkg/custom-triplets/arm64-linux-dynamic-release.cmake
COPY ci/vcpkg.json /usr/local/share/vcpkg/

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/share/vcpkg/installed/arm64-linux-dynamic-release/lib"
RUN vcpkg install --overlay-triplets=/usr/local/share/vcpkg/custom-triplets \
    --feature-flags="versions,manifests" \
    --x-manifest-root=/usr/local/share/vcpkg/ \
    --x-install-root=/usr/local/share/vcpkg/installed && \
    vcpkg list

# setting git safe directory is required for properly building wheels when
# git >= 2.35.3
RUN git config --global --add safe.directory "*"
