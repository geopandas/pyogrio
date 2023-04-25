set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# avoid using debug build for core tools like vcpkg-tool-meson and pkgconf
set(VCPKG_BUILD_TYPE release)