set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_BUILD_TYPE release)
set(VCPKG_OSX_DEPLOYMENT_TARGET "12.0")

set(VCPKG_LIBRARY_LINKAGE static)
if(PORT MATCHES "gdal")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
else()
    # Keep the symbols of the static libraries linked into libgdal private, so they
    # cannot clash with other copies of these libraries loaded in the same process,
    # such as those bundled in rasterio's wheels (macOS merges weak C++ symbols
    # across all loaded libraries).
    set(VCPKG_C_FLAGS "-fvisibility=hidden")
    set(VCPKG_CXX_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden")
endif()
