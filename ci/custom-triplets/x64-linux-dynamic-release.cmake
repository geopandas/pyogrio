set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

set(VCPKG_LIBRARY_LINKAGE static)
if(PORT MATCHES "gdal")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
else()
    # Keep the symbols of the static libraries linked into libgdal private: pyogrio
    # only calls GDAL's own API, and they could clash with other copies of these
    # libraries loaded in the same process.
    set(VCPKG_C_FLAGS "-fvisibility=hidden")
    set(VCPKG_CXX_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden")
endif()
