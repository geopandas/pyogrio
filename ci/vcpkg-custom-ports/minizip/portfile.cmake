# When this port is updated, the minizip port should be updated at the same time

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO madler/zlib
    REF v1.2.5.2
    SHA512 8b9a7b27c744d9347da308257ba39eb09bfc012dccfcccc6a6b31aa810f4f88655f028b03358abd610559fbff264c3af5fbb4a079495cbec6c945bbc0c2e504a
    HEAD_REF master
    PATCHES
    0001-remove-ifndef-NOUNCRYPT.patch
    0002-add-declaration-for-mkdir.patch
    pkgconfig.patch
)

# When zlib updated, the minizip port should be updated at the same time
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO madler/zlib
    REF v1.2.5.2
    SHA512 8b9a7b27c744d9347da308257ba39eb09bfc012dccfcccc6a6b31aa810f4f88655f028b03358abd610559fbff264c3af5fbb4a079495cbec6c945bbc0c2e504a
    HEAD_REF master
    PATCHES
    0001-remove-ifndef-NOUNCRYPT.patch
    0002-add-declaration-for-mkdir.patch
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_LIST_DIR}/minizip-win32.def"
    "${CMAKE_CURRENT_LIST_DIR}/unofficial-minizipConfig.cmake.in"
    DESTINATION "${SOURCE_PATH}/contrib/minizip"
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/contrib/minizip"
    OPTIONS
    ${FEATURE_OPTIONS}
    -DPACKAGE_VERSION=${VERSION}
    OPTIONS_DEBUG
    -DDISABLE_INSTALL_HEADERS=ON
    -DDISABLE_INSTALL_TOOLS=ON
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME unofficial-minizip)
vcpkg_fixup_pkgconfig()

configure_file("${CMAKE_CURRENT_LIST_DIR}/minizipConfig.cmake.in" "${CURRENT_PACKAGES_DIR}/share/${PORT}/minizipConfig.cmake" @ONLY)
file(COPY "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/contrib/minizip/MiniZip64_info.txt")
