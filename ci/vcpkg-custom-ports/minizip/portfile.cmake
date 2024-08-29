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
)

configure_file("${CMAKE_CURRENT_LIST_DIR}/minizipConfig.cmake.in" "${SOURCE_PATH}/cmake/minizipConfig.cmake.in" COPYONLY)
configure_file("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" "${SOURCE_PATH}/CMakeLists.txt" COPYONLY)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
    -DDISABLE_INSTALL_TOOLS=${VCPKG_TARGET_IS_IOS}
    OPTIONS_DEBUG
    -DDISABLE_INSTALL_HEADERS=ON
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup()
vcpkg_copy_tool_dependencies("${CURRENT_PACKAGES_DIR}/tools/minizip")

file(COPY "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${SOURCE_PATH}/contrib/minizip/MiniZip64_info.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)