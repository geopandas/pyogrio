set(VERSION 1.2.5.2)

vcpkg_download_distfile(ARCHIVE_FILE
    URLS "http://zlib.net/fossils/zlib-1.2.5.2.tar.gz"
    FILENAME "zlib-1.2.5.2.tar.gz"
    SHA512 d4bd29ebfd5642253cecb9b8364ee6de87442d192229a9080cc306b819745e80c0791bd0a8abefd0c5e11c958bc85485d5d5d051b4770e45f6f479f3bb16e867
)

vcpkg_extract_source_archive_ex(
    OUT_SOURCE_PATH SOURCE_PATH
    ARCHIVE ${ARCHIVE_FILE}
    REF ${VERSION}
    PATCHES
        "0002-skip-building-examples.patch"
)

# This is generated during the cmake build
file(REMOVE ${SOURCE_PATH}/zconf.h)

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    OPTIONS
        -DSKIP_INSTALL_FILES=ON
    OPTIONS_DEBUG
        -DSKIP_INSTALL_HEADERS=ON
)

vcpkg_install_cmake()
file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/vcpkg-cmake-wrapper.cmake DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT})

vcpkg_fixup_pkgconfig()

file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

vcpkg_copy_pdbs()

file(COPY ${CMAKE_CURRENT_LIST_DIR}/usage DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT})
