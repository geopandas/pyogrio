cmake_minimum_required(VERSION 3.25)
project(minizip LANGUAGES C)

set(SRC
    ioapi.c
    unzip.c
    zip.c
    mztools.c
)
set(HEADERS
    crypt.h
    ioapi.h
    unzip.h
    zip.h
    mztools.h
)

add_library(minizip ${SRC})
target_include_directories(minizip PUBLIC $<INSTALL_INTERFACE:include/minizip>)

find_package(ZLIB REQUIRED)
target_compile_definitions(minizip PRIVATE -D_ZLIB_H)
target_link_libraries(minizip PUBLIC ZLIB::ZLIB)
set(MINIZIP_REQUIRES zlib)

if(CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    target_compile_definitions(minizip PRIVATE -DIOWIN32_USING_WINRT_API)
endif()

install(
    TARGETS minizip
    EXPORT unofficial-minizipTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/unofficial-minizipConfigVersion.cmake"
    VERSION "${PACKAGE_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(unofficial-minizipConfig.cmake.in
    unofficial-minizipConfig.cmake
    INSTALL_DESTINATION share/unofficial-minizip
)

install(FILES
    "${PROJECT_BINARY_DIR}/unofficial-minizipConfig.cmake"
    "${PROJECT_BINARY_DIR}/unofficial-minizipConfigVersion.cmake"
    DESTINATION share/unofficial-minizip
)

install(EXPORT unofficial-minizipTargets
    NAMESPACE unofficial::minizip::
    DESTINATION share/unofficial-minizip
)

if(NOT DISABLE_INSTALL_HEADERS)
    install(FILES ${HEADERS} DESTINATION include/minizip)
endif()

list(JOIN MINIZIP_REQUIRES " " MINIZIP_REQUIRES)
set(exec_prefix "\${prefix}")
set(libdir "\${prefix}/lib")
set(includedir "\${prefix}/include")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/minizip.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/minizip.pc"
    @ONLY
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/minizip.pc"
    DESTINATION lib/pkgconfig
)