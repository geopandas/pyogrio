cmake_minimum_required(VERSION 3.15...4.4)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(Cython MODULE REQUIRED VERSION 3.1)
include(UseCython)

find_package(GDAL CONFIG REQUIRED)
message(STATUS "Found GDAL: ${GDAL_VERSION}")

set(CYTHON_EXTENSIONS
    _err
    _geometry
    _io
    _ogr
    _vsi)

foreach(module ${CYTHON_EXTENSIONS})
  cython_transpile(
    ${SKBUILD_PROJECT_NAME}/${module}.pyx
    LANGUAGE C
    INCLUDE_DIRECTORIES "./${SKBUILD_PROJECT_NAME}"
    OUTPUT_VARIABLE ${module}_c
    CYTHON_ARGS "-E" "CTE_GDAL_MAJOR=${GDAL_VERSION_MAJOR},CTE_GDAL_MINOR=${GDAL_VERSION_MINOR},CTE_GDAL_PATCH=${GDAL_VERSION_PATCH}"
  )
  python_add_library(${module} MODULE "${${module}_c}" WITH_SOABI)
  target_link_libraries(${module} PRIVATE GDAL::GDAL)
  target_include_directories(${module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME}")
endforeach()

install(TARGETS ${CYTHON_EXTENSIONS} DESTINATION ${SKBUILD_PROJECT_NAME})
