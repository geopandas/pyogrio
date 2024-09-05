# pyogrio - Vectorized spatial vector file format I/O using GDAL/OGR

Pyogrio provides a
[GeoPandas](https://github.com/geopandas/geopandas)-oriented API to OGR vector
data sources, such as ESRI Shapefile, GeoPackage, and GeoJSON. Vector data sources
have geometries, such as points, lines, or polygons, and associated records
with potentially many columns worth of data.

Pyogrio uses a vectorized approach for reading and writing GeoDataFrames to and
from OGR vector data sources in order to give you faster interoperability. It
uses pre-compiled bindings for GDAL/OGR so that the performance is primarily
limited by the underlying I/O speed of data source drivers in GDAL/OGR rather
than multiple steps of converting to and from Python data types within Python.

We have seen \>5-10x speedups reading files and \>5-20x speedups writing files
compared to using non-vectorized approaches (Fiona and current I/O support in
GeoPandas).

You can read these data sources into
`GeoDataFrames`, read just the non-geometry columns into Pandas `DataFrames`,
or even read non-spatial data sources that exist alongside vector data sources,
such as tables in a ESRI File Geodatabase, or antiquated DBF files.

Pyogrio also enables you to write `GeoDataFrames` to at least a few different
OGR vector data source formats.

Read the documentation for more information:
[https://pyogrio.readthedocs.io](https://pyogrio.readthedocs.io/en/latest/).

WARNING: Pyogrio is still at an early version and the API is subject to
substantial change. Please see [CHANGES](CHANGES.md).

## Requirements

Supports Python 3.9 - 3.13 and GDAL 3.4.x - 3.9.x.

Reading to GeoDataFrames requires `geopandas>=0.12` with `shapely>=2`.

Additionally, installing `pyarrow` in combination with GDAL 3.6+ enables
a further speed-up when specifying `use_arrow=True`.

## Installation

Pyogrio is currently available on
[conda-forge](https://anaconda.org/conda-forge/pyogrio)
and [PyPI](https://pypi.org/project/pyogrio/)
for Linux, MacOS, and Windows.

Please read the
[installation documentation](https://pyogrio.readthedocs.io/en/latest/install.html)
for more information.

## Supported vector formats

Pyogrio supports some of the most common vector data source formats (provided
they are also supported by GDAL/OGR), including ESRI Shapefile, GeoPackage,
GeoJSON, and FlatGeobuf.

Please see the [list of supported formats](https://pyogrio.readthedocs.io/en/latest/supported_formats.html)
for more information.

## Getting started

Please read the [introduction](https://pyogrio.readthedocs.io/en/latest/supported_formats.html)
for more information and examples to get started using Pyogrio.

You can also check out the the [API documentation](https://pyogrio.readthedocs.io/en/latest/api.html)
for full details on using the API.

## Credits

This project is made possible by the tremendous efforts of the GDAL, Fiona, and
Geopandas communities.

-   Core I/O methods and supporting functions adapted from [Fiona](https://github.com/Toblerity/Fiona)
-   Inspired by [Fiona PR](https://github.com/Toblerity/Fiona/pull/540/files)
