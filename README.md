# pyogrio - bulk-oriented spatial vector file I/O using GDAL/OGR

Pyogrio provides fast, bulk-oriented read and write access to 
[GDAL/OGR](https://gdal.org/en/latest/drivers/vector/index.html) vector data
sources, such as ESRI Shapefile, GeoPackage, GeoJSON, and several others.
Vector data sources typically have geometries, such as points, lines, or
polygons, and associated records with potentially many columns worth of data.

The typical use is to read or write these data sources to/from
[GeoPandas](https://github.com/geopandas/geopandas) `GeoDataFrames`. Because
the geometry column is optional, reading or writing only non-spatial data is
also possible. Hence, GeoPackage attribute tables, DBF files, or CSV files are
also supported.

Pyogrio is fast because it uses pre-compiled bindings for GDAL/OGR to read and
write the data records in bulk. This approach avoids multiple steps of
converting to and from Python data types within Python, so performance becomes
primarily limited by the underlying I/O speed of data source drivers in
GDAL/OGR.

We have seen \>5-10x speedups reading files and \>5-20x speedups writing files
compared to using row-per-row approaches (e.g. Fiona).

Read the documentation for more information:
[https://pyogrio.readthedocs.io](https://pyogrio.readthedocs.io/en/latest/).

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

Pyogrio supports most common vector data source formats (provided they are also
supported by GDAL/OGR), including ESRI Shapefile, GeoPackage, GeoJSON, and
FlatGeobuf.

Please see the [list of supported formats](https://pyogrio.readthedocs.io/en/latest/supported_formats.html)
for more information.

## Getting started

Please read the [introduction](https://pyogrio.readthedocs.io/en/latest/supported_formats.html)
for more information and examples to get started using Pyogrio.

You can also check out the [API documentation](https://pyogrio.readthedocs.io/en/latest/api.html)
for full details on using the API.

## Credits

This project is made possible by the tremendous efforts of the GDAL, Fiona, and
Geopandas communities.

-   Core I/O methods and supporting functions adapted from [Fiona](https://github.com/Toblerity/Fiona)
-   Inspired by [Fiona PR](https://github.com/Toblerity/Fiona/pull/540/files)
