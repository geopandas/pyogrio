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

```{warning}
Pyogrio is still at an early version and the API is subject to substantial change.
```

```{toctree}
---
maxdepth: 2
caption: Contents
---

about
concepts
supported_formats
install
introduction
api
known_issues
```
