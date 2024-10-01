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
errors
known_issues
```
