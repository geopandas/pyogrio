# pyogrio - Vectorized spatial vector file format I/O using GDAL/OGR

Pyogrio provides a
[GeoPandas](https://github.com/geopandas/geopandas)-oriented API to OGR vector
data sources, such as ESRI Shapefile, GeoPackage, and GeoJSON. This converts to
/ from `geopandas.GeoDataFrame`s when the data source includes geometry and
`pandas.DataFrame`s otherwise.

Pyogrio uses a vectorized approach for reading and writing DataFrames, resulting in
\>5-10x speedups reading files and \>5-20x speedups writing files compared to using
non-vectorized approaches (Fiona and current I/O support in GeoPandas).



```{warning}
This is an early version and the API is subject to substantial change.
```

```{toctree}
---
maxdepth: 2
caption: Contents
---

install
api
```
