# About

## How it works

The "standard" mode in Pyogrio uses a numpy-oriented approach in Cython to read
information about data sources and records from spatial data layers. Geometries
are extracted from the data layer as Well-Known Binary (WKB) objects and fields
(attributes) are read into numpy arrays of the appropriate data type. These are
then converted to a GeoPandas `GeoDataFrame`.

When the "Arrow" mode is used, (`use_arrow=True`), Pyogrio uses the Arrow Stream
interface of GDAL, which reads the data to the
[Apache Arrow](https://arrow.apache.org/) memory format. After reading the data,
Pyogrio converts the data to a `GeoDataFrame`. Because this code path is even
more optimized, also in GDAL, using `use_arrow=True` can give a significant
performance boost, especially when reading large files.

All records are read into memory in bulk. This is very fast, but can give memory
issues when reading very large data sources. To solve this, Pyogrio exposes
several options offered by GDAL to filter the data while being read.
Some examples are a filter on a `bbox`, use `skip_features` / `max_features`,
using a `sql` statement, etc. The performance of the filtering depends on the
file format being read, e.g. the availability of (spatial) indexes, etc.

When writing, the entire `GeoDataFrame` is written at once, but it is possible
to append data to an existing data source.

## Comparison to Fiona

[Fiona](https://github.com/Toblerity/Fiona) is a full-featured, general-purpose
Python library for working with OGR vector data sources. It is **awesome**, has
highly-dedicated maintainers and contributors, and exposes more functionality than
Pyogrio ever will. Finally it is used in many projects in the Python ecosystem.

In contrast, Pyogrio specifically targets the typical needs of GeoPandas. It uses
a stateless approach, so all data are read or written in a single pass. This
bulk-oriented approach enables significantly faster I/O operations, especially
for larger datasets.

Pyogrio borrows from the internal mechanics and lessons
learned of Fiona and so this project would not have been possible without Fiona
having come first.
